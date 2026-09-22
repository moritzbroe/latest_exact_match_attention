"""Inference: a LEMA head IS a key-value store, so run it as one.

During training, attention is a dense O(T^2) kernel. At inference nothing about a LEMA head
requires that: its entire state after position i is the map

    code -> value of the most recent position that emitted that code

so generating a token costs one lookup and one insert per (layer, head), independent of how
long the context is, and the state holds one entry per DISTINCT code rather than one per
token.

Getting the ASYMPTOTICS right is easy; getting the CONSTANT right is the whole engineering
problem, because a decode step touches depth x heads stores and a large model has thousands
of them. Three things make that affordable here:

  * ONE hash table for the whole model, shared by every (layer, stream, head): all heads of
    a layer are read and written by a single C++ pass, and memory is provisioned once for
    the total number of distinct codes rather than per head;
  * codes cross the GPU boundary once per layer, not once per head;
  * a step of T positions is one call that looks each query up and inserts each key in
    position order -- the LEMA rule itself -- so a prefill chunk and a decode step are the
    same operation and there is no in-chunk matching to do on the GPU.

The store is `HashStore`: open addressing in host RAM, PRESIZED to a memory budget and
never resized; see `HashStore` for why.

Nothing here knows about the model: `Transformer.forward(x, cache=...)` hands one
`LayerStore` to each layer, and the attention op only ever asks it to exchange codes for
values. Feeding a sequence in any chunking -- including one token at a time -- reproduces
the single-shot forward exactly.
"""
from __future__ import annotations

import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


EMPTY = np.uint64(0)      # state word of a free slot. Zero, deliberately: a zeroed
# allocation is then already an empty table, so no init pass ever streams over the
# (possibly huge) array. The state lives in the slot's second word, next to the code, so
# every 64-bit code is an ordinary key (csrc/store.cpp).

_EXT = None


def _pick_cxx():
    """Find a g++ torch accepts (>= 9) if CXX is not already set."""
    if os.environ.get("CXX"):
        return
    import subprocess
    cands = ["g++"] + sorted(glob.glob(str(Path(sys.prefix).parent / "*/bin/*-g++")))
    for c in cands:
        try:
            v = subprocess.run([c, "-dumpversion"], capture_output=True, text=True).stdout
            if int(v.strip().split(".")[0]) >= 9:
                os.environ["CXX"] = c
                os.environ.setdefault("CC", c.replace("g++", "gcc"))
                return
        except Exception:
            continue


def _ext():
    """The C++ store, JIT-built on first use.

    Probing is the one thing numpy cannot do well: an exchange is a chain of dependent
    random accesses per partition, which vectorised gather/scatter passes cannot express
    and per-element python cannot afford. One C++ loop with software prefetch does it at
    the memory system's speed.

    Needs a C++17 compiler that torch accepts (GCC >= 9). Set CXX if the default is older.
    """
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _pick_cxx()
        try:
            # -fopenmp is not optional: at::parallel_for in an extension is a plain
            # sequential loop unless the extension itself is built with OpenMP
            _EXT = load(name="lema_store",
                        sources=[str(Path(__file__).with_name("csrc") / "store.cpp")],
                        extra_cflags=["-O3", "-funroll-loops", "-fopenmp"],
                        extra_ldflags=["-fopenmp"], verbose=False)
        except Exception as e:      # a bad toolchain is the common case; say so plainly
            raise RuntimeError(
                f"could not build the C++ store ({type(e).__name__}: {e}).\nIt needs a "
                f"C++17 compiler torch accepts (GCC >= 9) and ninja; set CXX to one if the system default "
                f"is older.") from e
    return _EXT


# The table holds exactly what the model produced -- no more precision and no less. numpy
# has no bfloat16, so a bf16 table is carried as raw uint16 bit patterns; that is sound here
# because nothing outside the model ever INTERPRETS a value. The store copies rows of bytes,
# and only torch, which does know bfloat16, ever reads one as a number.
_CARRIER = {torch.float32: np.float32, torch.float16: np.float16,
            torch.bfloat16: np.uint16}


def _lock(t: torch.Tensor):
    """Pin a host tensor's pages in RAM (mlock). A table that fills most of the machine's
    memory is exactly what the kernel likes to swap out, and a swapped table turns random
    probes into disk reads: measured, a 50 GB table on a 62 GB machine had ~3 GB paged
    out and spent its first 50 seconds faulting it back in. Locking makes the table's
    residency a fact rather than a hope; if the lock limit (ulimit -l) forbids it, say so
    and carry on unlocked -- or refuse, when LEMA_REQUIRE_MLOCK=1 is set, as the
    benchmarks do: a curve measured on a partially swapped table is worthless."""
    import ctypes
    import warnings
    if not hasattr(ctypes, "CDLL") or sys.platform != "linux":
        return
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.mlock(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(t.numel() * t.element_size())):
        msg = (f"could not mlock the {t.numel() * t.element_size() / 1e9:.1f} GB store "
               f"table (errno {ctypes.get_errno()}); it may get swapped out. Raise "
               f"`ulimit -l` or use a smaller budget.")
        if os.environ.get("LEMA_REQUIRE_MLOCK") == "1":
            raise RuntimeError(msg)
        warnings.warn(msg)


# ------------------------------------------------------------------- the store
class HashStore:
    """ONE open-addressing table (linear probing, fixed capacity) for every (layer, stream,
    head) partition of a model; an entry's key is the pair (group, code).

    Sharing is the point. Heads differ enormously in how many distinct codes they emit, so
    with a table per head the memory would have to be provisioned for the busiest head
    times the number of heads; here a busy head just takes more slots and the table fills
    at the TOTAL number of distinct codes. Occupancy, and with it the probe length, is one
    number for the whole model.

    `capacity` is slots, exact -- not rounded to a power of two -- so the table can be sized
    to a memory budget (the slot index is a multiply, not a mask; see csrc/store.cpp).

    The table does not grow. Growth would restore "memory proportional to distinct codes",
    but it costs a full rehash inside one token's decode step. Constant time per token is
    the claim, so capacity is provisioned instead, exactly as the softmax baseline
    provisions its kv cache; overrunning it raises. The whole reservation is touched here,
    at provisioning time, on purpose: left lazy, the kernel zero-fills each page on first
    use, and since numpy asks for transparent huge pages a first touch is a 2 MB memset --
    random inserts reach every huge page within the first ~50 decoded tokens, which used
    to charge the allocation to the first tokens of every generation curve as a 3x spike.
    """

    def __init__(self, d_v: int, groups: int, dtype=np.float16, capacity: int = 256):
        # The store copies values, never computes on them, so the C++ side is told a row's
        # size in BYTES and the dtype is only a carrier. uint16 is how bfloat16 is held,
        # since numpy cannot name it.
        if np.dtype(dtype) not in (np.float16, np.float32, np.uint16):
            raise ValueError(f"store dtype must be float16, float32, or uint16 (the "
                             f"carrier for bfloat16), got {dtype}")
        self.d_v, self.dtype, self.groups = d_v, dtype, groups
        self.vbytes = d_v * np.dtype(dtype).itemsize      # one value row, in bytes
        if self.vbytes % 8:
            # slots are [code u64 | state u64 | value row]; the header of slot i sits at
            # i * stride and is read as aligned uint64s, so the stride must be a multiple of 8
            raise ValueError(f"value row of {self.vbytes} bytes breaks slot alignment "
                             f"(d_v * itemsize must be a multiple of 8)")
        self.stride = 16 + self.vbytes
        self.cap = max(16, int(capacity))
        self.table = torch.from_numpy(np.zeros((self.cap, self.stride), dtype=np.uint8))
        self.table.fill_(0)                                # touch every page now
        _lock(self.table)
        self.counts = torch.zeros(groups, dtype=torch.int64)   # entries per partition

    def __len__(self):
        return int(self.counts.sum())

    def nbytes(self) -> int:
        return self.table.numel()

    def load(self) -> float:
        """Fraction of slots occupied. Linear probing's cost is set by this and nothing
        else: ~6 probes on a miss at 0.7, ~50 at 0.9, ~200 at 0.95."""
        return len(self) / self.cap

    def exchange(self, qcodes: torch.Tensor, kcodes: torch.Tensor, values: torch.Tensor,
                 out: torch.Tensor, T: int, g0: int):
        """One pass of P partitions x T positions against the store, in one C++ call:
        item i (partition i // T, position i % T, group g0 + i // T) looks `qcodes[i]` up
        into `out[i]` (zeros on a miss) and then inserts `kcodes[i] -> values[i]`, each
        partition in position order. That is the LEMA rule verbatim -- a query sees every
        key before it, including this chunk's, and not its own -- so a prefill chunk and a
        decode step (T = 1) are the same call.

        All arguments are host tensors: int64 codes (uint64 bit patterns) and rows of the
        carrier dtype. A full table raises from C++ rather than growing. There is
        deliberately no pre-check: `len + inserts > cap` is only an UPPER bound (an insert
        that hits an existing key takes no new slot), and the last few percent of capacity,
        where probe runs lengthen, is exactly the regime worth measuring, not forbidding.
        """
        P = qcodes.numel() // T
        self.counts[g0:g0 + P] += _ext().exchange(self.table, qcodes, kcodes, values, out,
                                                  T, g0, self.cap, self.vbytes)


# ------------------------------------------------------------------- the cache
class Staging:
    """Pinned host buffers for one exchange size: `codes` holds the queries in its first
    half and the keys in its second, `vals` the new values, `out` the retrieved ones.

    The store lives in CPU memory by design, so every layer must move codes and values
    across the bus. Pinned (page-locked) memory is what lets those copies be issued
    asynchronously -- and captured into a CUDA graph -- and completed with ONE
    synchronisation per layer instead of one per tensor. Built once per size and reused
    by every layer and every step.
    """

    __slots__ = ("n", "codes", "vals", "out")

    def __init__(self, n: int, d_v: int, dtype, pin: bool):
        self.n = n
        self.codes = torch.empty(2 * n, dtype=torch.int64, pin_memory=pin)
        self.vals = torch.empty(n, d_v, dtype=dtype, pin_memory=pin)   # STORE dtype: the
        self.out = torch.empty(n, d_v, dtype=dtype, pin_memory=pin)    # copy is a memcpy


def _carrier(t: torch.Tensor) -> torch.Tensor:
    """bfloat16 has no numpy name, so the C++ side gets it as raw uint16 bit patterns."""
    return t.view(torch.uint16) if t.dtype is torch.bfloat16 else t


class Buffers:
    """What every layer's view shares besides the store: the pinned staging buffers (built
    once per exchange size) and the optional codes hook. A separate object rather than the
    cache itself so that views do not point back at the cache -- a reference cycle would
    keep a freed cache's multi-gigabyte table alive until the cyclic garbage collector
    happened to run, which with two such tables is what swapped a machine."""

    def __init__(self, dtype, pin: bool, codes_hook=None):
        self.dtype, self.pin, self.codes_hook = dtype, pin, codes_hook
        self._staging = {}

    def staging(self, n: int, d_v: int) -> Staging:
        s = self._staging.get(n)
        if s is None:
            s = self._staging[n] = Staging(n, d_v, self.dtype, self.pin)
        return s


class LayerStore:
    """One layer's view of the cache: its slice of partitions in the shared store, plus the
    shared staging buffers.

    This is what a LEMA block is handed. It is deliberately the ONLY thing the attention
    op knows about caching -- the op asks it to exchange codes for values and has no idea
    how the table is allocated or shared.

    The exchange is three phases, exposed separately because single-token decoding
    (lema/decode.py) captures the two device phases into CUDA graphs and runs only the
    host phase in python between replays:

        s = stage(qc, kc, v)     device: codes and values -> pinned host buffers (async)
        host(s)                  host, after the copies landed: the one C++ call
        fetch(s, got)            device: retrieved values -> a device tensor (async)

    `exchange` is the eager composition of the three.
    """

    __slots__ = ("store", "buffers", "g0", "partitions")

    def __init__(self, store, buffers: Buffers, g0: int, partitions: int):
        self.store, self.buffers, self.g0, self.partitions = store, buffers, g0, partitions

    def stage(self, qc: torch.Tensor, kc: torch.Tensor, v: torch.Tensor) -> Staging:
        # [B, H, T] codes and [B, H, T, d_v] values flatten partition-major, position-minor,
        # which is the item order the store takes
        if self.buffers.codes_hook is not None:      # on the device, inside the captured step
            qc, kc = self.buffers.codes_hook(self.g0 // self.partitions, qc, kc)
        n, d_v = qc.numel(), v.shape[-1]
        s = self.buffers.staging(n, d_v)
        s.codes[:n].copy_(qc.reshape(-1), non_blocking=True)
        s.codes[n:].copy_(kc.reshape(-1), non_blocking=True)
        s.vals.copy_(v.reshape(-1, d_v), non_blocking=True)
        return s

    def host(self, s: Staging):
        n = s.n
        self.store.exchange(s.codes[:n], s.codes[n:], _carrier(s.vals), _carrier(s.out),
                            n // self.partitions, self.g0)

    @staticmethod
    def fetch(s: Staging, got: torch.Tensor):
        got.copy_(s.out.view(got.shape), non_blocking=True)

    def exchange(self, qc: torch.Tensor, kc: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Look up every query, fold in every key, in ONE host round trip: the values each
        query retrieves, `[B, H, T, d_v]` like `v`, zeros where nothing matched."""
        s = self.stage(qc, kc, v)
        if v.is_cuda:
            torch.cuda.current_stream().synchronize()   # one sync covers all three copies
        self.host(s)
        got = torch.empty_like(v)
        self.fetch(s, got)
        return got


class LemaCache:
    """The retrieval state of a model: one shared hash table, viewed per layer.

    ALWAYS in CPU memory -- never in VRAM. That is the claim being made, not an
    implementation detail: GPU memory stays independent of context length because this
    lives on the host side of the bus.
    """

    def __init__(self, cfg, dtype=torch.bfloat16, pin: bool = True,
                 max_cache_len: int = 8192, max_load: float = 0.7,
                 batch: int = 1, capacity: int | None = None, codes_hook=None):
        """Provision the cache, once. It never resizes; see `HashStore`.

        Size it either way round:
          `max_cache_len`  the longest sequence to serve -- slots = depth * batch * heads *
                           len / max_load, on the worst-case assumption that every token
                           contributes a distinct code to every head (a real stream revisits
                           codes, so this over-provisions);
          `capacity`       slots, exact. Use this to spend a MEMORY budget: bytes /
                           (16 + d_v * itemsize). Wins, when given, because a budget is the
                           thing usually known.

        Provisioning is what the baseline does too (`KVCache(max_len=)`), so the
        comparison stays symmetric: both sides reserve for a maximum, ours in host memory
        instead of VRAM.

        `codes_hook(layer, qc, kc) -> (qc, kc)`, if given, is applied on the device to
        the `[B, H, T]` int64 query and key codes of every layer before they are staged,
        so under CUDA graphs it is captured into the step and costs no host time.
        Benchmarks use it to impose an access pattern on the store; generation does not
        need it.
        """
        # Named explicitly rather than taking **kwargs: a mistyped or stale argument would
        # otherwise be swallowed and silently ignored, which for a SIZING parameter means
        # quietly getting the un-provisioned behaviour back.
        if dtype not in _CARRIER:
            raise ValueError(f"cache dtype must be one of {list(_CARRIER)}, got {dtype}")
        if not 0 < max_load <= 0.95:
            # open addressing cannot hold more entries than slots, and linear probing
            # degrades quadratically on the way there; above 0.95 it is unusable
            raise ValueError(f"max_load must be in (0, 0.95], got {max_load}")
        P = batch * cfg.num_heads                     # partitions per layer
        slots = int(capacity) if capacity else math.ceil(cfg.depth * P * max_cache_len
                                                         / max_load)
        self.cfg, self.batch = cfg, batch
        self.store = HashStore(cfg.d_v, groups=cfg.depth * P, dtype=_CARRIER[dtype],
                               capacity=slots)
        self.t = 0
        self.buffers = Buffers(dtype, pin and torch.cuda.is_available(), codes_hook)
        self._views = [LayerStore(self.store, self.buffers, i * P, P)
                       for i in range(cfg.depth)]

    def layer(self, i: int) -> LayerStore:
        return self._views[i]

    def staging(self, n: int, d_v: int) -> Staging:
        """The pinned buffers for an exchange of `n` rows, shared by all layers."""
        return self.buffers.staging(n, d_v)

    def counts(self) -> torch.Tensor:
        """Entries per partition as `[depth, batch, heads]`: how many distinct codes each
        head has emitted so far."""
        return self.store.counts.view(self.cfg.depth, self.batch, self.cfg.num_heads)

    def entries(self) -> int:
        return len(self.store)

    def nbytes(self) -> int:
        """Bytes the table reserves. Excludes the pinned staging buffers, which are a
        fixed-size implementation detail."""
        return self.store.nbytes()

    def load(self) -> float:
        return self.store.load()


class KVCache:
    """Preallocated per-layer key/value buffers: what the softmax baseline decodes with,
    always in VRAM (its operator reads every cached entry each step).

    Laid out `[batch, max_len, heads, head_dim]`, the layout `flash_attn_with_kvcache`
    takes, with the fill level `pos` kept ON THE DEVICE: the kernel appends the new
    key/value at `pos` and reads only `[:pos]`, so a decode step has no shape that depends
    on the context length and can be captured into a CUDA graph once and replayed for
    every token -- while still costing only what the live context costs to read. (The
    usual alternative, a mask over the whole buffer, is capturable but reads all of it
    every step.) Rotary tables for every position are precomputed here so the kernel can
    apply them fused, at positions `pos + i`.

    Preallocation (rather than growing by concatenation) is what makes this a fair
    baseline: re-allocating and copying the cache every step would add an O(context)
    memcpy that no real implementation pays. The memory footprint is exactly weights +
    this cache, the same accounting a static cache produces.
    """

    def __init__(self, model, max_len: int, batch: int = 1, device=None, dtype=None,
                 num_splits: int = 0):
        cfg = model.cfg
        p = next(model.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        self.k = [torch.empty(batch, max_len, cfg.num_kv_heads, cfg.d_qk, device=device,
                              dtype=dtype) for _ in range(cfg.depth)]
        self.v = [torch.empty(batch, max_len, cfg.num_kv_heads, cfg.d_v, device=device,
                              dtype=dtype) for _ in range(cfg.depth)]
        self.pos = torch.zeros(batch, dtype=torch.int32, device=device)
        self.n = 0                       # host mirror of pos; see Transformer._run_kv
        self.max_len, self.batch = max_len, batch
        # The decode kernel splits the key/value range it is GIVEN into pieces, one thread
        # block per piece and key/value head, and blocks whose piece lies beyond the live
        # context exit at once, so a single-token step hands it the cache sliced to the
        # live length (`Transformer._run_kv`, `decode.KVDecoder`). `num_splits` 0 takes the
        # count that fills the SMs with one wave (`model.decode_splits`); a positive value
        # fixes it.
        self.num_splits = num_splits
        self.cos = self.sin = None
        if cfg.pos == "rope":            # rotate-half, the model's base, as in model.rope
            half = cfg.d_qk // 2
            freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                              * (-math.log(float(cfg.rope_base)) / half))
            ang = torch.outer(torch.arange(max_len, device=device, dtype=torch.float32),
                              freqs)
            self.cos, self.sin = ang.cos().to(dtype), ang.sin().to(dtype)

