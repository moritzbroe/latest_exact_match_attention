// The LEMA retrieval store: ONE open-addressing hash table for every (layer, stream, head).
//
// A slot is [code u64 | state u64 | value bytes] (stride = 16 + vbytes; vbytes is a multiple
// of 8, so the header stays aligned). The key of an entry is the pair (group, code): a code is
// the 64-bit encoding of a head's query/key pattern, the group names the (layer, stream, head)
// partition it belongs to. All partitions share the slots, so a busy head simply takes more
// of them and the table fills at the TOTAL number of distinct codes rather than at the busiest
// head's -- with one table per head, memory would have to be provisioned for the busiest
// head times the number of heads. The group is mixed into the hash, so the same code in two
// heads has two unrelated home slots (a random-weight model emits one code in every head; on
// a shared run that would be one long cluster).
//
// The state word holds the group of a published entry, offset by 2, and doubles as the slot
// state: EMPTY (0) is a free slot, LOCKED (1) a slot some thread is filling. A zeroed
// allocation is therefore already an empty table, and every one of the 2^64 codes is an
// ordinary key -- the sentinels live in the state word, not in the code word.
//
// `exchange` is the whole interface. Its items are partition-major, position-minor (item i is
// position i % T of partition i / T), and each partition's positions run IN ORDER: look the
// query up, then insert the key. That is the LEMA rule verbatim -- a query sees every key
// before it, including this chunk's, and not its own -- so a prefill chunk of T tokens and a
// decode step (T = 1) are the same call. Hits and inserts into a run mostly touch lines the
// prefetch of W items ahead already fetched: an item's home slot depends only on its code,
// not on what earlier items did, so the chain is ordered but not latency-bound.
//
// Threads split PARTITIONS (independent streams of the same table). Cross-thread interaction
// is only through shared slots, and the state word keeps it safe without locks: a claim is a
// compare-and-swap EMPTY -> LOCKED, the code and value are written, then the group is
// published with a release store. A reader (acquire load) sees EMPTY (end of run), LOCKED
// (someone else's key being written: skip it), or a published group whose code and value are
// complete. A key can only be read and written by the thread owning its partition, so
// latest-wins within a partition needs no further care. Runs only ever grow, so a key present
// when a probe starts stays reachable.
//
// The GIL is released for the whole call.

#include <torch/extension.h>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint64_t EMPTY = 0, LOCKED = 1;
constexpr int W = 16;                       // prefetch depth: items in flight per thread

// Items per thread below which one thread wins. Measured on a 14 GB table at 50% load (16
// threads): 24 items (batch 1, 24 heads) cost 3.5 us on one thread and 6.4 us on two; 384
// items 34 us on one thread and 17 threaded; 1536 items 138 vs 48 us, where the machine's
// random-access throughput (~125M lines/s) becomes the limit. Threads pay off from a few
// hundred items: batched decoding, or any prefill chunk.
#ifndef GRAIN
#define GRAIN 128
#endif

// splitmix64 finalizer. Codes are highly structured (they are what the model chose, not
// random bits), so they are mixed before being used as slot indices.
inline uint64_t mix(uint64_t x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27; x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  return x;
}

// Home slot of (group, code), in [0, cap) for ANY cap (Lemire's fastrange: the high half of
// the 128-bit product is uniform). No power-of-two rounding, so a table can be sized to a
// memory budget exactly.
inline int64_t home(uint64_t code, uint64_t group, int64_t cap) {
  const uint64_t h = mix(code ^ (group * 0x9E3779B97F4A7C15ULL));
  return int64_t((static_cast<__uint128_t>(h) * uint64_t(cap)) >> 64);
}

struct Table {
  uint8_t* base;
  int64_t cap, stride, vbytes;

  uint64_t* code(int64_t s) const { return reinterpret_cast<uint64_t*>(base + s * stride); }
  uint64_t* state(int64_t s) const { return code(s) + 1; }   // EMPTY, LOCKED, or group + 2
  uint8_t* value(int64_t s) const { return base + s * stride + 16; }
  int64_t next(int64_t s) const { return ++s == cap ? 0 : s; }
  void prefetch(uint64_t c, uint64_t g) const { __builtin_prefetch(base + home(c, g, cap) * stride); }

  // The value stored under (group, code) into dst; zeros if absent.
  void read(uint64_t c, uint64_t g, uint8_t* dst) const {
    const uint64_t tag = g + 2;
    int64_t s = home(c, g, cap);
    for (int64_t n = 0; n < cap; ++n, s = next(s)) {
      const uint64_t k = __atomic_load_n(state(s), __ATOMIC_ACQUIRE);
      if (k == EMPTY) break;
      if (k == tag && *code(s) == c) { std::memcpy(dst, value(s), vbytes); return; }
    }
    std::memset(dst, 0, vbytes);
  }

  // (group, code) -> value, latest write wins. Returns the number of newly claimed slots
  // (0 or 1), or -1 if the partition's probe went all the way round: the table is full.
  int insert(uint64_t c, uint64_t g, const uint8_t* src) {
    const uint64_t tag = g + 2;
    int64_t s = home(c, g, cap);
    for (int64_t n = 0; n < cap; ++n, s = next(s)) {
      uint64_t k = __atomic_load_n(state(s), __ATOMIC_ACQUIRE);
      if (k == EMPTY) {                        // claim it
        uint64_t expect = EMPTY;
        if (__atomic_compare_exchange_n(state(s), &expect, LOCKED, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
          *code(s) = c;
          std::memcpy(value(s), src, vbytes);
          __atomic_store_n(state(s), tag, __ATOMIC_RELEASE);
          return 1;
        }
        k = expect;                            // lost the race: judge what the slot now holds
      }
      if (k == tag && *code(s) == c) { std::memcpy(value(s), src, vbytes); return 0; }
      // LOCKED, or another key: keep probing
    }
    return -1;
  }
};

}  // namespace

// One exchange against the store: items i in [0, n) are position i % T of partition i / T,
// whose group id is g0 + i / T. Reads qcodes[i] into out[i] (zeros on a miss), then inserts
// kcodes[i] -> values[i], partition by partition in position order. `vbytes` is the size of
// one value row in bytes; the store never interprets a value. Returns the number of newly
// occupied slots per partition.
torch::Tensor store_exchange(torch::Tensor table, torch::Tensor qcodes, torch::Tensor kcodes,
                             torch::Tensor values, torch::Tensor out, int64_t T, int64_t g0,
                             int64_t cap, int64_t vbytes) {
  for (auto& t : {table, qcodes, kcodes, values, out})
    TORCH_CHECK(t.is_contiguous(), "store buffers must be contiguous");
  const int64_t n = qcodes.numel();
  TORCH_CHECK(kcodes.numel() == n && T > 0 && n % T == 0, "items must be partitions x T");
  const int64_t P = n / T;
  Table tb{static_cast<uint8_t*>(table.data_ptr()), cap, 16 + vbytes, vbytes};
  const uint64_t* qc = reinterpret_cast<const uint64_t*>(qcodes.data_ptr<int64_t>());
  const uint64_t* kc = reinterpret_cast<const uint64_t*>(kcodes.data_ptr<int64_t>());
  const uint8_t* sp = static_cast<const uint8_t*>(values.data_ptr());
  uint8_t* op = static_cast<uint8_t*>(out.data_ptr());
  auto added = torch::zeros({P}, torch::kInt64);
  int64_t* ap = added.data_ptr<int64_t>();
  std::atomic<bool> full{false};

  py::gil_scoped_release release;
  at::parallel_for(0, P, std::max<int64_t>(1, GRAIN / T), [&](int64_t p_lo, int64_t p_hi) {
    const int64_t lo = p_lo * T, hi = p_hi * T;
    auto ahead = [&](int64_t i) {
      if (i < hi) { tb.prefetch(qc[i], g0 + i / T); tb.prefetch(kc[i], g0 + i / T); }
    };
    for (int64_t i = lo; i < lo + W; ++i) ahead(i);
    for (int64_t i = lo; i < hi; ++i) {
      ahead(i + W);
      const uint64_t g = g0 + i / T;
      tb.read(qc[i], g, op + i * vbytes);
      const int r = tb.insert(kc[i], g, sp + i * vbytes);
      if (r < 0) full = true; else ap[i / T] += r;
    }
  });
  TORCH_CHECK(!full, "hash table full: the store is presized and never grows -- "
                     "provision more capacity");
  return added;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exchange", &store_exchange,
        "look up every query, then insert every key, partition by partition in order");
}
