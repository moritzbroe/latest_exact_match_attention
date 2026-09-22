"""Reference LEMA transformer matching the appendix.

precision selects one of two paths.

  'int<k>'      exact-integer shortcut used to check the binary construction of
                Theorem 1. This is a verification mode, not a model format.
  'fp16/32/64'  arithmetic in the corresponding IEEE format, rounding to
              nearest with ties to even. Matrix products are accumulated in
              increasing index order with rounding after every step, as
              the appendix requires and as BLAS does not do. A nonfinite
              intermediate raises: the theorems concern finite executions.
"""

import numpy as np

_FLOAT = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}


def lema(q, k, v):
    """Latest exact match attention, strictly causal.

    q, k: (N, d_h) in {-1,1};  v: (N, d_h).  Row i is v_j for the largest j < i
    with k_j == q_i, and 0 if there is no such j. This is the dictionary of the
    paper: each position looks its query up, then inserts its own key.
    """
    src, last = [-1] * len(q), {}
    for i in range(len(q)):
        src[i] = last.get(q[i].tobytes(), -1)
        last[k[i].tobytes()] = i
    src = np.array(src)
    out = np.zeros((len(q), v.shape[1]), dtype=v.dtype)
    out[src >= 0] = v[src[src >= 0]]
    return out


def _format(precision):
    """(dtype, limit). limit is None for float formats, else |v| < limit."""
    if precision in _FLOAT:
        return _FLOAT[precision], None
    if isinstance(precision, str) and precision.startswith("int") and precision[3:].isdigit():
        bits = int(precision[3:])
        if not 2 <= bits <= 63:
            raise ValueError("integer precision must be between int2 and int63")
        return np.int64, 1 << (bits - 1)
    raise ValueError(f"precision must be 'int<k>' or one of {sorted(_FLOAT)}, got {precision!r}")


class LemaTransformer:
    def __init__(self, emb, unemb, layers, precision="int32", strict=True, binary=False):
        """emb, unemb: (|V|, d).  layers: one dict per layer with

            "heads": [(W_Q, W_K, W_V, W_O), ...]   shapes (d_h,d) (d_h,d) (d_h,d) (d,d_h)
            "W1", "b", "W2"                      shapes (d_ff,d) (d_ff,) (d,d_ff)

        Supplied head and MLP widths may differ; they are padded to the largest
        supplied size so the resulting model has the usual uniform dimensions,
        with one head dimension d_h shared by queries, keys and values as in
        the appendix's transformer definition.  Padding a head's key/query rows duplicates its first row,
        which leaves every exact match and hence every output unchanged.
        strict raises if sgn is ever applied to 0 or if an output argmax is tied,
        i.e. if the construction relies on either convention.
        binary additionally requires every residual state to lie in {-1,1}^d
        and every ReLU pattern neuron to output 0 or 1.
        """
        self.precision, self.strict, self.binary = precision, strict, binary
        self.dtype, self.limit = _format(precision)
        self.integer = self.limit is not None
        emb, unemb, layers = self._pad_and_check(emb, unemb, layers)
        self.emb = self._cast(emb, "emb")
        self.unemb = self._cast(unemb, "unemb")
        self.layers = [{
            "heads": [tuple(self._cast(M, f"layer {n} head {i}") for M in h)
                      for i, h in enumerate(layer["heads"])],
            **{p: self._cast(layer[p], f"layer {n} {p}") for p in ("W1", "b", "W2")},
        } for n, layer in enumerate(layers)]

        self._max_abs = 0.0
        for A in [self.emb, self.unemb] + [M for layer in self.layers
                                           for M in [x for h in layer["heads"] for x in h]
                                           + [layer[p] for p in ("W1", "b", "W2")]]:
            self._track(A)
        self._param_abs = self._max_abs

    @staticmethod
    def _pad_and_check(emb, unemb, layers):
        """Validate shapes and pad all layers to uniform H, d_h and d_ff."""
        emb, unemb = np.asarray(emb), np.asarray(unemb)
        if emb.ndim != 2 or unemb.shape != emb.shape:
            raise ValueError("emb and unemb must have the same two-dimensional shape")
        if not layers:
            raise ValueError("a transformer needs at least one layer")
        d = emb.shape[1]
        checked, H, d_h, d_ff = [], 0, 0, 0
        for n, layer in enumerate(layers):
            heads = []
            for i, head in enumerate(layer["heads"]):
                if len(head) != 4:
                    raise ValueError(f"layer {n} head {i} must contain Q, K, V, O")
                Q, K, V, O = map(np.asarray, head)
                if Q.ndim != 2 or K.shape != Q.shape or Q.shape[1] != d:
                    raise ValueError(f"bad query/key shape in layer {n} head {i}")
                if V.ndim != 2 or V.shape[1] != d or O.shape != (d, V.shape[0]):
                    raise ValueError(f"bad value/output shape in layer {n} head {i}")
                if Q.shape[0] == 0 or V.shape[0] == 0:
                    raise ValueError("head dimensions must be positive")
                heads.append((Q, K, V, O))
                d_h = max(d_h, Q.shape[0], V.shape[0])
            W1, b, W2 = (np.asarray(layer[p]) for p in ("W1", "b", "W2"))
            if W1.ndim != 2 or W1.shape[1] != d or b.shape != (W1.shape[0],):
                raise ValueError(f"bad first MLP shape in layer {n}")
            if W2.shape != (d, W1.shape[0]):
                raise ValueError(f"bad second MLP shape in layer {n}")
            checked.append({"heads": heads, "W1": W1, "b": b, "W2": W2})
            H, d_ff = max(H, len(heads)), max(d_ff, W1.shape[0])
        if H == 0:
            raise ValueError("at least one layer must contain an attention head")
        d_ff = max(d_ff, 1)

        for layer in checked:
            padded = []
            for Q, K, V, O in layer["heads"]:
                if Q.shape[0] < d_h:       # duplicate a row: exact matches are unchanged
                    qn = d_h - Q.shape[0]
                    Q = np.concatenate((Q, np.repeat(Q[:1], qn, axis=0)))
                    K = np.concatenate((K, np.repeat(K[:1], qn, axis=0)))
                if V.shape[0] < d_h:
                    vn = d_h - V.shape[0]
                    V = np.pad(V, ((0, vn), (0, 0)))
                    O = np.pad(O, ((0, 0), (0, vn)))
                padded.append((Q, K, V, O))
            while len(padded) < H:
                Q = np.zeros((d_h, d), dtype=emb.dtype)
                K = np.zeros((d_h, d), dtype=emb.dtype)
                Q[:, 0] = K[:, 0] = 1       # coordinate 0 is constant in our constructions
                padded.append((Q, K, np.zeros((d_h, d), dtype=emb.dtype),
                               np.zeros((d, d_h), dtype=emb.dtype)))
            layer["heads"] = padded
            n = d_ff - layer["W1"].shape[0]
            if n:
                layer["W1"] = np.pad(layer["W1"], ((0, n), (0, 0)))
                layer["b"] = np.pad(layer["b"], (0, n))
                layer["W2"] = np.pad(layer["W2"], ((0, 0), (0, n)))
        return emb, unemb, checked

    def _cast(self, A, name):
        """Parameters must be exactly representable in the model's format."""
        A = np.asarray(A)
        B = A.astype(self.dtype)
        if not np.array_equal(B.astype(A.dtype), A):
            raise ValueError(f"{name} is not exactly representable in {self.precision}")
        if not self.integer and not np.isfinite(B).all():
            raise ValueError(f"{name} contains a nonfinite value")
        return B

    def _finite(self, value):
        """Reject overflow and NaNs; finite IEEE underflow is left unchanged."""
        if not np.isfinite(value).all():
            raise OverflowError(f"nonfinite intermediate in {self.precision}")
        return value

    def _track(self, value):
        if not self.integer:
            value = self._finite(value)
        m = float(np.abs(value).max())
        if not np.isfinite(m):
            raise OverflowError(f"NaN in {self.precision}")
        self._max_abs = max(self._max_abs, m)
        if self.limit is not None and m >= self.limit:
            raise OverflowError(f"magnitude {m:.0f} does not fit in {self.precision}")
        return value

    def _mm(self, A, B):
        """A @ B. On the integer path the tracked bound sum_j |A_ij||B_jk| covers
        every partial sum, and is somewhat stricter than necessary: a product
        whose partial sums cancel is charged the uncancelled total. On a float
        path the sum is accumulated in increasing index order, rounding after
        every multiplication and addition."""
        if self.integer:
            a = int(np.abs(A).max()) if A.size else 0
            b = int(np.abs(B).max()) if B.size else 0
            if A.shape[1] * a * b > np.iinfo(np.int64).max:
                raise OverflowError("integer matrix-product bound overflows int64")
            self._track(np.abs(A) @ np.abs(B))
            return A @ B
        acc = np.zeros((A.shape[0], B.shape[1]), dtype=self.dtype)
        for j in range(A.shape[1]):
            product = self._finite(A[:, j, None] * B[None, j, :])
            acc = self._finite(acc + product)
        return self._track(acc)

    def _sgn(self, z):
        if self.strict and (z == 0).any():
            raise ValueError("sgn applied to 0")
        return np.where(z >= 0, 1, -1).astype(self.dtype)

    def dims(self):
        """Dimensions of the (already uniformly padded) model."""
        head = self.layers[0]["heads"][0]
        return {"vocab": self.emb.shape[0], "L": len(self.layers), "d": self.emb.shape[1],
                "H": len(self.layers[0]["heads"]), "d_h": head[0].shape[0],
                "d_ff": self.layers[0]["W1"].shape[0]}

    def _state(self, x, where):
        self._track(x)
        if self.binary and not np.all((x == -1) | (x == 1)):
            bad = np.argwhere((x != -1) & (x != 1))[0]
            raise ValueError(f"non-binary residual state {x[tuple(bad)]} at {where}, index {tuple(bad)}")
        return x

    def forward(self, tokens, return_trace=False, *, final_only=False):
        """Predicted next token at every position, plus an info dict holding

            s_T      distinct keys for these input tokens, summed over all heads
            max_abs  largest magnitude reached, weights and activations
            layers   per-layer activations, only if return_trace=True

        If final_only=True, compute logits only at the final position and return
        only that position's prediction.  The layer states are still computed
        at every position.  This is the autoregressive path: logits at earlier
        prompt positions are unused.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            return self._forward(tokens, return_trace, final_only)

    def _forward(self, tokens, return_trace=False, final_only=False):
        self._max_abs = self._param_abs
        x = self._state(self.emb[np.asarray(tokens)], "embedding")
        acts, s_T = [], 0
        for layer in self.layers:
            a = {"heads": []}
            s = np.zeros_like(x)
            for W_Q, W_K, W_V, W_O in layer["heads"]:
                q = self._sgn(self._mm(x, W_Q.T))
                k = self._sgn(self._mm(x, W_K.T))
                v = self._mm(x, W_V.T)
                o = lema(q, k, v)
                s = self._track(s + self._mm(o, W_O.T))
                s_T += len({row.tobytes() for row in k})
                a["heads"].append({"q": q, "k": k, "v": v, "o": o})
            x = self._state(x + s, f"layer {len(acts)} attention")
            a["x_mid"] = x
            z = np.maximum(self._track(self._mm(x, layer["W1"].T) + layer["b"]), 0)
            if self.binary and not np.all((z == 0) | (z == 1)):
                raise ValueError(f"non-binary MLP activation in layer {len(acts)}")
            x = self._state(x + self._mm(z, layer["W2"].T), f"layer {len(acts)} MLP")
            a["z"], a["x"] = z, x
            acts.append(a)

        logits = self._mm(x[-1:] if final_only else x, self.unemb.T)
        if self.strict:
            top = logits == logits.max(axis=1, keepdims=True)
            if (top.sum(axis=1) > 1).any():
                raise ValueError(f"tie in argmax at positions {np.flatnonzero(top.sum(1) > 1)}")
        info = {"s_T": s_T,
                "max_abs": int(self._max_abs) if self.integer else self._max_abs}
        if return_trace:
            info["layers"] = acts
        return logits.argmax(axis=1), info

    def generate(self, prompt, limit, stop=None):
        """Autoregressive generation, for small examples only. Returns the full
        token sequence. Checking a whole transcript is much cheaper with forward,
        which gives the prediction at every position in one pass."""
        toks = list(prompt)
        for _ in range(limit):
            toks.append(int(self.forward(toks, final_only=True)[0][0]))
            if stop is not None and toks[-1] == stop:
                break
        return toks
