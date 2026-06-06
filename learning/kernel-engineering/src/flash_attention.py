"""FlashAttention-2 algorithm — tile + online softmax + fuse."""
from __future__ import annotations
import math


def attention_naive(Q: list[list[float]], K: list[list[float]],
                    V: list[list[float]]) -> list[list[float]]:
    """Materializes full N×N attention matrix in HBM."""
    N, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d)
    # S = Q K^T
    S = [[sum(Q[i][k] * K[j][k] for k in range(d)) * scale
          for j in range(N)] for i in range(N)]
    # P = softmax(S, axis=-1)
    P = []
    for i in range(N):
        m = max(S[i])
        e = [math.exp(s - m) for s in S[i]]
        z = sum(e)
        P.append([x / z for x in e])
    # O = P V
    O = [[sum(P[i][j] * V[j][k] for j in range(N))
          for k in range(d)] for i in range(N)]
    return O


def attention_flash(Q: list[list[float]], K: list[list[float]],
                    V: list[list[float]], block_n: int = 4) -> list[list[float]]:
    """FlashAttention-2: outer loop over Q rows, inner loop over K/V blocks.

    Online running stats per Q row:
      m: running max
      l: running normalizer
      O: running output (un-normalized in numerator)
    """
    N, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d)
    O = [[0.0] * d for _ in range(N)]

    for i in range(N):
        m = -math.inf
        l = 0.0
        o = [0.0] * d
        for j_start in range(0, N, block_n):
            j_end = min(j_start + block_n, N)
            # Local block
            s_block = [sum(Q[i][k] * K[j][k] for k in range(d)) * scale
                       for j in range(j_start, j_end)]
            m_new = max(m, max(s_block))
            rescale = math.exp(m - m_new) if m != -math.inf else 0.0
            l = l * rescale
            o = [v * rescale for v in o]
            for jj, s in enumerate(s_block):
                p = math.exp(s - m_new)
                l += p
                j = j_start + jj
                for k in range(d):
                    o[k] += p * V[j][k]
            m = m_new
        # Final normalize
        O[i] = [v / l for v in o]
    return O


def _self_test() -> None:
    import random
    random.seed(7)
    N, d = 8, 4
    Q = [[random.random() for _ in range(d)] for _ in range(N)]
    K = [[random.random() for _ in range(d)] for _ in range(N)]
    V = [[random.random() for _ in range(d)] for _ in range(N)]
    O1 = attention_naive(Q, K, V)
    O2 = attention_flash(Q, K, V, block_n=3)  # block_n not dividing N
    for i in range(N):
        for k in range(d):
            assert abs(O1[i][k] - O2[i][k]) < 1e-9, (i, k, O1[i][k], O2[i][k])
    print("[OK] flash_attention (online == naive)")


if __name__ == "__main__":
    _self_test()
