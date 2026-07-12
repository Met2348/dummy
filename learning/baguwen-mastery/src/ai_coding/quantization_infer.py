"""INT8 对称量化 / 反量化（纯 Python list，不依赖 numpy/torch）。

对称量化：scale = absmax / 127，量化值 q = round(x / scale)，裁剪到 [-127, 127]。
反量化：x_hat = q * scale。
理论误差上界：量化-反量化的往返误差 |x - x_hat| <= scale / 2
（四舍五入引入的误差不超过半个量化步长），加一个很小的浮点容差。
"""
from __future__ import annotations

INT8_MAX = 127


def quantize_int8(tensor: list[float]) -> tuple[list[int], float]:
    """对称 INT8 量化。返回 (量化后的整数列表, scale)。"""
    if not tensor:
        raise ValueError("tensor 不能为空")

    absmax = max(abs(x) for x in tensor)
    if absmax == 0:
        # 全零向量：scale 取任意正数即可（约定为 1.0），量化结果全 0
        return [0] * len(tensor), 1.0

    scale = absmax / INT8_MAX
    quantized = []
    for x in tensor:
        q = round(x / scale)
        q = max(-INT8_MAX, min(INT8_MAX, q))  # 裁剪，防止浮点误差导致越界
        quantized.append(q)
    return quantized, scale


def dequantize_int8(quantized: list[int], scale: float) -> list[float]:
    """反量化：q * scale。"""
    return [q * scale for q in quantized]


def roundtrip_max_error(tensor: list[float]) -> float:
    """量化再反量化一轮后，往返误差的最大值（用于自查/单测）。"""
    quantized, scale = quantize_int8(tensor)
    dequantized = dequantize_int8(quantized, scale)
    return max(abs(orig - deq) for orig, deq in zip(tensor, dequantized))


def _self_test() -> None:
    tensor = [-2.5, -1.0, 0.0, 0.3, 1.7, 2.5]
    quantized, scale = quantize_int8(tensor)

    # scale 应等于 absmax(2.5) / 127
    assert abs(scale - 2.5 / 127) < 1e-12

    # 量化后的整数都应落在 int8 对称范围内
    assert all(-127 <= q <= 127 for q in quantized)
    assert isinstance(quantized[0], int)

    # 最大绝对值元素应被量化到边界 -127 / 127
    assert quantized[0] == -127  # -2.5 是 absmax，取负
    assert quantized[-1] == 127  # 2.5 是 absmax，取正

    # 往返误差应落在理论上界 scale/2 (+ 浮点容差) 之内
    dequantized = dequantize_int8(quantized, scale)
    tol = scale / 2 + 1e-9
    for orig, deq in zip(tensor, dequantized):
        assert abs(orig - deq) <= tol, (orig, deq, tol)

    # 0.0 应该精确量化回 0.0（round(0/scale) == 0）
    zero_idx = tensor.index(0.0)
    assert dequantized[zero_idx] == 0.0

    # 全零向量的边界情况：不应除零，量化结果应全为 0
    zeros = [0.0, 0.0, 0.0]
    q0, s0 = quantize_int8(zeros)
    assert all(v == 0 for v in q0)
    assert dequantize_int8(q0, s0) == [0.0, 0.0, 0.0]

    # 空 tensor 应报错
    try:
        quantize_int8([])
        assert False, "应对空 tensor 报错"
    except ValueError:
        pass

    # 误差上界应随 absmax 增大而增大（scale 变大 -> 每个量化步长更粗）
    small_range = roundtrip_max_error([-1.0, 0.5, 1.0])
    large_range = roundtrip_max_error([-100.0, 50.0, 100.0])
    assert small_range < large_range

    print(f"[PASS] quantization_infer: INT8对称量化/反量化往返误差 <= scale/2（scale={scale:.6f}）")


if __name__ == "__main__":
    _self_test()
