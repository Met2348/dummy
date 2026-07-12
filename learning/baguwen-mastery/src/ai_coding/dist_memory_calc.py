"""分布式训练显存占用估算（纯 stdlib，不依赖 numpy/torch）。

按混合精度训练的经典拆解估算显存：参数 + 梯度 + 优化器状态，
激活值只给一个粗略的、随 batch/seq_len/层数线性增长的简化比例说明，
不追求精确建模（不同实现的重计算/融合算子细节差异很大，精确建模意义不大）。

参考量级（Adam + fp16/bf16 混合精度，来自 ZeRO 论文的经典"16Ψ"估算）：
    参数(fp16)      2 字节/参数
    梯度(fp16)      2 字节/参数
    优化器状态(fp32) 一阶矩 4 + 二阶矩 4 + fp32主权重 4 = 12 字节/参数
    合计            16 字节/参数，即约"参数量的16倍字节数"
"""
from __future__ import annotations

BYTES_PER_ELEMENT = {"fp16": 2, "bf16": 2, "fp32": 4}


def estimate_training_memory(num_params: int, precision: str = "fp16", optimizer: str = "adam") -> dict:
    """估算训练显存占用（字节），返回参数/梯度/优化器状态三项拆解 + 总量。

    - precision: "fp16" / "bf16" / "fp32" —— 决定参数和梯度各自的字节数。
    - optimizer: "adam"（一阶矩+二阶矩+fp32主权重，共12字节/参数）
                 "sgd"（无额外状态）
                 "sgd_momentum"（一份fp32动量，4字节/参数）
    """
    if num_params <= 0:
        raise ValueError(f"num_params 必须为正数，收到 {num_params}")
    if precision not in BYTES_PER_ELEMENT:
        raise ValueError(f"不支持的精度: {precision}，可选 {list(BYTES_PER_ELEMENT)}")

    param_bytes = num_params * BYTES_PER_ELEMENT[precision]
    grad_bytes = num_params * BYTES_PER_ELEMENT[precision]

    if optimizer == "adam":
        # fp32 一阶矩(momentum) + fp32 二阶矩(variance) + fp32 主权重(master weight)
        optimizer_bytes = num_params * 4 * 3
    elif optimizer == "sgd":
        optimizer_bytes = 0
    elif optimizer == "sgd_momentum":
        optimizer_bytes = num_params * 4 * 1
    else:
        raise ValueError(f"不支持的优化器: {optimizer}，可选 adam/sgd/sgd_momentum")

    total_bytes = param_bytes + grad_bytes + optimizer_bytes
    return {
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_bytes": optimizer_bytes,
        "total_bytes": total_bytes,
        "bytes_per_param": total_bytes / num_params,
    }


def estimate_activation_bytes_rough(
    num_layers: int, hidden_size: int, seq_len: int, batch_size: int,
    bytes_per_element: int = 2, checkpointing: bool = False,
) -> int:
    """粗略估算激活值显存，不追求精确建模，只用于定性说明：
    激活值随 层数 x batch_size x seq_len x hidden_size 线性增长，
    gradient checkpointing 只保留检查点位置的激活，大致把每层的"存储因子"从
    一个较大的常数（这里粗略取 20，对应一层里 Q/K/V/attention输出/MLP中间层等
    若干份 hidden_size 大小的中间张量）降到一个小得多的常数（这里粗略取 2）。
    """
    if num_layers <= 0 or hidden_size <= 0 or seq_len <= 0 or batch_size <= 0:
        raise ValueError("num_layers/hidden_size/seq_len/batch_size 必须为正数")
    factor = 2 if checkpointing else 20
    return num_layers * batch_size * seq_len * hidden_size * bytes_per_element * factor


def human_readable(num_bytes: float) -> str:
    """把字节数格式化成带单位的可读字符串，便于打印。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f}PB"


def _self_test() -> None:
    # 用一个 1 亿参数量级（1e8）的小模型，fp16 + Adam 混合精度训练估算
    num_params = 100_000_000
    breakdown = estimate_training_memory(num_params, precision="fp16", optimizer="adam")

    assert breakdown["param_bytes"] == 200_000_000
    assert breakdown["grad_bytes"] == 200_000_000
    assert breakdown["optimizer_bytes"] == 1_200_000_000
    assert breakdown["total_bytes"] == 1_600_000_000

    # 定性结论：优化器状态显存明显大于参数本身显存（Adam 下约 6 倍）
    assert breakdown["optimizer_bytes"] > breakdown["param_bytes"] * 3, "优化器状态应明显大于参数本身"

    # 混合精度 Adam 下经典的"16 字节/参数"（16Ψ）估算
    assert abs(breakdown["bytes_per_param"] - 16.0) < 1e-9

    # 总显存应落在一个合理区间内（1GB ~ 2GB）
    assert 1e9 <= breakdown["total_bytes"] <= 2e9

    # sgd 无优化器状态，总显存应明显低于 adam
    sgd_breakdown = estimate_training_memory(num_params, precision="fp16", optimizer="sgd")
    assert sgd_breakdown["optimizer_bytes"] == 0
    assert sgd_breakdown["total_bytes"] < breakdown["total_bytes"]

    # fp32 训练下参数/梯度显存翻倍
    fp32_breakdown = estimate_training_memory(num_params, precision="fp32", optimizer="adam")
    assert fp32_breakdown["param_bytes"] == 400_000_000
    assert fp32_breakdown["param_bytes"] == 2 * breakdown["param_bytes"]

    # 非法输入应报错
    try:
        estimate_training_memory(-1)
        assert False, "应对非正数 num_params 报错"
    except ValueError:
        pass
    try:
        estimate_training_memory(num_params, precision="int8")
        assert False, "应对不支持的精度报错"
    except ValueError:
        pass

    # 激活值：确定性输入下 gradient checkpointing 应显著降低激活显存
    act_full = estimate_activation_bytes_rough(
        num_layers=32, hidden_size=4096, seq_len=2048, batch_size=1, checkpointing=False,
    )
    act_ckpt = estimate_activation_bytes_rough(
        num_layers=32, hidden_size=4096, seq_len=2048, batch_size=1, checkpointing=True,
    )
    assert act_ckpt < act_full
    assert act_full == act_ckpt * 10  # factor 20 vs 2

    assert human_readable(1_600_000_000).endswith("GB")

    print(
        f"[PASS] dist_memory_calc: 1亿参数 fp16+Adam 总显存={human_readable(breakdown['total_bytes'])}"
        f"（优化器状态占比 {breakdown['optimizer_bytes'] / breakdown['total_bytes']:.0%}）"
    )


if __name__ == "__main__":
    _self_test()
