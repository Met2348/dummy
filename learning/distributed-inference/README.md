# Topic 5: Distributed Inference（分布式推理）

> Module 5 「用大模型」第 5 专题 · 12 lectures · 12 notebooks · ~14h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 分布式推理全图 | — |
| L02 | Tensor Parallel | `tp_demo.py` |
| L03 | Megatron-style TP | `tp_demo.py` |
| L04 | Pipeline Parallel | `pp_demo.py` |
| L05 | 1F1B schedule | `pp_demo.py` |
| L06 | Expert Parallel (MoE) | `ep_demo.py` |
| L07 | All-to-All 通信 | `ep_demo.py` |
| L08 | **Disaggregated P/D** ⭐ | `disaggregated_mock.py` |
| L09 | KV cache 跨节点 | `kv_transfer_mock.py` |
| L10 | Prefix-aware routing | `routing_policies.py` |
| L11 | 多节点部署 (Ray Serve) | — |
| L12 | **Capstone: Disagg 3-config** ⭐ | `capstone_disagg.py` |

## Tags

- `distrib-infer` — 最终（含 Capstone + README）

## Capstone 模拟实测

```
| config | TTFT ms | TPOT ms | tok/s | wall s |
|---|---|---|---|---|
| colocate | 307.2 | 8.0 | 384.6 | 10.65 |
| disagg-near | 307.5 | 8.0 | 769.1 | 5.33 |
| disagg-remote | 312.4 | 8.0 | 766.2 | 5.35 |
```

→ disagg-near 比 colocate **吞吐 +100%**（模拟数据）。

## 并行策略决策

| 模型 | TP | PP | EP | 节点数 |
|------|----|----|----|--------|
| 7B | 1-2 | 1 | - | 1 |
| 70B | 8 | 1 | - | 1 |
| 405B | 8 | 2 | - | 2 |
| **DeepSeek-V3 671B** | 8 | 2 | 64 | 8 |

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试 (16/16 全绿)
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_distrib"

# Capstone
python src/capstone_disagg.py
```

## 退出条件 checklist

- [x] 12 lecture + 12 notebook
- [x] 16 tests pass (TP/PP bubble/EP/disagg/routing 全验证)
- [x] Capstone 3 config 表
- [x] git tag `distrib-infer` ✓

## 关键文献

- Megatron-LM (Shoeybi 2020)
- DistServe (PKU OSDI'24)
- Mooncake (Moonshot 2024)
- DeepSeek-V3 (DeepSeek 2024.12)
- DualPipe schedule

## 一句话

> 7B 单 GPU 够、70B 用 TP 8、405B+ 加 PP、MoE 加 EP、长 prompt 必须 disaggregated。
