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

## 运行验证（Runbook）

> 本模块的"可运行入口"即 [`runbook.yaml`](runbook.yaml) 登记的 8 个**单进程模拟** demo，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）V1 验证通过。全部无 `torch.distributed`/多卡依赖，纯 CPU 秒级直跑。
> 一键复验：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules distributed-inference
> ```

8 个 demo 均无需传参（自带场景规模）：

```powershell
# 并行三件套
python learning/distributed-inference/src/tp_demo.py          # Tensor Parallel：列/行切分 + all-reduce
python learning/distributed-inference/src/pp_demo.py          # Pipeline Parallel：微批 + bubble 占比
python learning/distributed-inference/src/ep_demo.py          # Expert Parallel(MoE)：放置 + all-to-all
# Disaggregated serving
python learning/distributed-inference/src/disaggregated_mock.py        # Prefill/Decode 分离(interference 模型)
python learning/distributed-inference/src/kv_transfer_mock.py          # KV-cache 跨节点迁移开销
python learning/distributed-inference/src/distserve_original_minimal.py # DistServe goodput 模拟器(SLO 搜配比)
python learning/distributed-inference/src/routing_policies.py          # 请求路由策略对比
# Capstone
python learning/distributed-inference/src/capstone_disagg.py           # 综合对比 + Markdown 报告
```

> 注（demo 性质，非 bug）：
> - 这 8 个全是**单进程模拟**——用解析/带宽/interference 模型推算并行与 disagg 的延迟/吞吐，**不起真多卡**（3080 Ti 单卡）。要真多卡需 `torchrun` + ≥2 GPU（讲义 §并行策略给了真实命令）。
> - `disaggregated_mock` 的 disagg 增益**由 prefill/decode interference 模型推出**（colocate 的 TPOT 被并发 prefill 拖慢，disagg 分池后无干扰）+ KV 迁移加到 TTFT——非硬编码常数；增益随 prompt 变长而增大、随链路变慢而缩小，与讲义 L08 一致。
> - `tp_demo` 等 7 个脚本原先缺 `__main__`（直跑无输出），本轮补了 `demo()` + `__main__` 使其名副其实可跑。

**测试（V2）**：20 个测试覆盖 TP 通信量 / PP bubble / EP 负载均衡 / disagg interference / DistServe goodput / routing：

```powershell
python -m pytest learning/distributed-inference/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules distributed-inference --tests
```

## 退出条件 checklist

- [x] 12 lecture + 12 notebook
- [x] 20 tests pass (TP/PP bubble/EP/disagg/routing 全验证)
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
