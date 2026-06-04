# L08 · Disaggregated Prefill/Decode（DistServe / Mooncake）⭐

## 1 · 痛点
传统 colocate：同 GPU 既 prefill 又 decode。
- prefill compute-bound（长 prompt）
- decode memory-bound（KV 读）
- 共存 → 互相干扰，SLO 难达

## 2 · 解：物理分离
- **Prefill cluster**: 一组 GPU 专做 prefill (高 compute/带宽比)
- **Decode cluster**: 另一组专做 decode (大显存装 KV)

数据流：
```
User → Router → Prefill GPU → 生成 KV
                         ↓
            KV cache 传输 (NVLink/NVSwitch/RDMA)
                         ↓
              Decode GPU → 流式生成 token → User
```

## 3 · 论文 / 系统
- **DistServe** (PKU OSDI'24)：开山
- **Mooncake** (Moonshot 2024.06)：Kimi 商业实践
- **vLLM 0.7+**: 开源支持
- **SGLang**: 后跟进

## 4 · 收益
| 指标 | colocate | disaggregated |
|------|---------|---------------|
| TTFT | 中 | **快** (prefill 不被 decode 干扰) |
| TPOT | 中 | **快** (decode 独占资源) |
| goodput | 1x | **1.5-2x** |

## 5 · KV cache 传输代价
- 7B 8k ctx KV ≈ 8 GB
- NVLink 4 900 GB/s → 9 ms (不能忽略)
- NVSwitch / NVSHMEM 直接 P2P
- 对 long prompt 才划算（KV 大但 decode 久）

## 6 · 跨节点
- 同节点：NVLink + NVSwitch
- 跨节点：InfiniBand RDMA / NVSHMEM
- 跨节点 KV 传输：10-50 ms (vs 同节点 1-10 ms)
- → 跨节点 disaggregated 用于离线/批

## 7 · 配置
```
# vLLM 0.7
python -m vllm.entrypoints.openai.api_server \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector",...}' \
    ...
```

## 8 · 实现：[disaggregated_mock.py](../src/disaggregated_mock.py)
- 单机模拟 P/D 分离
- 估算 KV 传输时间
- 对比 colocate
