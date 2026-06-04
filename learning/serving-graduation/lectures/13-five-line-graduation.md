# L13 · 五线综合 lecture ⭐⭐⭐⭐⭐⭐

90 分钟，30 slides。

---

## Part I：18 专题地图回顾 (20 min, 6 slides)

### Slide 1 — 4 大 Module 全图
```
+--------------------+
| Module 1 PEFT      |  prompt-tuning / lora / adapter (3 topic)
+--------------------+
| Module 3 造大模型  |  data / arch / weight / ctx / infra (8 topic)
+--------------------+
| Module 4 改大模型  |  RL + 对齐 + R1 (7 topic)
+--------------------+
| Module 5 用大模型  |  服务 / 推理 / 量化 / 部署 (7 topic, 含本)
+--------------------+
```

### Slide 2 — 25 个 topic 覆盖
- 全部 200+ 方法
- 总学时 ~330h

### Slide 3 — 「造-改-用」三角形
```
       造 (Module 3)
       /       \
     /           \
   改 (Module 4) — 用 (Module 5)
```

### Slide 4 — 每条边的代表方法
- 造↔改: pretrain → SFT → RL
- 造↔用: 训出模型 → 量化部署
- 改↔用: agent 用 ckpt → routing

### Slide 5 — 商业 vs 开源
- 商业 (Claude/GPT/Gemini): 不用知道造+改，只用接 API + cost
- 开源: 必须造-改-用全栈

### Slide 6 — 个人/企业选择
| 谁 | 重点 |
|----|-----|
| 个人学习 | 全栈 |
| 公司业务 | 用 + 改 (LoRA) |
| AI infra 公司 | 造 + 用 |
| Foundation lab | 造 + 改 |

---

## Part II：服务视角的统一公式 (30 min, 10 slides)

### Slide 7 — 任意 LLM 服务 =
```
Serve(model, quant, schedule, kv_mgmt, spec_decode) → API
```

### Slide 8 — 5 个 lever 的影响
| lever | speedup |
|-------|---------|
| model size↓ | 5-10x |
| quant (FP16→AWQ-4) | 1.6-2x |
| schedule (naive→continuous+chunked) | 5-10x |
| KV mgmt (naive→paged) | 2-3x |
| spec decode (EAGLE-2) | 2-4x |

总: **500-2000x** 理论加速（vs naive 2022 GPT-3 部署）。

### Slide 9 — DeepSeek-V3 100k tok/s 拆解
- model: MoE (671B 但 active 37B)
- quant: FP8 (训推一致)
- schedule: continuous + chunked + 自研 dispatcher
- KV: paged + EP-256
- spec: 暂无（未来加）

### Slide 10 — Claude Sonnet 定价拆解
$3 / M token，估 ≈ 30k tok/s/GPU + $4/h GPU。

### Slide 11-16 — 5 个真实案例的 5-lever 配置

### Slide 17 — 「造改用」对偶
- 训越大越好用 → 用必须越省钱越好
- 训追 PPL → 用追 cost
- 量化是 bridge

---

## Part III：选型决策树 (25 min, 8 slides)

### 4 场景演练

**Slide 18-20: 单用户 chatbot**
- volume: low
- latency: <500ms
- 选: vLLM + AWQ + EAGLE-2

**Slide 21-23: 离线 1M 文档处理**
- volume: huge
- latency: hours OK
- 选: TRT-LLM batch + FP8 + max_batch_size 256

**Slide 24-25: agent (cursor-like)**
- multi-turn + tool
- 选: SGLang + radix + multi-model router

**Slide 26-27: VLM 服务**
- image + text
- 选: vLLM + Qwen2-VL + image cache

---

## Part IV：5 module 闭环 + 下一程 (15 min, 6 slides)

### Slide 28 — 闭环
```
data + arch (Module 3)
    → pretrain → SFT (Module 4)
       → RLHF / R1 (Module 4)
          → quant (Module 5)
             → serve (Module 5)
                → user
                   → 反馈 data
                      ↻ data (Module 3)
```

### Slide 29 — 已掌握 + 缺
| 已学 | 缺 |
|------|---|
| 造 / 改 / 用 全栈 | Eval (Module 6) |
| 训 + 部署 | Safety / red-team |
| 单模型 | Agent application (Module 7) |
| | MLOps 全套 |

### Slide 30 — 下一程候选
- Module 6 评测/安全
- Module 7 Agent 应用层
- 个人作品集
- 商业产品落地

---

## 一句话总结

> **造一个模型是开始，改它让它有用，用它让它赚钱**。Module 5 闭环了「用」，但「用得好」是工程问题，不是 LLM 问题——也是你这 4 个 module 学完真正进入的赛道。
