# Interview-Prep Track（研究主线 / coding 保险）

> 定位:**不是第 86 个学习专题**。85 专题给了广度;本 track 把知识变成**面试可演示的表现**——限时、口头、真框架。
>
> 目标画像:**前沿实验室研究岗(RS/RE)+ PhD 暑研为主,不走 heavy-coding;coding 作"技术面不翻车"的保险层**。

## 为什么建它

你的 portfolio 全是 stdlib-only 的 mock——**教学一流,但面试考的是另一件事**:能不能在 30 分钟里用**真 PyTorch** 手写、报出**真数字**、当着人**讲清楚**。这个 track 补的是这层"表现差",不是知识差。

## 配比:研究 70% / coding 保险 30%

| 层 | 模块 | 状态 |
|----|------|------|
| **保险·真框架** | `src/mlcoding/` — 真 torch 从零 9 件(attention/norm/rope/block/sampling/kv_cache/lora/training_loop/bpe) | ✅ 真跑真反传 |
| **保险·算法** | `src/leetcode/` — 15 高频 pattern 范例解 + 间隔复习 tracker | ✅ |
| **薄底·基础** | `src/mlqa/` — 45 题/8 类 rapid-fire + 关键词自评（完整扩展版 306 题/23 类见 `learning/baguwen-mastery/`，含后端通用八股） | ✅ |
| **主线·研究** | `docs/.../interview-research-hero-project.md` — 对接你 PhD 仓的真复现 spec | 📋 spec(不硬建) |

## mlcoding —— 面试保险 × 研究肌肉双用

每个模块 `_self_test()` **真跑**,能对拍处与 torch 官方对齐(回答"你怎么知道你写对了"):

| 模块 | 高频度 | self_test 证明了什么 |
|------|:---:|------|
| `attention.py` | ★★★★★ | SDPA 对拍 `F.scaled_dot_product_attention` + 因果不泄露 + MHA 反传 |
| `norm.py` | ★★★★ | LayerNorm 对拍 `F.layer_norm` + RMSNorm 单位 RMS |
| `rope.py` | ★★★★ | 保范数 + `<RoPE(q,m),RoPE(k,n)>` 只依赖相对位置 |
| `transformer_block.py` | ★★★★ | 预归一化 block + 残差直通(零权重时 out==in)+ 因果 |
| `sampling.py` | ★★★★★ | greedy/top-k/top-p/温度熵 + beam 破"贪心陷阱" |
| `kv_cache.py` | ★★★★ | **逐 token 缓存 == 全量前向**(err 1e-7) |
| `lora.py` | ★★★★ | 初始等价原层 + 仅低秩可训 + 基座冻结 |
| `training_loop.py` | ★★★★★ | zero_grad→backward→clip→step 四步 + warmup/cosine 边界 + 真拟合 |
| `bpe.py` | ★★★ | 训练 merges + encode/decode 往返还原 |

## 环境（torch 是全仓唯一破例）

```powershell
$env:PYTHONIOENCODING="utf-8"
pip install torch numpy          # 本 track 需要真框架
python learning/interview-prep/environment/verify_env.py
```

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/interview-prep/src/tests/test_all.py     # 预期 12/12
```

单模块直接跑也可(每个都有 `__main__`),例如:
```powershell
python learning/interview-prep/src/mlcoding/attention.py
python learning/interview-prep/src/leetcode/patterns.py
python learning/interview-prep/src/mlqa/qbank.py
```

## Lectures

| # | 主题 |
|---|------|
| 01 | 研究岗面试全景 + 你的 70/30 配比 |
| 02 | mlcoding 打法:attention 家族(得分点在细节) |
| 03 | mlcoding 打法:解码采样 + KV cache 不变量 |
| 04 | LeetCode 15 pattern 识别地板 |
| 05 | rapid-fire:30 秒说清的刻意练习 |
| 06 | ML 系统设计(研究向格式) |
| 07 | 研究面试:论文 drill + 开放式推理 + hero project |
| 08 | behavioral:把 85 专题讲成 impact |
| 09 | 研究 hero-project spec(对接 PhD 仓,不硬建) |

## 与全仓其它内容的边界

- 不重复 `interp-foundations`/`probing-and-activations` 等**概念**专题——本 track 是把概念**在真模型上跑出数字 + 讲出来**。
- 研究 hero project **对接已有 `research/` 仓**(70-paper repo + 24-src audit + PhD judge-internals 方向),不另起炉灶。

## 一句话

> 研究是矛,coding 是盾。这个 track 把盾打厚(真 torch + pattern 地板 + 基础秒答),把矛的复现留给你的 PhD 仓。
