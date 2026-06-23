# 10.1 vision-encoders — 视觉编码器: 图像如何变成 token

> **Module 10「多模态/VLM 基础」· 第 1 专题 (视觉地基)**
> VLM = 一个懂语言的视觉塔 + 一个 LLM + 一个连接器。本专题造**视觉塔**: 把图像变成 LLM 能吃的、且「懂语言」的 token。

---

## 这个专题要解决的真问题

你已会 transformer 处理文本 (天然 token 序列)。但图像是连续像素网格, 没有现成 token。**怎么让吃 token 的 transformer 吃图、还让它的视觉表示「懂语言」?**

- **ViT** 的答案: 把图切成 patch、当 token (L1)。
- **CLIP/SigLIP** 的答案: 用海量图文对做对比学习, 让视觉表示和语言对齐 (L2/L3)。
- **DINOv2** 的答案: 纯自监督, 不用文本也学出强视觉表示 (L3)。

> 这是整个 Module 10 的地基。后面 10.2 (怎么把视觉塔接进 LLM)、10.3 (怎么训 VLM)、乃至 M11 VLA、M13 视频, 都建在「图像→懂语言的 token」之上。

```
   图像 → [patchify] → patch → [ViT] → 视觉 token → (对比学习让它懂语言)
                                            └→ 接进 LLM (M10.2)
```

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-pixels-to-tokens.md` | ViT: patchify + patch embedding, 图变 token 序列 | 视觉 token |
| L2 | `lectures/L2-contrastive-clip.md` | 对比学习 + CLIP 的 InfoNCE, 让视觉懂语言 | 图文对齐 |
| L3 | `lectures/L3-siglip-dinov2.md` | SigLIP (sigmoid 损失) + DINOv2 (自监督), 三塔选型 | 视觉塔决策 |
| L4 | `lectures/L4-visual-representations.md` | 接进 LLM 前的接口决策 (CLS/层/冻结) | VLM 视觉接口 |

## 动手 (2 个 notebook)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-vit-patchify.ipynb` | 用 `src/tiny_vit.py` 把合成图 patchify + 可视化 16 个 patch, 跑 mini-ViT 看视觉 token 形状, 对比 [CLS] vs patch token |
| `notebooks/N2-mini-clip.ipynb` | 用 `src/contrastive.py` 算图文相似度矩阵 + InfoNCE/sigmoid 损失, 改配对噪声看「对齐越差损失越高」 |

## 工具 (`src/`)

- `tiny_vit.py` — 最小 ViT (patchify + patch embedding + CLS + transformer), 合成图离线可跑
- `contrastive.py` — InfoNCE (CLIP) + sigmoid (SigLIP) 损失 + 图文相似度矩阵

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / torch (tiny 模型 CPU 秒跑) / numpy / matplotlib。合成图离线可跑, 无需真实数据集。

## 完成本专题后你应该能 (产出 checklist)
- [ ] 解释 ViT 怎么把图变成 token 序列 (patchify→embedding→CLS→transformer)
- [ ] 推导 InfoNCE 损失, 说清 temperature / 负样本 / 大 batch 的关系
- [ ] 对比 CLIP/SigLIP/DINOv2, 给造 VLM 选对视觉塔
- [ ] 说清接 LLM 前的三个接口决策 (CLS vs patch / 哪层 / 冻不冻)

---

## 在 Module 10 中的位置
```
M10 多模态/VLM:
  10.1 vision-encoders        ◄── 你在这里 (视觉塔地基)
  10.2 vl-fusion-architectures    (视觉怎么接进 LLM)
  10.3 vlm-training-recipe        (训一个 VLM)
  10.4-10.7 生成/视频音频/评测/capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
