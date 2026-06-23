# Module 10「多模态 / VLM 基础」设计 spec

> Date: 2026-06-24 · 用户 (博0, EE 本硕, 主做 NLP/LLM, 明牌未来方向 = VLM)
> 终态: 在 48 工程专题 + Module 9 科研技能之上, 新增 Module 10 —— 从纯文本 LLM 跨到**视觉-语言基础模型 (VLM)**, 补齐用户体系里最大的能力空白。

## 1. 背景与动机

用户的 portfolio 在文本 LLM 上极深 (预训练/后训练/推理/评测/agent/infra 全覆盖), 但**多模态只有 `multimodal-agent` 一个应用层专题, 没有 VLM 的训练与架构**。而:
- **2026 行情**: 多模态 (文/图/音/视频) 被点名为 "top emerging area"; 多模态推理是 frontier 实验室竞价专长之一 (来源见 portfolio_v4 招聘核对)。
- **用户明牌**: 自述「日后可能做 VLM」; 这是他研究方向的天然延伸。
- **杠杆**: VLM 训练 ≈ LLM 训练 + 一个视觉塔 + 一个连接器。用户的 transformer/pretraining/instruction-tuning 知识**直接迁移**, 不是从零。
- **地基地位**: M10 是 M11 (VLA = VLM + action head) 和 M13 (视频/世界模型) 的前置。**先建 M10。**

### 关键洞察
1. VLM 的核心设计空间是「**视觉如何接入 LLM**」—— cross-attention vs 投影 adapter vs early-fusion。讲清这个设计空间是本模块的灵魂。
2. 用户要的是**能跑**: 所有 notebook 用小尺度 (tiny ViT + tiny LLM + 合成/小图) 在 CPU/单卡跑通, 复用 Module 9 的「确定性 + 离线可跑」纪律。
3. 评测复用: VLM 幻觉/评测直接接 Module 6 (评测安全) 和 9.6 (出版级图)。

## 2. 专题蓝图 (7 专题, 按 encoder→fusion→train→generate→multimodal→eval→capstone)

| # | slug | 覆盖技能 | 核心产出 |
|---|---|---|---|
| 10.1 | `vision-encoders` | 视觉编码器: ViT / CLIP / SigLIP / DINOv2, 对比预训练, patch embedding, 图像 tokenization | 手搭 mini-ViT + 对比损失, 理解视觉表示 |
| 10.2 | `vl-fusion-architectures` | VL 融合设计空间: cross-attn(Flamingo) / 投影(LLaVA) / early-fusion(Chameleon/Fuyu) / Q-Former | 一张「连接器」决策树 + 三种连接器对比 |
| 10.3 | `vlm-training-recipe` | VLM 训练: 预训练 + 指令微调, 数据(图文对/交错), 冻结策略, LLaVA 配方 | 端到端训一个 mini-VLM (tiny encoder+LLM) |
| 10.4 | `visual-tokenization-generation` | 离散视觉 token(VQ-VAE/VQGAN), 统一理解+生成(Chameleon/Transfusion), any-to-any | 手搭 VQ codebook, 从 token 重建图像 |
| 10.5 | `video-audio-omni` | 视频编码(时序), 音频(Whisper 式), 语音, omni 模型, 实时多模态 | 时序/音频 token 化的最小实现 |
| 10.6 | `vlm-eval-hallucination` | VLM benchmark(MMMU/MME/POPE), 视觉幻觉, 评测陷阱 | 一个视觉幻觉探测 + 评测脚本 (接 M6/9.6) |
| 10.7 | `multimodal-graduation` | Capstone: 装配一个 mini-VLM, 接回 `multimodal-agent` | 端到端 mini-VLM + 研究 gap 卡 |

## 3. 逐专题详细设计

### 10.1 vision-encoders
- **lectures (4)**: L1 从像素到 token (ViT/patch embedding, 为什么 transformer 也能吃图) · L2 对比学习与 CLIP (图文对齐, InfoNCE 损失逐项交代) · L3 SigLIP/DINOv2 (sigmoid loss / 自监督视觉) · L4 视觉表示的性质 (冻结塔 vs 微调, 哪层适合喂 LLM)
- **notebooks (2)**: N1 手搭 mini-ViT + 在合成图上跑 patch embedding 可视化 · N2 mini-CLIP 对比损失训练 (tiny 图文对), 看图文相似度矩阵
- **src**: `tiny_vit.py` (patch embed + transformer block, numpy/torch) · `contrastive.py` (InfoNCE/sigmoid loss + 相似度矩阵)

### 10.2 vl-fusion-architectures
- **lectures (4)**: L1 核心问题「视觉怎么接进 LLM」+ 设计空间总览 · L2 cross-attention 路线 (Flamingo/perceiver resampler) · L3 投影 adapter 路线 (LLaVA: 一个 MLP 就够? 为什么 work) · L4 early-fusion / 统一 token (Chameleon/Fuyu) + 决策树
- **notebooks (2)**: N1 三种连接器在同一 tiny setup 上的参数量/信息流对比 · N2 实现 LLaVA 式投影连接器, 把 mini-ViT 接到 tiny LLM
- **src**: `connectors.py` (cross_attn / projection / early_fusion 三种连接器 + 对比表)

### 10.3 vlm-training-recipe
- **lectures (4)**: L1 VLM 训练两阶段 (对齐预训练 + 指令微调) · L2 数据 (图文对/交错文档/指令数据) + 冻结策略 (先冻 LLM 训连接器, 再解冻) · L3 LLaVA 配方逐步拆 · L4 训练陷阱 (模态坍缩/灾难遗忘/数据配比)
- **notebooks (2)**: N1 端到端训一个 mini-VLM (tiny ViT + tiny LLM + 投影器, 合成图文指令) · N2 冻结策略消融 (接 9.4 实验设计: 冻 vs 不冻对模态对齐的影响)
- **src**: `mini_vlm.py` (组装 encoder+connector+LLM 的最小可训 VLM) · 复用 9.5 exp_tracker 留痕

### 10.4 visual-tokenization-generation
- **lectures (4)**: L1 离散视觉 token (VQ-VAE/VQGAN, codebook, 为什么离散化) · L2 用 LLM 生成图像 (image token 自回归) · L3 统一理解+生成 (Chameleon early-fusion / Transfusion 混合) · L4 any-to-any 的设计权衡
- **notebooks (2)**: N1 手搭 VQ codebook, 把小图编码成 token 再重建 (看重建质量随 codebook 大小变化) · N2 在视觉 token 上跑自回归生成 (tiny, 看采样)
- **src**: `vq_tokenizer.py` (向量量化 + codebook + 重建)

### 10.5 video-audio-omni
- **lectures (4)**: L1 视频 = 图像 + 时序 (时序 attention / 帧采样 / 时空 patch) · L2 音频与语音 (mel 谱 / Whisper 式编码 / 音频 token) · L3 omni 模型 (一个模型吃所有模态) · L4 实时多模态的工程约束 (接回 harness-engineering)
- **notebooks (2)**: N1 视频帧的时空 token 化最小实现 · N2 音频 mel 谱 → token 的最小管线
- **src**: `temporal_tokens.py` (时空 patch) · `audio_features.py` (mel + 分帧)

### 10.6 vlm-eval-hallucination
- **lectures (4)**: L1 VLM 评测的特殊性 (为什么文本 benchmark 不够) · L2 主流 benchmark (MMMU/MME/POPE/幻觉测) · L3 视觉幻觉机理与探测 · L4 评测陷阱 + 诚实报告 (接 9.3 攻击清单 + 9.6 诚实图)
- **notebooks (2)**: N1 实现一个 POPE 式视觉幻觉探测 (问图里有没有某物, 测 yes-bias) · N2 VLM 评测结果出版级图 (复用 9.6 plotstyle)
- **src**: `vlm_eval.py` (幻觉探测 + 指标), 复用 M6 eval 工具思路

### 10.7 multimodal-graduation (Capstone)
- **lectures (2)**: L1 把 10.1-10.6 装配成一个完整 mini-VLM 流水线 · L2 VLM 研究前沿 + 用 9.3 gap 雷达扫多模态研究题目 (接 PhD 方向)
- **notebooks (2)**: N1 端到端: 图 → mini-ViT → 连接器 → tiny LLM → 文字回答 (全跑通) · N2 用 9.3/9.4 把一个多模态 gap 起成 idea 卡
- 接回用户已有 `multimodal-agent`: 现在他不只会**用** VLM, 还懂 VLM 怎么**造**出来。

## 4. 与现有资产整合
- **复用 Module 9**: 每个 notebook 用 9.5 的可复现纪律 + exp_tracker; 评测出图用 9.6 plotstyle; Capstone 用 9.3/9.4 找 gap 设计实验。
- **复用 M3/M4**: VLM 的 LLM 主体直接用用户 transformer-deep/pretraining 的知识; 指令微调接 M1 PEFT。
- **接 M11/M13**: M10 是地基 —— VLA (M11) = M10 VLM + action head; 视频生成 (M13) 用 M10 的视觉 token 化。
- **接 multimodal-agent**: 从「用 VLM 做 agent」补上「造 VLM」。

## 5. 成功标准
- [ ] 7 专题完整落地, 结构同 Module 9 (README+lectures+notebooks+templates+src+environment)。
- [ ] 全部 notebook nbconvert 跑通 0 报错, 小尺度 CPU/单卡可跑。
- [ ] 课件研究生级: 图 + 清单 + 公式逐项交代 (尤其 InfoNCE/VQ/cross-attn)。
- [ ] 至少一个 notebook 端到端训出可问答的 mini-VLM。
- [ ] Capstone 用 9.3 gap 雷达产出多模态研究 idea 卡。
- [ ] portfolio 更新, 体现「文本 → 多模态」能力扩张。
