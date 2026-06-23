# Module 13 扩散/世界模型 Implementation Plan

**Goal**: 新建 Module 13「扩散/生成式媒体/世界模型」(7 专题), 补齐扩散范式空白。2D 玩具分布 + tiny 图, CPU 上把 DDPM→flow matching→DiT→视频→世界模型→dLLM 全跑通, 可视化去噪轨迹。

**Design**: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`

**Tech Stack**: Python 3.13 / torch (tiny) / numpy / matplotlib / nbformat. 确定性, 玩具分布 (双月/螺旋) + 合成/MNIST 级小图离线可跑。

**依赖**: **独立可建**; 是 M11 (diffusion policy 11.3 / world model 11.5) 的理论前置。**建议在 M11 之前建** (让 M11 动作头/世界模型有理论支撑)。`diffusion.py`/`world_model.py` 与 M11 同源共享。

**构建顺序 (模块内)**: 13.1 foundations → 13.2 flow matching → 13.3 DiT → 13.4 video → 13.5 world models → 13.6 dLLM → 13.7 capstone。

---

## Phases
- **P1 `diffusion-foundations`**: diffusion.py (DDPM, 与 M11 同源); N1 2D 去噪轨迹可视化 / N2 tiny 图 DDPM 生成
- **P2 `flow-matching-sota`**: flow_matching.py; N1 flow vs DDPM 步数/质量 / N2 rectified flow 拉直
- **P3 `dit-latent-diffusion`**: dit.py (DiT+CFG); N1 tiny DiT 条件生成 / N2 CFG 强度消融 (复用 9.4)
- **P4 `video-generation`**: video_diffusion.py; N1 玩具运动序列时空扩散 / N2 时序连贯度量+出图
- **P5 `world-models`**: world_model.py (动作条件, 与 M11 共享); N1 学世界模型+想象 rollout / N2 预测质量评测
- **P6 `diffusion-language-models`**: diffusion_lm.py; N1 masked diffusion LM 并行生成 / N2 dLLM vs AR 对比
- **P7 `generative-media-graduation`**: Capstone; N1 端到端条件扩散生成 / N2 扩散/dLLM gap→idea 卡

## 成功标准
- [ ] 7 专题完整, verify_env 全过, 14 notebook 0 报错 (玩具分布/tiny 图 CPU 可跑)。
- [ ] 至少一个 notebook 可视化去噪轨迹 (看见扩散在干什么)。
- [ ] dLLM 专题让用户在 NLP 本行理解扩散。
- [ ] 课件公式逐项 (前向加噪/score matching/flow/CFG)。
- [ ] Capstone 产出扩散/dLLM/世界模型 idea 卡。
