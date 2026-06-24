# 13.4 video-generation — 扩散加时间维: 视频生成

> **Module 13「扩散/生成式媒体/世界模型」· 第 4 专题 (时间维)**
> M13.1-13.3 让你懂图扩散 (机制/速度/骨架)。本专题把扩散扩到**视频** = 加时间维。核心难题是**时序连贯**。Sora = 你学过的部件 (时空 patch + DiT + latent) + 规模。

---

## 这个专题要解决的真问题

- **视频和图差在哪?** → 多了**时间维**; 灵魂是**帧之间的连贯 (运动)**。
- **为什么不能逐帧生成?** → 帧独立 → 抖动/闪烁; 必须**时空联合建模** (帧互相 attend)。
- **Sora 怎么做的?** → **latent 视频扩散 (3D VAE)** + **时空 patch** (接 M10.5) + DiT + CFG。
- **难在哪?** → 长程一致、物理合理性、闪烁 (open 问题); 评估是瓶颈。
- **为什么是少数玩家?** → token 数爆炸 → 训练百万美元级 (接 M8 基础设施)。

> **核心信念再验证**: Sora = 时空 patch (M10.5) + DiT (M13.3) + latent (M13.3) + 时空扩散。每块你都学过, 只差规模。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-video-as-spacetime.md` | 视频扩散 = 图扩散 + 时间维; 连贯来自时空联合建模 (非免费) |
| L2 | `lectures/L2-sora-spacetime-patches.md` | Sora = latent 视频扩散 (3D VAE) + 时空 patch (接 M10.5) |
| L3 | `lectures/L3-temporal-coherence-challenges.md` | 连贯分尺度: 帧间(易)→长程一致/物理合理(难, open) |
| L4 | `lectures/L4-cost-engineering.md` | 成本驱动架构: token 爆炸 → 百万美元级 (接 M8 + M13.2 少步采样) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-video-diffusion.ipynb` | 用「运动轨迹」当视频, 训时空 vs 逐帧两个扩散模型, 看连贯从哪来 |
| `notebooks/N2-temporal-coherence.ipynb` | 把连贯量化成度量: 帧间跳变 (易) vs 长程一致 (难), 体会分尺度 |

## 工具 (`src/`)
- `video_diffusion.py` — 视频 (时空) 扩散: 运动轨迹数据 + 时空/逐帧去噪器 + 训练/采样 + 连贯度量 (torch CPU)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。玩具运动轨迹离线确定性。

> **为什么这里用 toy 而非真实视频模型?** 真实视频扩散 (Sora) 推理需 GPU 集群, CPU 跑不动。但**机制完全一样**: 把 toy 的「2D 点轨迹」换成「latent 时空 patch」就是 Sora (L2)。toy 让你在秒级看清时空建模的本质。

## 完成本专题后你应该能
- [ ] 解释视频扩散 = 图扩散 + 时间维, 机制不变
- [ ] 说清为什么逐帧独立会抖动, 时空联合才连贯
- [ ] 拆解 Sora: 时空 patch + DiT + latent + 文本条件, 每块来自哪个专题
- [ ] 列举时序连贯的多尺度 open 难题 (长程一致/物理/闪烁)
- [ ] 解释成本如何驱动视频架构 (省 token = 省钱)

---
## 在 Module 13 中的位置
```
  13.1 地基 → 13.2 加速 → 13.3 骨架 → 13.4 视频(时间维) ◄你在这 → 13.5 世界模型 → 13.6 dLLM → 13.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
