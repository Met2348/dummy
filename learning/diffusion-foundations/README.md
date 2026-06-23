# 13.1 diffusion-foundations — 扩散模型: 从噪声去噪生成

> **Module 13「扩散/生成式媒体/世界模型」· 第 1 专题 (扩散地基)**
> 补齐你体系的范式级空白: 全 portfolio 是自回归, 完全没有扩散。本专题在 2D 玩具分布上让你**看见**扩散在干什么。也被 M11 diffusion policy 复用。

---

## 这个专题要解决的真问题

你精通自回归生成 (LLM)。但生成有**另一条路**: 扩散 (Stable Diffusion/Sora/机器人 diffusion policy 的核心)。

```
   自回归: 一个个生成 token (你已会)
   扩散:   从纯噪声逐步去噪成数据 (本模块)
           前向加噪 (固定) → 学反向去噪 → 生成 = 从噪声去噪
```

> 核心: 把「生成复杂分布」(难) 拆成「逐步去一点噪声」(每步简单)。你会在 2D 双月上**亲眼看见**点云从噪声收敛成数据 (去噪轨迹)。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-two-paths-to-generation.md` | 自回归 vs 扩散; 扩散像「雕塑」 |
| L2 | `lectures/L2-forward-noising.md` | 前向加噪: 马尔可夫链 + 一步闭式 (公式逐项) |
| L3 | `lectures/L3-reverse-denoising-score.md` | 反向去噪 = 预测噪声 = 估计 score (训练目标) |
| L4 | `lectures/L4-ddpm-sampling-tradeoffs.md` | 采样 T 步: 质量 vs 速度 (命门, 铺垫 flow matching) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-2d-diffusion.ipynb` | 用 `src/diffusion.py` 在 2D 双月上跑 DDPM: 可视化前向加噪 + 训练去噪网络 + **去噪轨迹** (点云从噪声→双月) |
| `notebooks/N2-image-diffusion.ipynb` | 扩到 tiny 合成图, 训图像 DDPM 生成, 对比采样步数 (50 vs 10) 质量 |

## 工具 (`src/`)
- `diffusion.py` — 最小 DDPM (前向闭式/去噪网络/训练/反向采样/去噪轨迹), 2D + tiny 图, torch CPU 秒级。**M11 diffusion policy 复用。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。2D 玩具分布离线确定性。

## 完成本专题后你应该能
- [ ] 对比自回归 vs 扩散两条生成路
- [ ] 推导前向加噪闭式, 逐项解释 β/α/ᾱ
- [ ] 解释「预测噪声 = 估计 score」的训练目标
- [ ] 说清采样 T 步的质量 vs 速度权衡
- [ ] 在 2D 上跑通 DDPM 并看懂去噪轨迹

---
## 在 Module 13 中的位置
```
M13 扩散/世界模型:
  13.1 diffusion-foundations  ◄你在这 (地基)
  13.2 flow-matching (加速) → 13.3 DiT → 13.4 视频 → 13.5 世界模型 → 13.6 dLLM → 13.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
