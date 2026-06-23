# 13.2 flow-matching-sota — 攻扩散的 T 步慢, 少步生成

> **Module 13「扩散/生成式媒体/世界模型」· 第 2 专题 (加速)**
> M13.1 的 DDPM 要迭代 T 步采样, 慢。本专题用 flow matching / rectified flow / consistency 把生成压到 1-4 步, 是 2024-26 SOTA 采样范式。你 EE 的 ODE 直觉是优势。

---

## 这个专题要解决的真问题

```
   DDPM: 上千步去噪采样, 慢 (M13.1 命门)
   flow matching: 学速度场, ODE 积分, 路径可拉直 → 少步生成
```

- **L1 flow matching**: 不学去噪, 学**速度场** $v_\theta(x,t)$, ODE 积分采样。
- **L2 rectified flow**: reflow 把路径**拉直** → 1-4 步。
- **L3 consistency**: 直接训**一步映射**。
- **L4 采样器全景**: 统一视角 (路径随机/直度) + 怎么选。

> 速度是扩散从「能生成」到「能落地」(实时 omni M10.5 / 机器人 M11.3) 的命门。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-ddpm-to-flow-matching.md` | 学速度场而非去噪; ODE 视角 (公式逐项) |
| L2 | `lectures/L2-rectified-flow.md` | reflow 拉直路径, 逼近 1 步 |
| L3 | `lectures/L3-consistency-models.md` | 直接训一步映射 (一致性) |
| L4 | `lectures/L4-sampler-landscape.md` | 采样器全景地图 + 决策树 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-flow-vs-ddpm.ipynb` | 用 `src/flow_matching.py` 训速度场, 不同步数采样, 看少步也能生成 (质量 vs 步数曲线) |
| `notebooks/N2-rectified-flow.ipynb` | 对比 flow matching vs 一轮 reflow: reflow 后用更少步达同等质量 (std_err 0.24→0.05) |

## 工具 (`src/`)
- `flow_matching.py` — 速度场训练 + ODE 采样 + **reflow** + 质量-步数曲线 (torch CPU)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / numpy / matplotlib。2D 玩具分布离线确定性。

## 完成本专题后你应该能
- [ ] 解释 flow matching 学速度场 + ODE 采样 (vs DDPM 去噪)
- [ ] 说清「路径越直步数越少」+ rectified flow 怎么拉直 (reflow)
- [ ] 区分 rectified flow vs consistency 两条一步生成路
- [ ] 用采样器全景地图按约束选方法
- [ ] 在 2D 上验证 reflow 后少步达同等质量

---
## 在 Module 13 中的位置
```
  13.1 扩散地基 → 13.2 加速 ◄你在这 → 13.3 DiT → 13.4 视频 → 13.5 世界模型 → 13.6 dLLM → 13.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module13-diffusion-world-models-design.md`
