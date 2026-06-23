# Modules 10-13 扩展路线图 (前沿 4 模块, 2026-06-24)

> 在 48 工程专题 (M1-8) + Module 9 科研技能 + harness-engineering 之上, 新增 4 个 2026 前沿模块, 补齐用户体系的全部主要空白。用户决策: **4 个全建, 规划完成后一口气全自动做完**。

## 为什么这 4 个 (缺口 × 2026 高薪 × 适配用户)

| 模块 | 缺口 | 2026 行情 | 适配用户 |
|---|---|---|---|
| **M10 多模态/VLM** | 只有应用层 multimodal-agent, 无 VLM 训练 | top emerging area | 明牌未来方向 + 一切的桥 |
| **M11 具身/VLA** | 完全空白 | 2026 具身基石, ICLR 164 篇 | 已有 IsaacLab 文件 + NLP 最自然转型 |
| **M12 机制可解释性** | 完全空白 | MIT 2026 十大突破 | frontier 安全实验室硬通货 + EE 数学优势 |
| **M13 扩散/世界模型** | 范式级空白 (全是自回归) | ICLR 主线 + dLLM | dLLM 直连 NLP 本行 |

来源核对见 `portfolio_v4.md` (2026-06-24 web 核对当下招聘)。

## 模块依赖图

```
            M10 多模态/VLM (地基)
           /                  \
   M11 具身/VLA            M13 扩散/世界模型
   (VLM backbone +         (图/视频/世界/dLLM)
    action head)               |
        ↑__________ 共享 diffusion.py / world_model.py
                 (M11 的动作头/世界模型用 M13 理论)

   M12 机制可解释性 (独立, 横切所有模型, 接 reasoning-r1)
```

## 全局构建顺序 (依赖驱动)

**M10 → M13 → M11 → M12**

1. **M10** 先建: 是 M11/M13 的视觉地基。
2. **M13** 次建: 独立, 且提供 M11 动作头(diffusion policy)/世界模型的理论与共享 src。
3. **M11** 第三: 复用 M10 VLM backbone + M13 扩散/世界模型。
4. **M12** 最后: 完全独立, 最偏纯研究, 收尾时建 (接 reasoning-r1 当 PhD 候选方向)。

每模块 7 专题 × (4-5 讲 + 2 notebook + 2 src + 模板 + env), 同 Module 9 标准: 研究生级课件 + nbconvert 全跑通 + 小尺度 CPU 可跑 + 复用 9.3/9.4/9.5/9.6 的找gap/实验/复现/出图工具。

合计: **4 模块 × 7 专题 = 28 新专题** (体系: 48 工程 + 9 研究 + 28 前沿 = 85 专题 + harness 双栖)。

## 设计文档索引
- specs: `docs/superpowers/specs/2026-06-24-module1{0,1,2,3}-*-design.md`
- plans: `docs/superpowers/plans/2026-06-24-module1{0,1,2,3}-*.md`

## 收尾 (28 专题建完后)
- portfolio 升 v5: 加「多模态/具身/可解释/生成」4 大新画像; 新增 robotics/interp 研究轨道。
- 更新 memory (新增 modules10-13-frontier 记忆)。
