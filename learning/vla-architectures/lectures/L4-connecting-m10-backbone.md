# L4 · 把 M10 VLM backbone 接动作头 = mini-VLA

> 20-min lecture · 目标: 把「装配 VLA」落地 —— 你 M10 的 mini-VLM 怎么接动作头变 mini-VLA。收口 11.2。

---

## 0. 你已造好一半 VLA

L1-L3 讲了 VLA = backbone + 动作头。好消息: **backbone 你 M10 已经造好了** (mini-VLM)。这一讲把最后一步走完: 接上动作头。

---

## 1. mini-VLM (M10.3) 就是 backbone

回看你 M10.3 的 `mini_vlm.py`: 视觉编码 (M10.1) + 投影 (M10.2) + LLM, 输出对图像+指令的理解。这正是 VLA backbone 要的:

```
   M10.3 mini-VLM:  图像 + 指令 → 理解特征 → (原来接) 分类答 VQA
   mini-VLA:        图像 + 指令 → 理解特征 → (现在接) 动作头 → 动作
```

> 唯一的改动: **把 VQA 输出层换成动作头**。backbone 不变, 它学到的视觉/语言理解直接服务于控制。这就是 RT-2 的做法 (M11.1-L2): VLM 的世界知识迁移到机器人。

---

## 2. 装配 mini-VLA 的三步

```
   ① 取 backbone:  M10 的 mini-VLM (或玩具里的状态编码器)
   ② 接动作头:     离散 token 头 / 连续头 / 扩散头 (M11.3)
   ③ 训练:         模仿专家 (CE for 离散 / MSE for 连续), 接 M11.4 模仿学习
```

你的 `mini_vla.py` 实现了②③ (backbone + 可换动作头 + 训练)。N1 会:
- import M10 的 `mini_vlm` 确认它是 backbone (感知核插槽)。
- 用 `mini_vla` 在 toy 控制任务上组装「backbone + 动作头」并训练 rollout。

> 注意玩具的简化: toy 的「观测」是状态向量 (非图像), 所以 backbone 用小 MLP。但**结构和角色与真实 VLA 完全一致**: backbone 编码观测 → 动作头出动作。真实里 backbone = M10 VLM, 观测 = 图像。机制守恒, 规模不同。

---

## 3. 冻结策略: 训哪部分 (接 M10.3)

装好后, 训练时冻哪部分? 和 M10.3 的 VLM 训练食谱一脉相承:
- **冻 backbone, 只训动作头**: 省算力, 防遗忘 VLM 知识。适合 backbone 已很强时。
- **联合微调**: backbone + 动作头一起训。更强但更贵, 可能遗忘。
- **两阶段**: 先冻 backbone 训动作头, 再小学习率联合微调 (M10.3 的食谱)。

> VLA 训练 = M10.3 的 VLM 训练食谱 + 一个动作头。你学的冻结/解冻/联合微调直接搬过来。**没有新训练范式, 只是新输出头。**

---

## 4. 从玩具到真实 (差的只是规模)

你的 mini-VLA 和 OpenVLA 差什么?
| | 你的 mini-VLA | OpenVLA |
|---|---|---|
| backbone | 小 MLP / mini-VLM | 7B VLM |
| 观测 | 状态向量 / toy 图 | 真实相机图 |
| 动作头 | 离散/连续 (9类/2D) | 离散 token (高维) |
| 数据 | 合成专家 demo | Open X-Embodiment |
| 机制 | **完全一样** | **完全一样** |

> 又一次验证课程核心信念 (M13.7-L1): **前沿系统 = 你已学部件 + 规模**。mini-VLA → OpenVLA 只差规模和真实数据。你懂结构的每一块。

---

## 5. 本讲小结 (11.2 收口) + 通往 11.3

- backbone 你 M10 已造好 (mini-VLM); VLA = 把 VQA 输出层换成动作头。
- 装配三步: 取 backbone → 接动作头 → 模仿训练 (接 M11.4)。
- 冻结策略 = M10.3 的 VLM 训练食谱 (冻/联合/两阶段), 直接搬。
- mini-VLA → OpenVLA 只差规模 + 真实数据, 机制完全一样。

> **11.2 收口**: VLA = backbone (M10 VLM) + 动作头; backbone 趋同、动作头分化; 离散vs连续各有代价 → 扩散补齐。
> **下一专题 M11.3「action-heads-diffusion-policy」**: 动作头的巅峰 —— 用扩散 (你的 M13!) 做动作头, 解决多峰+连续+平滑。这是 π 的核心, 把 M13 直接接到机器人。

**动手**: 完成 N1 (装配 mini-VLA) + N2 (离散vs连续) 后, 画出「mini-VLA = M10 backbone + 动作头」的结构图, 标出每块来自哪个专题。
