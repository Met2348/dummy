# 基线可复现性调研:哪些能自己跑,哪些只能引用论文数字

> 老师要求"进一步调研,摸清楚基线的情况"——这是在呼应 Weikai 最早的指导"开搞第一步永远是
> 先确立 evaluation protocol,先把 1-2 个重要 baseline 都 eval 一遍"。此前 `05`/`06` 已经
> 逐字通读了 11 篇最近邻竞品论文的完整内容(架构/公式/结果/消融/局限性),但从未调研过这些
> 方法**能不能被我们自己实际跑起来**——这份文档补的就是这一块:代码是否公开、权重是否可下载、
> 依赖的 benchmark/仿真器是否公开可用,以及最关键的——我们未来真正做实验要用的骨干 world
> model(DreamerV3/TD-MPC2/MuZero 系)现在是什么状态、插入自定义想象控制器的门槛有多高。
>
> **方法论**:三路独立调研(11 篇竞品方法分两组 + 骨干 world model 单独深挖),全部用
> WebSearch/WebFetch/GitHub API 做一手核查(star 数、最近 commit 时间、issue 响应情况均为
> 实测,不是转述),部分骨干 world model 的结论直接抓取了源码文件验证。凡未查到确凿证据的
> 地方,一律标注"未找到/未能确认",不做推测性填空。

---

## 0. 一页说清楚

**三条核心结论**:

1. **11 篇竞品方法里,代码状态参差不齐**——只有 2 篇(Video-T1、Finding the Time to Think)
   代码+权重+benchmark 全部齐全,可以直接拿来当"确实能跑"的 baseline;World-in-World 基础
   设施基本齐全但要签学术协议拿场景数据;AVIC 权重/代码齐全但世界模型部分需要 HuggingFace
   审批+大显存;ITP、Astra 都是"半发布"状态(training-free/世界模型那部分能跑,核心 RL 训练
   出的控制器权重还没放出来);ELASTIC、ROI-Reasoning 代码未公开但底层完全开源,方法论上可以
   重新实现;FFDC、RARRL 复现难度最高(前者关键超参数缺失,后者绑定闭源 GPT-4o-mini API)。
2. **World-in-World 已确认是 ICLR 2026 Oral**——这是本次调研的一个重要副产品,此前
   `03-novelty-competitive-landscape.md`/`05`/`06` 都明确写"无法在 PDF 里确认接收状态",
   现在有独立的 OpenReview 页面和 ICLR 官方日程可以交叉确认,已同步更新到相关文档(见本文档
   末尾)。
3. **骨干 world model 里,TD-MPC2 目前摩擦最小**——官方直接发布了 300+ 个预训练 checkpoint,
   覆盖 DMControl/Meta-World/ManiSkill2/MyoSuite 四个任务域,理论上可以完全跳过"从零训练"
   这一步,直接在别人训好的 world model 上插入我们自己的想象预算控制器做实验。DreamerV3 是
   有力的并列候选,直接抓取源码发现 `RSSM.imagine(..., length, ...)` 是一个独立方法、想象
   步数已经是显式参数,插入点非常清楚,缺点是没有官方 checkpoint、必须自己训练(不过多数
   benchmark 量级可控,<2 GPU-day)。MuZero 系(选的话选 LightZero)排第三优先级,因为
   "预算"在 MCTS 里是离散树搜索的模拟次数/深度,机制比连续潜空间 rollout 复杂。

---

## 1. 十一个竞品方法:代码/复现情况总表

```{=latex}
\begin{landscape}
```

| 方法 | 代码是否公开 | 权重/checkpoint | 依赖 benchmark 是否公开 | 复现判断 |
|---|---|---|---|---|
| Video-T1 | ✅ 317★,持续维护1年+ | 依赖模型全开源(Pyramid-Flow/VisionReward) | VBench 完全开源,`pip install`即可 | **可以自己跑**,11篇里门槛最低 |
| AVIC | ✅ 18★,README完整 | LoRA adapter已发布(HF) | SAT-Real/R2R均公开 | **可以自己跑**,但世界模型SVC权重需HF审批+≥80GB显存 |
| Finding-the-Time-to-Think | ✅ 14★,含checkpoint | 已随附(Git LFS,~200MB) | 5个环境全部随代码发布,含vendored Jumanji | **可以自己跑**,复现门槛最低(代码+权重+环境全齐) |
| World-in-World | ✅ 181★,持续维护半年+ | 需按文档逐个第三方world model配置 | 基本公开,但Matterport3D/HM3D需签学术协议 | **基础设施齐全**,协议是流程性门槛非技术门槛;已确认ICLR 2026 Oral |
| ITP | ✅ 16★,73次提交 | training-free版(ITP-I)不需要;RL版(ITP-R)"Coming Soon" | ALFWorld/WebShop/ScienceWorld/StableToolBench均公开但安装门槛不同 | **ITP-I+全部benchmark现在能跑**;更强的ITP-R需自己重训 |
| Astra | ⚠️仅README+assets,无源码 | 世界模型Astra-WM已发布;核心控制器Astra-VL未发布 | MMSI-Bench/MindCube均公开 | **半成品**,核心"控制"逻辑代码/权重/评测脚本都没放出 |
| ELASTIC | ❌未找到 | 无(但底层π0.5基座公开:Physical Intelligence openpi) | LIBERO-10/Robomimic/PushT均公开 | **只能引用论文数字**,但底层全公开,方法论上可重新实现 |
| ROI-Reasoning | ❌未找到 | 无(但基座Qwen2.5系列/DeepSeek-V3.2等全部公开权重) | GSM8K/MATH/AIME均标准公开数据集 | **只能引用论文数字**,关键训练超参数(lr/batch/GPU配置)缺失 |
| FFDC | ❌未找到 | 无(但backbone Motus独立开源,~16GB checkpoint) | RoboTwin 2.0公开维护活跃(2621★),但仅Linux+NVIDIA GPU完整支持 | **只能引用论文数字**,verifier关键超参数(层数/参数量/数据规模)未披露 |
| RARRL | ❌未找到 | 不适用(思考模块直接调用闭源GPT-4o-mini API) | ALFRED+AI2-THOR公开,但自建抽象任务场景数据未发布 | **复现难度最高**,绑定闭源API+核心数据未发布,双重障碍 |

```{=latex}
\end{landscape}
```

**逐项细节**(代码链接/star数/checkpoint链接等一手查证结果,均含来源URL,详见调研过程存档;
以下只列每篇最关键的一条补充说明):

- **AVIC**(GitHub: `Yui010206/Adaptive-Visual-Imagination-Control`,18★,最后push
  2026-06-02):SAT-Real 实为 Ray et al. 2024《SAT》的子集(非AVIC自建),HF公开
  `array/SAT`;R2R 需额外接入 MapGPT 框架。
- **FFDC**:backbone "Motus" 是完全独立团队的开源项目(`thu-ml/Motus`,HF权重约16GB),
  可以白拿,但 FFDC 自己加的 verifier 模块层数/隐藏维度/学习率/数据规模论文均未披露,
  复现数字对不上论文报告值(69.10%前向传播减少等)的风险较大。
- **Video-T1**(GitHub: `liuff19/Video-T1`,317★,17 forks,MIT协议,最后push
  2026-03-07):已被 **ICCV 2025** 接收,是11篇里唯一确认非ICLR/UAI体系的方法。
- **Astra**(GitHub: `ZCMax/Thinking-With-Imagination`,30★):仓库自带"Release
  Progress"清单,作者自己标注 Astra-WM checkpoint 已发布✓、Astra-VL checkpoint/训练
  代码/评测脚本均未发布——不是我们判断出来的,是作者自己承认的。
- **ITP**(GitHub: `loyiv/ITP`,16★,73次提交):v1摘要写"代码即将发布",v2补充了
  具体链接,说明这是作者履约后才补上的,不是长期占位。
- **ELASTIC**:通讯作者Andrea Bajcsy的CMU主页、CMU-IntentLab组织(22个公开仓库)均未
  见相关代码,确认非"藏得深"而是"确实没发布"。
- **ROI-Reasoning**:唯一提及"publicly available"的地方是Ethics部分说明"实验都在
  公开benchmark上进行"(指数据集,不是作者自己的代码)。
- **RARRL**:15位作者、12个机构(CMU/东北大学/哈佛/康奈尔/MIT/清华/北大等),规模不小
  但没有任何一方发布项目主页或代码。
- **World-in-World**(GitHub: `World-In-World/world-in-world`,181★,4 forks,MIT
  协议):arXiv摘要页Comments字段直接写"Code is at this https URL",commit历史显示
  2025-10至2026-04持续更新(集成OpenPI、新增LIBERO后端支持),是11篇里维护时间跨度
  最长的仓库。
- **Finding the Time to Think**(GitHub: `Aneeshers/realtime-rl-code`,14★):README
  专门有"Caveats & known limitations"一节讨论跨GPU型号(H100/A100/A40)的复现方差,
  CPU-only未测试过——这种坦诚记录本身是"真实可跑的研究代码"的佐证。

---

## 2. 骨干 World Model:能不能拿来做真实 Pilot

### 2.1 三选一对比表

| 维度 | DreamerV3 | TD-MPC2 | MuZero系(LightZero) |
|---|---|---|---|
| GitHub | `danijar/dreamerv3`,3581★ | `nicklashansen/tdmpc2`,898★ | `opendilab/LightZero`,1625★ |
| 最近有效更新 | 2026-05-25(真实bug修复,已交叉验证) | **2026-07-13(issue当天响应,三者最活跃)** | 2026-07-17(团队内部快速迭代) |
| 官方预训练checkpoint | **无**(三个独立信源交叉确认) | **300+个,覆盖4任务域,直接下载免申请** | 有基础设施,具体覆盖范围未查全 |
| 从零训练算力 | 多数任务<1-2 GPU-day(V100),有据可查 | 多任务3.7~33 GPU-day(RTX3090);单任务数字缺失 | 未查到官方具体数字 |
| "想象/规划"代码隔离度 | 高——`RSSM.imagine(length=...)`独立方法,想象步数已是显式参数 | 高——`_plan()`独立私有方法,采样数/迭代次数/horizon均在其中 | 有,但对象是离散MCTS树搜索的模拟次数/深度,机制更复杂 |
| 生态 | JAX(相对小众) | PyTorch(更普及) | PyTorch |
| 二次开发先例 | 多篇+2个完整PyTorch重写版(`dreamerv3-torch`/`sheeprl`) | 多篇+官方衍生分布式训练分支 | 团队自产UniZero/ReZero/**ScaleZero(ICLR 2026)** |

### 2.2 关键代码证据(直接抓取源码得到,不是转述)

**DreamerV3**(`dreamerv3/rssm.py` 第94行附近):
```
RSSM.imagine(self, carry, policy, length, training, single=False)
```
想象展开长度 `length` 已经是这个方法的显式参数,`agent.py` 里 `imag_loss(...)`(基于想象
轨迹的actor-critic损失)和 `repl_loss(...)`(基于真实回放数据的表征学习损失)是分开的两个
函数——理论上可以通过包装/替换这个方法的调用点、动态决定传入的 `length`,插入自适应想象
预算控制器,不需要大改训练主循环。

**TD-MPC2**(`tdmpc2/tdmpc2.py`):真正做轨迹优化的逻辑封装在独立私有方法
`_plan(self, obs, t0=False, eval_mode=False, task=None)` 里,与价值估计
`_estimate_value` 分开;训练循环被拆到独立的 `tdmpc2/trainer/` 目录,与核心算法类解耦。
采样数、迭代次数、规划horizon这些"预算"旋钮都在 `_plan()` 内部,边界清晰。

**LightZero**:`lzero/` 包结构里 `mcts/` 整体是可拦截/替换的独立模块,`policy/` 目录下
每种算法变体(`muzero.py`/`efficientzero.py`/`gumbel_muzero.py`)共享同一套MCTS后端。
LightZero本身内置的 Gumbel MuZero 正是"用更少模拟次数也能获得策略提升保证"这个方向的
现成先例,说明"改造模拟预算"在这个代码库里已有算法级参考,但改造的是整套离散树搜索机制,
复杂度高于前两者的连续潜空间rollout。

### 2.3 MuZero 系的补充深挖(额外一路更细的调研,专门针对 MCTS 预算机制)

> 这一小节的信息来自对 MuZero 系的一次更深入的专项复核,重点回答"MCTS 里的模拟次数预算,
> 具体接入自适应控制器的难度有多大"这个更细的问题,和 §2.1-2.2 的结论互相印证、互相补充。

- **MuZero 从未有官方代码,这不只是我们的调研结论,是原作者自己说的**:MuZero 共同一作
  Julian Schrittwieser 在自己的博客里明确写,他只知道有第三方复现版本存在,**但自己没有
  逐一验证过这些复现的正确性**([julian.ac 博客](https://www.julian.ac/blog/2020/05/02/opensource-muzero-implementations/))。
- **多了一个候选:`google-deepmind/mctx`**(2645★,DeepMind 官方 JAX 搜索库,最近提交
  2026-07-09)——这是纯粹的"搜索算法库",不含 world model 训练管线(表示/动力学/预测网络
  需要自己写)。它的优势是 `num_simulations` **直接是策略函数的显式关键字参数**
  (`mctx.gumbel_muzero_policy(..., num_simulations=32)`),是四个候选里"预算"这个变量
  最干净、最容易在外层循环里动态计算传入的一个——代价是完全没有训练管线,复杂度从"改造
  现成代码"变成"自己从零搭一半"。
- **一条很有说服力的"未满足需求"证据**:mctx 仓库里有一条至今未解决的 issue
  ([#102](https://github.com/google-deepmind/mctx/issues/102)),用户明确提出想要"提前
  停止搜索"(比如搜索到某一步后,最优动作的表现已经连续多步没有提升就提前退出)——这正是
  我们想做的"自适应停止"这个方向,而且是社区自己提出来、至今没人实现的需求,可以直接用作
  论文动机部分的佐证。
- **LightZero 内部的模拟次数目前是"实例级配置",不是"每次调用的参数"**:实际抓取
  `mcts_ctree.py` 源码确认,`num_simulations` 是从 `self._cfg.num_simulations` 读取的
  (构造时设定一次),循环写法是 `for simulation_index in range(self._cfg.num_simulations)`,
  **不是**每次调用可以单独传入的参数——想做到真正的"每个状态自适应"还需要一层包装(在
  两次调用之间修改配置,或者给这个方法加一个可选参数),工作量不算大但也不是零成本。
  好消息是 LightZero **同时维护了一套纯 Python 实现(`ptree`)和编译版实现(`ctree`)**,
  两者算法逻辑对齐——可以先在慢但好改的 `ptree` 上验证自适应控制器逻辑,跑通后再考虑要
  不要移植到快但要碰 C++ 的 `ctree` 上,这是一条现成的、风险递增可控的开发路径。
- **EfficientZero V2 的 README 自己承认部分 Atari 环境复现不出论文数字**——原文:
  "the performance in some Atari environments does not match the results in the paper. We
  are still working on resolving this discrepancy."这是一个很好的提醒:开源代码存在≠
  能复现论文数字,即使是同一批作者发布的官方代码也可能有这个问题,选定骨干后仍需要自己
  先跑一遍确认。
- **一篇值得补进文献库的新论文**:*Demystifying MuZero Planning*(arXiv:2411.04580)
  用实证方式展示"多加一次模拟能带来多少改进"在不同状态/不同游戏之间差异很大——这是支持
  "模拟预算应该自适应分配"这个大前提的一篇好的动机性引用,此前项目文献库里没有这一篇,
  建议正式写作前补充下载核验。
- **Dreamer/TD-MPC2 vs MuZero 系,在"接入难度"这个维度上有一个本质区别**:Dreamer/
  TD-MPC2 的想象/规划是一个**扁平的循环**,把隐动力学模型往前展开固定步数,"预算"就是
  一个整数(H 或候选轨迹数);MuZero 系的"预算"对应的是**一整棵树的模拟次数**,每次模拟
  都会更新沿途节点的访问计数和 Q 值,拦截/改造的复杂度天然更高。这条判断加强了 §2.1
  "TD-MPC2/DreamerV3 优先"的结论,而不是推翻它——但 MuZero 系"预算"这个概念本身更贴近
  经典文献(MCTS 模拟次数)和我们自己 pilot 里已经用的语言,这是一个纯工程难度 vs 叙事
  贴合度的权衡,值得和 Weikai 讨论时提出来。

### 2.4 初步判断(供讨论,不是最终决定)

**TD-MPC2 目前摩擦最小,是2个月时间线里最值得优先尝试的骨干**——核心原因是它把"从零训练
算力是否够用"这个最大的时间线风险直接消除了:如果我们要用的benchmark落在DMControl/
Meta-World/ManiSkill2/MyoSuite这四个任务域内,可以完全跳过训练,直接在官方checkpoint上
插入想象预算控制器做实验。**DreamerV3是有力的并列备选**,`imagine(length=...)`是三者里
找到的最直接"可插入证据",但必须自己训练、且JAX生态门槛略高。**MuZero系排第三优先级**,
维护活跃、有ICLR 2026同题材后续论文(ScaleZero)撑腰,但"预算"概念对应的是MCTS模拟次数/
深度这一整套机制,改造复杂度更高。

**这个排序只是本轮调研的技术判断,不构成最终选择**——最终选哪个,还要结合具体研究假设
(想在哪类任务上体现"想象预算自适应分配"最有说服力)来定,这一步应该留给和 Weikai 的讨论。

---

## 3. 对项目下一步的具体启示

呼应 Weikai 最早"先把1-2个重要baseline都eval一遍"的指导,基于本次调研,具体建议:

1. **如果想先低成本验证"能不能自己跑通一个baseline"这件事本身**:优先选
   **Video-T1**(代码/依赖模型/benchmark全部无审批门槛开源,11篇里摩擦最小)或
   **Finding the Time to Think**(代码+checkpoint+环境全部随仓库发布,复现门槛同样很低,
   而且它是idea 7这条线索目前最近邻的论文之一,顺带验证有双重价值)。
2. **如果要把三轮pilot从合成格子世界迁移到真实world model**(05/06反复标注的悬而未决
   项):建议优先尝试 **TD-MPC2**,理由见§2.4——300+现成checkpoint能让我们跳过训练直接
   开始插入控制器逻辑做实验,这对2个月时间线是最大的风险消除。
3. **World-in-World的ICLR 2026 Oral身份现在已确认**,意味着idea 1"多候选比较"这个
   表层机制的既有工作,权威性比此前判断的更高——`02-deep-gap-analysis.md`里"idea 1
   风险从低上调为中"这个判断应该进一步巩固(不是推翻,是加强)。

---

## 4. 已回填更新的既有文档

`03-novelty-competitive-landscape.md`、`05-full-technical-briefing.md`、
`06-world-model-primer.md` 三处此前关于 World-in-World "无法确认ICLR接收状态"的记录,
均已更新为"已确认ICLR 2026 Oral"(附OpenReview页面`openreview.net/forum?id=yDmb7xAfeb`
及ICLR官方日程交叉核实来源)。

---

## 参考文献索引

本文档核查的11篇竞品方法与3个骨干world model的arXiv编号见`papers/INDEX.md`(AVIC
2602.08236、FFDC 2605.06222、Video-T1 2503.18942、Astra 2606.06476、ITP 2601.08955、
ELASTIC 2606.31132、RARRL 2603.16673、Finding-the-Time-to-Think 2606.26463、
ROI-Reasoning 2601.03822、World-in-World 2510.18135、DreamerV3 2301.04104、
TD-MPC2 2310.16828、MuZero 1911.08265)。本文档新增的代码仓库/checkpoint/benchmark
链接均为一手GitHub API/WebFetch核查所得,未逐条写入INDEX.md(那是论文索引,不是代码
索引),完整链接见本文档正文。
