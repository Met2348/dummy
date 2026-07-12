# 01 · 项目深挖 — interp-graduation Capstone（Module 12.7）

> 素材来源：`learning/interp-graduation/src/interp_capstone.py`、`lectures/L1-full-interp-workflow.md`、`lectures/L2-frontiers-and-your-path.md`、`README.md`、`notebooks/N1-full-interp.ipynb`（含真实执行输出）、以及被跨专题 import 的 `tiny_transformer.py` / `probing.py` / `patching.py` / `sae.py`。
> 用法见 `00-how-to-defend.md` §3：合上文档口头讲一遍主线，再翻开追问树逐层自问自答。这篇是脚手架，不是台词稿。

---

## 1. 背景与目标

这是「机制可解释性」(mech interp) 系列 Module 12 的毕业设计（12.1–12.6 走完后的第 7 个专题），要验证的不是某个新方法，而是**能不能把一整套 interp 工具串成一次完整、自洽的逆向工程**，对一个模型给出「连贯、因果、有证据」的机制故事，并从中提炼出可执行的研究方向。

具体规模（如实写，不夸大）：
- **模型**：不是 GPT-2/TinyLlama，而是一个从零手写的 toy transformer（`interp-foundations/src/tiny_transformer.py`）——词表 `V=12`、序列长度 `SEQ=8`、`d_model=32`、`2` 层、`4` 头注意力，纯 CPU、torch 手写。
- **任务**：`increment-mod-V`——序列每一位 = 前一位 + 1 (mod 12)，任务结构完全已知，专门设计成有 ground truth、方便解剖，不是自然语言任务。
- **训练规模**：2000 条训练序列（`make_data(2000, seed=0)`），800 epoch，Adam，lr=2e-3（`tiny_transformer.train()` 默认值），测试集 400 条（`seed=9`）。

capstone 脚本 `interp_capstone.py` 做两件事：
1. `assembly_check()`：跨专题 import M12 全部 6 个子专题的 `src`（12.1 tiny_transformer / 12.2 probing / 12.3 patching / 12.4 sae / 12.5 circuits / 12.6 cot_probe），证明它们能装配成一个工具箱。
2. `run_full_interp()`：对上面的 toy transformer **实际跑**探针 (12.2) → activation patching (12.3) → SAE (12.4) 三步，产出一条证据链；外加一个基于 M9「找 gap」框架整理的 5 条 interp 研究 gap 雷达（2 条 ★ 标记为 interp×reasoning，对应用户 NLP+EE+reasoning-r1 背景的 PhD 方向候选）。

> 需要先说清楚的边界：`run_full_interp()` 实际执行的只是 12.1–12.4（tiny_transformer/probing/patching/sae）四个子专题；12.5 circuits（induction head）和 12.6 cot_probe（CoT 忠实性）只在 `assembly_check()` 里被 import 检查，**没有**被这个 capstone 脚本实际调用执行。「完整逆向工程」这个说法要 scope 到探针→patching→SAE 这三步。

---

## 2. 个人贡献

自学项目，个人贡献 = 全部实现，具体决策点：
- 设计 `STACK` 列表把 M12 六个子专题的 `(topic, module, 描述)` 统一注册，写 `add_paths()` 动态把六个子专题的 `src/` 目录加进 `sys.path`，实现跨专题 import（而不是把六份代码复制粘贴到一个文件里）。
- 在 `run_full_interp()` 里选定具体的探针层 (`resid_post_1`，最后一个 block 的残差流输出)、patching 的干净/污染对照构造方式、SAE 的过完备倍数和稀疏系数，把三个独立工具接到同一个 toy 模型、同一个「当前值」概念上，让三条证据能互相印证。
- 设计并写下 5 条研究 gap（`GAPS` 列表，每条含 `area/gap/why_hard/min_exp/connects` 五个字段）和渲染函数 `make_idea_card()` / `gap_radar()`，并且明确标出哪两条最匹配自己的背景（★）。
- 写 `environment/verify_env.py` 里对 capstone 输出做量化校验的阈值逻辑（`story_ok` 判断），不是只打印数字看着舒服，而是有具体的通过/失败判据。

---

## 3. 关键技术决策与理由

| 决策 | 理由（代码/文档可查） |
|---|---|
| 用手写 toy transformer 而非直接上 gpt2 | `tiny_transformer.py` docstring 原话：「真 gpt2 太大、电路未知，适合做『真实』演示，但教方法需要干净的玩具」——需要 ground truth 才能验证探针/patching 是否"读对了"。 |
| 任务选 increment-mod-V | 任务结构完全已知（下一个 token = 当前+1），给探针一个明确的监督标签（"当前值"），给 patching 一个可预判的因果位置（答案只依赖最后一个 token），避免自然语言任务里"这个概念到底是什么"的模糊性。 |
| 模型刻意做小（d_model=32, 2层, seq=8） | 多处 docstring 强调「纯 torch tiny CPU 确定性」——刻意牺牲规模换取可复现性和无 GPU 依赖，这是教学向 capstone 的一致设计原则，不是能力不足。 |
| 探针用 `resid_post_1`（最后一层残差流） | `probing.tiny_layer_activations()` 默认取最后一个 block 输出位置，这是"预测即将形成"的最后一站，最适合验证"当前值"是否被线性编码。 |
| patching 用 denoising 方向（clean→corrupt） | `patch_recovery()`：干净、被污染一对序列只在最后一个 token 上不同，把 clean 的某 (layer, pos) 激活贴进 corrupt 运行，测多大程度能恢复 clean 的答案；用 `(patched - corrupt基线)/(clean基线 - corrupt基线)` 归一化成 0~1 的「恢复率」，让不同 (layer, pos) 的因果贡献可以直接比较数值大小。 |
| SAE 过完备倍数选 `V*3=36`（vs d_in=32），l1=1e-2 | `interp_capstone.py` 里显式调用 `S.train_sae(sae, acts, epochs=500, l1=1e-2)`——这个 l1 系数比 `sae.py` 自己独立 demo 用的默认值（`train_sae` 签名默认 l1=2e-3，standalone `__main__` 用 3e-3）更大，意图是在只有 500 epoch 的预算内把纯度差距拉得更明显，方便教学演示"SAE 解叠加有效"。 |
| 单义性 (monosemanticity) 指标用质量加权最大类别占比 | `sae.py monosemanticity()`：对每个"活跃"特征（`>1e-3` 激活且至少 3 个样本触发），算它的激活质量里哪个 label 占比最大，取平均。这是本 repo 自定义的简化 proxy，不是照搬 Anthropic 论文的标准评估指标（诚实局限见第 5 节）。 |

---

## 4. 踩过的坑与解决

文档/代码里**明确记载**的坑（不是推测）：
- **相关 ≠ 因果的陷阱**：`probing.py` docstring 明确写「注意：探针能读出 ≠ 模型在用它（相关非因果）。这是探针的根本陷阱」——这是 M12.2→M12.3 这个顺序本身要解决的问题：先用探针拿到一个"相关性假设"，再用 patching 做因果验证，不能只做探针就下结论。
- **SAE 死特征稀释纯度指标**：`monosemanticity()` 显式跳过几乎不激活的特征（`(col > 1e-3).sum() < 3: continue  # 几乎不激活的特征跳过`），如果不跳过，大量"死"特征会把平均纯度算得没有意义（分母上全是噪声）。
- **跨专题装配的容错**：`assembly_check()` 对六个子专题逐个 try/except，单个 import 失败不会让整体检查连锁失败，而是分别报告哪个子专题的工具装不上——这是为了让装配检查本身能诊断"具体哪一环"出问题。
- **重依赖的降级**：`verify_env.py` 把 `torch`/`transformers` 标为 `OPTIONAL`，只有检测到 torch 才会真的跑 `run_full_interp()`，否则只做轻量的 import/gap 卡检查——这是为了让这个检查脚本在没装 torch 的环境里也能跑一部分。

以下是**推测**（非文档明确记载，基于代码里的设计一致性推断）：
- 每个函数都显式传 `seed` 参数（模型初始化 seed=0，训练数据 seed=0，测试数据 seed=9，探针激活数据 seed=5，patching 对照 seed=3……）——（推测，非文档明确记载）这大概率是为了让 toy 实验在教学场景下完全可复现，避免"这次跑出来因果位置对了，下次换个种子就不对"这种不稳定性混淆教学效果，但代码里没有写"为什么选这几个具体 seed 数字"。
- `format_reward`/`monosemanticity` 这类函数对边界条件（空激活、空字符串）都做了保护性判断——（推测）这类防御性写法通常是调试过程中撞到过对应的边界 bug 后加上的，但 interp-graduation 本身没有留下具体的调试记录/commit log 可查证这一点。

---

## 5. 结果与诚实局限

### 实际跑出的数字（来自 `notebooks/N1-full-interp.ipynb` 的真实执行输出，不是编的）

```
行为: 玩具 transformer 预测「下一个 = 当前 + 1」(任务准确率 1.00)
② 探针 (12.2):    residual 线性编码「当前值」准确率 1.00  [相关线索]
③ patching (12.3): 答案信息因果定位在位置 7 (=最后位置 7)  [因果证据]
④ SAE (12.4):     「当前值」编码为单义特征, 纯度 0.56 >> 原始 0.15  [表示]
```

`environment/verify_env.py` 里对应的量化通过判据（真实代码，非编造）：
```python
story_ok = r["probe_acc"] > 0.9 and r["causal_pos"] == r["last_pos"] and r["sae_purity"] > r["raw_purity"]
```
即：探针准确率 > 0.9（实测 1.00）、因果位置精确等于序列最后一位（实测 7 == 7）、SAE 纯度严格大于原始神经元纯度（实测 0.56 > 0.15）。三项都通过。

### 诚实局限（不走过场）

1. **只做了充分性验证，没做必要性验证**：`patching.py` 里其实定义了 `ablate_effect()`（把某 (layer, pos) 激活置零，测试对 clean 答案 logit 的损害，即"必要性"方向），但 `run_full_interp()` **只调用了 `patch_recovery()`**（充分性/denoising 方向），从未调用 `ablate_effect()`。而 `L1-full-interp-workflow.md` 自己在"严谨性贯穿全程"一节明确把"充要（patching 充分 + ablation 必要都做）"列为标准——capstone 代码目前只做到了一半，这是一个可以直接在代码里核查的、真实的 gap。
2. **玩具 ≠ 真实**：这个结论建立在一个词表只有 12、序列长度只有 8、任务结构完全已知的极简玩具上。`L1` 讲义自己也强调"玩具教方法（可控），真模型验证（可信但乱）"是必须报告的边界——真实语言模型的电路远比这乱得多、冗余路径更多，这条"探针→patching→SAE 三重印证"的结论不能直接推广到任意规模模型。
3. **单次运行，没做多 seed 鲁棒性检查**：`run_full_interp(seed=0)` 只跑了一次固定的种子组合（各子步骤内部又各自用了不同的固定 sub-seed），没有跑多个随机种子看这三个指标的方差，不能排除这次「故事」在别的种子下不成立的可能。
4. **SAE 纯度指标是自定义 proxy，不是标准评估基准**：0.56 这个数字来自本 repo 自己写的质量加权最大类别占比，没有和已发表的 SAE 评估方法（如因果验证类指标）做过对比，也没有做"用同样维度的随机投影做对照"这类消融，无法排除"纯度提升部分来自特征数变多（36 vs 32）本身"的可能性。
5. **研究 gap 雷达是 proposal，不是已完成的研究**：`GAPS` 列表里的 5 条 gap（含 2 条 ★ interp×reasoning）目前只是结构化整理出的研究方向 + 对应的最小实验设计，`gap_radar()`/`make_idea_card()` 只是把它们打印出来，capstone 代码里**没有任何一行**真正执行了这些 `min_exp`（比如"对某一步做 activation patching 看改它答案是否变"）。这是明确的下一步待办，不是已产出的研究结果。

---

## 6. 追问树

### 链 1 · "这个逆向工程结论的把握有多大，换一个模型规模还成立吗？"
- **L1**：你这个「模型在最后位置用单义特征编码当前值，然后 +1」的结论，是怎么从三个独立证据（探针/patching/SAE）交叉验证出来的，而不是巧合？
  → probe_acc=1.00 说明相关性（residual 线性可读出当前值）；patching causal_pos=7=last_pos 说明因果（在这个位置 patch 能让 corrupt 序列的答案 logit 完全恢复，且 `verify_env.py` 断言 `causal_pos == last_pos`）；SAE purity 0.56 >> raw 0.15 说明该信息被更单义地表示。三个独立方法在同一位置/概念上收敛，是三角验证。
- **L2**：换一个模型规模（比如 d_model 从 32 变 256，或换真实 GPT-2）这个结论还成立吗？
  → 诚实回答：没测过。这是纯 toy 模型（V=12, SEQ=8, d_model=32, 2 层），任务本身结构已知、几乎线性可分，是刻意设计成"干净"以教方法，不代表真实语言模型电路必然如此干净。`L1` 讲义自己强调"玩具≠真实"是必须报告的边界。
- **L3**：那你凭什么说这套方法对 interp×reasoning 这种大得多、复杂得多的问题有用？
  → 方法论（探针读相关 + patching 验证因果 + SAE 解叠加）是 mech interp 领域公认的标准工具链，不依赖模型规模；capstone 证明的是"我能正确操作这套工具链、理解每一步在证明什么、不会把相关当因果"，而不是证明"这套工具在任意规模下都会给出干净结论"。规模化后电路更乱、冗余路径更多，这正是 `GAPS` 里"自动化/可扩展 circuit 发现"这条 gap 本身在问的问题。
- **L4 / pitfall**：如果考官追问"能不能当场估一下这套方法在 GPT-2 上概念保真度大概会怎么变差"？
  → 定性可以说：真实 transformer 因为叠加 (superposition) 会有更多特征纠缠在同一神经元里，probe_acc 可能仍然很高（线性探针擅长挖弱信号），但 patching 的因果定位会更分散，SAE purity 大概率明显低于 toy 的 0.56。**pitfall**：编一个"大概会掉到 0.3"这类具体数字就是编造——没做过的定量估计，不该假装有依据地给出数字。

### 链 2 · "你怎么排除这是过拟合到这一个 toy 任务的可能性？"
- **L1**：probe_acc=1.00 会不会只是探针自己在训练集上过拟合？
  → 不会，`linear_probe()` 做了 train/test 切分（`test_frac=0.3`），报告的是测试集准确率（te_acc），不是训练集准确率。
- **L2**：那会不会是整个"故事"过拟合到这一个特定 seed 组合的巧合？
  → 有风险——`run_full_interp` 只跑了一次固定的 seed 组合（模型 seed=0，训练数据 seed=0，测试数据 seed=9，探针激活数据 seed=5，patching 对照 seed=3），没有做多 seed 重复实验看这三个指标是否稳定重现。
- **L3**：那你怎么知道这不是脆弱的、换个种子就崩的结果？
  → 诚实局限：没做过。最小的补充实验是把 `run_full_interp(seed=...)` 换 5-10 个种子跑，看 `probe_acc`/`causal_pos`/`sae_purity` 的方差——如果方差大说明"故事"脆弱。（推测/待补，非文档记载的已完成工作）
- **L4 / pitfall**：causal_pos 为什么恰好等于 last_pos，这是不是设计好的？
  → 不是巧合，是任务结构决定的：`make_clean_corrupt()` 只改了序列最后一个 token，答案天然只依赖最后一个位置，`causal_pos=7`(=SEQ-1) 几乎是任务设计的直接推论。**pitfall**：不要把这包装成"意外发现的深刻电路结构"，诚实说法是"这验证了 patching 工具用对了"，而不是对模型的新发现。

### 链 3 · "你说做了 causal patching 验证，这个因果验证是完整的吗（充分性+必要性都做了）？"
- **L1**：`patch_recovery` 具体验证的是充分性还是必要性？
  → 充分性（sufficiency）方向——把 clean 的某 (layer, pos) 激活贴进 corrupt run，看是否足以恢复 clean 的答案（denoising 方向）。
- **L2**：那必要性（比如把这个位置置零/ablate，看 clean 答案是否被破坏）做了吗？
  → `patching.py` 里定义了 `ablate_effect()`，但 `run_full_interp()` 只调用了 `patch_recovery`，**没有**调用 `ablate_effect`——capstone 脚本本身没做必要性验证。
- **L3**：那 L1 讲义里"充要（patching+ablation 都做）"这条严谨性标准，capstone 代码没做到？
  → 对，这是已知 gap——lecture 把它列为标准，但实际实现只做了一半；`ablate_effect` 已经实现在 `patching.py` 里，随时可以加一行调用补上，只是目前没跑这一步。
- **L4 / pitfall**："所以你说的『连贯、因果、有证据的机制故事』其实证据不完整？"
  → 需要精确措辞：这是"因果充分性"证据，不是完整的双向因果证据。**pitfall**：不能因为怕露怯就说"两个都做了"，一旦被要求复述 `ablate_effect` 的返回值就会当场露馅，诚实承认这个可验证的 gap 比不懂装懂安全。

### 链 4 · "SAE 特征纯度 0.56 是好是坏？这个数字怎么来的，靠谱吗？"
- **L1**：purity=0.56 具体怎么算出来的？
  → `monosemanticity()` 对每个"活跃"特征（`>1e-3` 激活且至少 3 个样本触发）计算它的激活质量在各 label（0-11 的"当前值"）间的分布，取占比最大的 label 的比例作为该特征纯度，再对所有活跃特征取平均。0.56 意味着平均一个特征 56% 的激活质量集中在它的"主 label"上。
- **L2**：这个指标是标准的 SAE 评估指标吗？
  → 不是，是本 repo 自己写的简化 proxy，没有和已发表 SAE 评估基准做对比或校验——SAE 评估标准本身就是 `GAPS` 里列出的开放问题之一（gap #4：SAE 特征的因果验证 + 评估标准）。
- **L3**：0.56 对比 raw neuron 的 0.15，这个提升说明了什么/没说明什么？
  → 说明 SAE 字典学习确实把"当前值"从多义神经元里解离出更纯的表示（相对提升约 3.7 倍），是"解叠加有效"的证据；但离 1.0（完全单义）还很远，toy 任务概念数很少（只有 12 个可能值）、只训了 500 epoch，规模不足以断言这是 SAE 能力的上限。
- **L4 / pitfall**：你怎么排除这个纯度提升不是因为 SAE 特征数（36）本来就比 raw neuron 维度（32）多，单纯"分箱更细"导致的伪提升？
  → 诚实回答：没有排除这个可能性（推测，非文档明确记载）。补充实验可以做"随机投影到 36 维"的对照组。**pitfall**：不要假装这个消融已经做过。

### 链 5 · "你说 interp×reasoning 是你的 PhD 方向，这个 capstone 具体产出了什么支持这个判断，还是只是一个愿望？"
- **L1**：gap radar 这 5 个 gap 和 2 个 ★ 是怎么来的？
  → 用 M9「科研技能」模块教的找 gap 框架，对 M12（12.1-12.6）走完全套工具后反思出来的 5 个方向，写成结构化 idea 卡（`area/gap/why_hard/min_exp/connects`）。★ 标记的 2 个（CoT 忠实性的机制级验证、计算 vs 陈述一致性）因为直接连接 patching(12.3) + CoT 忠实性(12.6) + reasoning-r1 背景，对个人地基最匹配。
- **L2**：这些 gap 卡对应的 `min_exp` 实际跑过吗，还是纯粹的 proposal？
  → 纯 proposal，没有跑过。`gap_radar()`/`make_idea_card()` 只是把 5 个 gap 结构化打印出来，代码里没有任何一行实际执行这些最小实验。
- **L3**：如果现在让你当场把 ★1（CoT 忠实性机制级验证）的最小实验做出来，你需要什么？
  → 需要一个会输出 CoT 的小模型 + `patching.py` 同款的通用 activation patching 工具（已有，任意激活矩阵通用），把某个 CoT 步骤对应的中间表示替换掉，看最终答案是否随之改变——`L2` 讲义里写了这个可执行步骤，但 capstone 本身没有执行。
- **L4 / pitfall**："所以现在能不能说你已经找到了 CoT 不忠实的证据？"
  → 不能，这样说是过度推销。目前状态是"识别了一个有价值、可执行的研究 gap，并具备执行它所需的全部工具"，但还没有产出任何关于"CoT 是否忠实"的实证结果。**pitfall**：千万不要把"我知道怎么做这个实验"包装成"我已经做出了这个结论"。
