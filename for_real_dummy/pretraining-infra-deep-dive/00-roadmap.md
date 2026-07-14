# 预训练规模化基建深挖 —— 路线图与进度表

> 目标:约 60-70 个知识点,覆盖"从数据到千卡集群"的预训练规模化基建全链路——数据处理(data-curation)→ 训练规模化(scaling-infra)→ 预训练配方(pretraining-recipe)→ Module 3 毕业(small-model-graduation)→ CUDA 执行模型(cuda-essentials)→ 集群网络(cluster-networking)→ 存储与数据管线(storage-dataops)→ 训练编排(training-orchestration)→ Module 8 毕业(infra-graduation)。
> 定位:仓库"专题精读系列"第 6 条,直接对应 `learning/` 下 9 个专题模块——Module 3《造大模型》剩余 4 站 + Module 8《系统与Infra》剩余 5 站。这 9 个模块本身已有详尽 README(专题概览/横向对比/cheatsheet/自测题/坑注记一应俱全),本系列直接复用其中已核实的实测数字,但换成面试导向的追问链讲法。

---

## 和 `learning/` 9 个模块的关系(差异化声明,必须先读)

**范围边界**(易错点,必须先讲清楚):
- Module 3《造大模型》共 8 站,本系列只覆盖剩余 4 站:①`data-curation`、⑥`scaling-infra`、⑦`pretraining-recipe`、⑧`small-model-graduation`(M3 毕业)。②`transformer-deep`、⑤`long-context` 已被其他系列(`long-context-deep-dive` 等)覆盖,不在本系列范围,涉及处直接交叉引用。
- Module 8《系统与 Infra》共 7 站,本系列只覆盖剩余 5 站:②`cuda-essentials`、④`cluster-networking`、⑤`storage-dataops`、⑥`training-orchestration`、⑦`infra-graduation`(M8 毕业)。①`gpu-architecture`、③`kernel-engineering` 已被 [kernel-gpu-deep-dive](../kernel-gpu-deep-dive/00-roadmap.md) 覆盖,不在本系列范围,涉及处直接交叉引用(**注**:kernel-gpu-deep-dive 只对应这两个模块,不含 cuda-essentials——旧计划一度误记"前3站已覆盖",本系列写作前已核实订正)。
- `scaling-infra` 的 L08-L11(vLLM/PagedAttention、SGLang、投机解码、量化推理)与已完成的 [inference-serving-deep-dive](../inference-serving-deep-dive/00-roadmap.md) 01-04 号文件高度重叠(后者是深度复刻级别,前者只是"打印 setup 模板"级别的浅层 demo)。本系列 02 号文件对这 4 个 lecture **不重复展开七步模板**,合并成 1 个"训练+推理综合基建全局观"知识点并交叉引用。

**和源 `learning/` 模块 README 的定位差异**:源 README 是"研究者对研究者"的密度速查表,本系列每个知识点从"最笨的问题"讲起,额外补齐**底层机制/为什么这样设计**和**面试怎么问+追问链**两块,并且每个"可运行例子"都独立在 `.venv` 里重新跑过、不直接抄 README 里的数字。

---

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生,Python 3.13)。**本系列绝大部分内容纯 CPU、零第三方依赖**——9 个源模块的 `src/` 下脚本只依赖 `dataclasses`/`math`/`hashlib`/`enum`/`__future__`(9 个模块 README 均已逐一核实并列出 grep 结果),秒级跑完,不需要 GPU、不需要装包。

**两处例外,涉及真实 GPU 训练**(本机 RTX 3080 Ti Laptop GPU,16384MiB 显存空闲,已用 `nvidia-smi` 确认):
1. `pretraining-recipe/src/capstone_train.py --train --max_step 3 --micro_batch 2 --grad_accum 2 --seq_len 128` —— 真实 cuda bf16 训练 3 step(~16s,不落 ckpt),验证训练循环真实可跑通(而非只验证 dry-run 建模)。
2. `small-model-graduation/src/train_variant.py --variant A --max_step 2 --seq_len 64 --micro_batch 2 --grad_accum 2 --train` —— 同类真训练(2 step,~12s,不落 ckpt)。

这两处均在文件 03/04 收尾作为"真实验证"bonus,风险远低于上一系列的 WSL2+vLLM 部署(纯本地 `.venv` 直接跑,无网络/无跨系统依赖)。**除此之外全系列不涉及 WSL2**。

`infra-graduation` 模块有特殊的包导入结构(`src/sim/`、`src/eval/` 两个子包,5 个脚本需要 `PYTHONPATH=<module>/src` 或 `python -m sim.xxx` 才能跑,不能直接裸跑单文件)——09 号文件"可运行例子"代码块统一用 `sys.path.insert(0, ".../infra-graduation/src")` 处理,撰写时需注意。

---

## 知识点结构模板(七步,与 inference-serving-deep-dive/peft-deep-dive 完全一致)

标题 `# NN · 中文标题(English Title)`。每个知识点 `## N. 中文标题(`file.py`)—— 一句话副标题`。七步是加粗行内标签、顺序固定:**是什么** → **一句话** → **底层机制/为什么这样设计** → **AI 研究场景** → **可运行例子**(带 assert,代码块后另起段落写"**实测(`.venv` 真跑):**")→ **面试怎么问 + 追问链**(bullet list,`Q:`+`追问1/2/3`)→ **常见坑**。两个"毕业"文件(04/09)例外,采用叙事体 capstone 格式(题目→分步构建→追问→复盘),不套用七步模板。

---

## 进度表

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|-----------|------|
| 01 | 数据处理全流水 | [01-data-curation.md](01-data-curation.md) | 12 | ✅ 已完成(已验证,25/25代码块通过;独立发现MinHash对模板化文本几乎整簇塌缩、SemDeDup对纯语义改写不敏感但对局部编辑敏感、BPE在高重复语料下极端压缩到1 token等) |
| 02 | 训练规模化 | [02-scaling-infra.md](02-scaling-infra.md) | 10 | ✅ 已完成(已验证,19/19代码块通过,含L08-L11去重交叉引用处理;独立发现Chinchilla"最优比例"实际随算力预算漂移、不是固定20) |
| 03 | 预训练配方 | [03-pretraining-recipe.md](03-pretraining-recipe.md) | 10-11 | ⬜ 待撰写(含真实GPU训练bonus) |
| 04 | Module 3 毕业(五部曲) | [04-small-model-graduation.md](04-small-model-graduation.md) | 叙事体capstone | ⬜ 待撰写(含真实GPU训练bonus) |
| 05 | CUDA 执行模型 | [05-cuda-essentials.md](05-cuda-essentials.md) | 8-9 | ⬜ 待撰写 |
| 06 | 集群网络 | [06-cluster-networking.md](06-cluster-networking.md) | 7 | ⬜ 待撰写 |
| 07 | 存储与数据管线 | [07-storage-dataops.md](07-storage-dataops.md) | 7 | ⬜ 待撰写 |
| 08 | 训练编排 | [08-training-orchestration.md](08-training-orchestration.md) | 7-8 | ⬜ 待撰写 |
| 09 | Module 8 毕业(端到端系统设计) | [09-infra-graduation.md](09-infra-graduation.md) | 叙事体capstone | ⬜ 待撰写(链kernel-gpu-deep-dive) |

**预计合计:约 60-70 个知识点,8 篇正文 + 2 篇叙事体 capstone。**

---

## 明细(源码路径,撰写时需重新独立跑一遍确认数字)

### 01 数据处理全流水(源:`learning/data-curation/src/*.py`,12 lecture)
1. 数据时代鸟瞰(C4→Pile→RedPajama→FineWeb→DCLM,概念点)
2. CommonCrawl 抽取(`cc_extract.py`)
3. MinHash+LSH 去重(`minhash_dedup.py`)
4. SimHash/SemDeDup 去重对照(`simhash_dedup.py`+`semdedup_demo.py`)
5. 质量过滤(`quality_filter.py`,C4+Gopher启发式+FineWeb-Edu classifier)
6. 毒性+PII(`toxicity_pii_filter.py`)
7. 手写 BPE vs tiktoken(`bpe_trainer.py`+`bpe_tiktoken.py`)
8. SentencePiece Unigram(`spm_trainer.py`,含"vocab_size两头夹"坑)
9. 多语言压缩率对照(`vocab_compare.py`)
10. 数据配比 ablation(`data_mix_ablation.py`,Doremi概念)
11. Magpie 指令数据合成(`magpie_synthesis.py`)
12. Capstone:1B token 自制语料端到端(`capstone_mini_corpus.py`)

### 02 训练规模化(源:`learning/scaling-infra/src/*.py`,14 lecture)
1. Scaling Laws:Chinchilla vs Llama-3(`scaling_laws.py`)
2. 并行训练总览 DP/TP/PP/SP/ZeRO(`parallelism_demo.py`)
3. FSDP 显存分片(`fsdp_demo.py`)⭐
4. DeepSpeed ZeRO 配置(`deepspeed_config.py`)
5. Megatron-LM 张量并行(`megatron_tp_demo.py`)
6. Pipeline Parallel 1F1B(`pipeline_parallel_demo.py`)
7. 推理优化技术交叉引用(vLLM/SGLang/投机解码/量化——**已在 inference-serving-deep-dive 01-04 号文件深度覆盖,本点只从"训练+推理综合基建"全局视角串联,不重复展开**)
8. 混合精度与稳定性(`mixed_precision_demo.py`)
9. 训练监控与 MFU(`monitoring_demo.py`)
10. Capstone:训练估算器(`capstone_train_estimator.py`)

### 03 预训练配方(源:`learning/pretraining-recipe/src/*.py`,16 lecture)
1. 预训练总流水线总览(`common.py`,概念点)
2. 数据配比与课程学习(`data_mixture.py`)
3. 初始化与 LR schedule:WSD vs cosine vs μP(`init_schedule.py`)
4. 数据加载与 shard(`dataset_shards.py`)
5. Phi-tiny 270M 架构(`phi_tiny_model.py`)⭐⭐⭐⭐⭐
6. 训练 loop 与稳定性(`training_loop.py`)
7. 评测:val_loss/ppl/tiny-HellaSwag(`eval_benchmarks.py`)
8. 知识蒸馏(`distillation.py`)
9. Phi 风格合成数据(`synth_data_prompt.py`)
10. Llama-3 vs DeepSeek-V3 recipe 对照(概念点,合并 L14+L15)
11. Capstone:从零预训练 Phi-tiny(`capstone_train.py`,含**真实 GPU 训练 bonus**)

### 04 Module 3 毕业:五部曲(源:`learning/small-model-graduation/src/*.py`,叙事体capstone)
五个 checkpoint(A基线→B改数据→C改架构Phi-tiny→D长上下文YaRN→E课程学习综合)的渐进式改进故事,`bench_matrix.py`(6 metric×5 ckpt)+`visualize.py`+`generations_compare.py`。链接本系列文件 03(Phi-tiny 架构来源)和 [long-context-deep-dive](../long-context-deep-dive/00-roadmap.md)(YaRN 来源)。含**真实 GPU 训练 bonus**(`train_variant.py --variant A --train`)。

### 05 CUDA 执行模型(源:`learning/cuda-essentials/src/*.py`,7 lecture)
1. 三层执行模型:Grid/Block/Warp/Thread(`common.py`)
2. Vector-Add:ceil-div+边界检查(`vector_add.py`)
3. Warp 级原语:`__shfl_down_sync` 树规约(`warp_primitives.py`)
4. Shared Memory Bank Conflict(`shared_memory.py`)
5. Coalescing 全局内存合并访问(`coalescing.py`)
6. Reduce 三代 + Tiled GEMM(`reduce_kernel.py`+`gemm_tiled.py`)
7. Capstone:Online Softmax 递推(`capstone_softmax.py`,FlashAttention 前置知识)
8. 番外:官方 Guide 例子复现(`cuda_original_minimal.py`,streams/graphs/occupancy)

### 06 集群网络(源:`learning/cluster-networking/src/*.py`,6 lecture)
1. 三层网络+四类互连协议(`common.py`)
2. All-reduce 算法家族:Ring/Tree/Halving-doubling(`allreduce_algos.py`)⭐
3. Fat-tree/Dragonfly 拓扑(`fabric_topology.py`)
4. NCCL 五个核心 collective(`nccl_collectives.py`)
5. SHARP 交换机内聚合(`sharp_inline.py`,加速比精确收敛到 n-1)
6. Capstone:4 fabric×4 集群规模选型(`capstone_cluster_sim.py`)
7. 番外:NCCL protocol/channel 选择模型(`nccl_original_minimal.py`)

### 07 存储与数据管线(源:`learning/storage-dataops/src/*.py`,6 lecture)
1. 五层存储 BW/IOPS/延迟(`common.py`)
2. Dataloader 流水线加速(`dataloader.py`,6.8×分解为两个效应)⭐
3. Sharding 策略:Hash/Range/Round-robin(`sharding.py`)
4. Checkpoint 三代:Full/Sharded/Async(`checkpoint.py`)
5. WebDataset 顺序读 vs 随机小文件(`webdataset_style.py`)
6. Capstone:70B/512GPU ckpt 经济学(`capstone_ckpt_recovery.py`)
7. 番外:论文 cost model+locality-aware loading(`data_loading_original_minimal.py`)

### 08 训练编排(源:`learning/training-orchestration/src/*.py`,6 lecture)
1. Slurm FIFO+Backfill 调度(`slurm_scheduler.py`)
2. Gang Scheduling 原子分配(`gang_scheduling.py`)
3. 故障容忍:MTBF 可加性+Young's Formula(`fault_tolerance.py`)
4. Ray Actor 编程模型(`ray_actors.py`)⭐
5. Ray 系统架构:GCS/调度器/lineage(`ray_original_minimal.py`)
6. Elastic Training rendezvous(`elastic_training.py`)
7. Capstone:512GPU 集群调度快照(`capstone_cluster_run.py`)

### 09 Module 8 毕业:端到端系统设计(源:`learning/infra-graduation/src/**/*.py`,叙事体capstone)
Mini-Cluster 模拟器(18场景 time-to-train+TCO)→ Topology Selector → MLPerf 对比(H100 vs B200)→ TCO 模型 → Portfolio v3(46-topic 全系列收官)。链接 [kernel-gpu-deep-dive](../kernel-gpu-deep-dive/00-roadmap.md)(gpu-architecture+kernel-engineering 部分),综合本系列 05-08 号文件全部内容。

---

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通,绝大部分纯 CPU、秒级;两处真实 GPU 训练 bonus 需独立复验(新鲜进程重新跑一次,不能只信第一次日志)。
- 每个文件挑 1-3 处最重要的结论,换参数/方法独立复现一遍,不能只信 agent 报告或源 README 数字。
- `infra-graduation` 的包导入陷阱、`scaling-infra` L08-L11 的去重处理、`sharding.py` 用 `hashlib.sha1`(非内置 `hash()`,不受 `PYTHONHASHSEED` 影响)——这些已知细节撰写时直接应用,不用重新踩坑。
- 每写完一批,在本文件进度表如实更新状态(⬜ 待撰写 → 🔧 撰写中 → ✅ 已完成,验证通过才标"已完成")。

---
*创建:2026-07-14*
