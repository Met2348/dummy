# guide_Accelerating Data Loading in Deep Neural Network Training

<!-- manual-deep-guide -->

> 原论文: [Accelerating Data Loading in Deep Neural Network Training](https://arxiv.org/abs/1910.01196)
>
> 本地原文 PDF: `learning/storage-dataops/paper/01_accelerating_data_loading.pdf`
>
> 本地导读 PDF: `learning/storage-dataops/paper/guide_01_accelerating_data_loading.pdf`
>
> 作者: Chih-Chieh Yang, Guojing Cong
>
> 机构: IBM T. J. Watson Research Center
>
> 版本: arXiv v1, 2019-10-02, HiPC 2019

## 0. 这篇 guide 怎么读

这篇论文讲的不是“写一个更漂亮的 PyTorch DataLoader”。它真正研究的是:

> 当分布式训练扩到很多节点时，数据加载为什么会成为训练扩展性的下界，以及怎样用缓存和 locality-aware loading 降低带宽需求。

如果你只看 GPU kernel、all-reduce、mixed precision，会很容易忽略 data path。可是在大规模训练里，一个 step 的时间可以拆成:

- computation time: forward/backward。
- communication time: gradient synchronization。
- data loading time: 从存储读样本、解码、增强、组 batch。

前两项经常被论文和工程团队重点优化，第三项容易被一句“我们提前 cache 了数据”带过。但本文的核心判断是: cache 到本地 SSD/DRAM 或预处理数据并不总是可行。真实 HPC/大集群里，很多训练仍然要从网络文件系统或数据服务器持续读数据。

读完这篇 guide，你应该能解释:

- 为什么普通 loader 在小规模能被 prefetch 隐藏，大规模却会暴露。
- 为什么 `D / R` 是常规数据加载的扩展性下界。
- multiprocessing、multithreading、software cache 分别解决哪一段 overhead。
- locality-aware data loading 为什么不改变同步 mini-batch SGD 的全局梯度。
- 为什么 load imbalance 需要补一点点数据搬运，而这个搬运远小于常规方案。
- 实验里 34x、55x、60x、120x 分别来自什么场景。

## 1. 当时的历史语境

2017 到 2019 年左右，分布式图像训练速度快速下降。ImageNet-1K + ResNet-50 从小时级被推进到分钟级。大家的注意力集中在:

- 更快的 GPU/加速器。
- cuDNN、MKL-DNN 这类优化库。
- mixed precision。
- 更好的 all-reduce 和 layer-wise synchronization。
- 大 batch 训练和学习率技巧。

这些方向都在降低 computation time 和 communication time。可是大规模训练的另一个事实是: 节点越多，每个 epoch 的模型计算越分散，但所有 learner 合起来仍然要消费同一个 dataset。存储系统的全局供给能力不是无限增长的。

作者的切入点很直接: 如果数据加载跟不上，GPU 等数据，整体 epoch time 就不再随节点数下降。此时继续加 GPU 可能只是在更贵地空转。

这和 MLPerf 的思想正好衔接: end-to-end time-to-quality 不能只看 GPU 算力。数据路径、存储路径、CPU decode/augmentation 都是训练系统能力的一部分。

## 2. 论文主张

论文提出四个贡献:

- 分析现有 data loader 的 performance 和 scalability 问题。
- 用 multiprocessing、multithreading、software cache 改善 data loading cost。
- 提出 analytical model，解释为什么 storage I/O rate 会限制扩展性。
- 提出 locality-aware data loading，用缓存位置重排 batch，减少 storage 和远程 cache 的总数据搬运。

摘要里最醒目的数字是:

- 在 256 nodes、1,024 learners 上，data loading 可超过 30x speedup。
- ImageNet-1K classification 中，优化让 per-epoch training cost 相比常规 distributed training 有 92% improvement。

但不要只记数字。更重要的是它告诉你一个系统设计原则:

> 当消费者数量扩大而数据源带宽不随之线性增长时，必须减少数据移动量，而不仅仅是让每个 worker 更会搬数据。

## 3. 原论文结构地图

建议按下面顺序读:

- Abstract/Introduction: 读数据加载被忽视的原因，以及 30x/92% 的 claim。
- Section II Background: 读 mini-batch SGD、step/epoch、data-parallel learners、三段训练成本。
- Figure 1: 这是动机图，说明节点增加后 waiting for data 开始占主导。
- Section III: 读 multiprocessing、multithreading、caching 分别优化什么。
- Figure 2/3: 读单 learner timeline 和 batch-level/sample-level 并行。
- Section IV: 慢读性能模型，这是论文的理论骨架。
- Section V: 慢读 locality-aware data loading、等价性证明、load imbalance 和 Algorithm 1。
- Section VI: 读实验环境、ImageNet/UCF101/MuMMI、Figure 7 到 Figure 12、Table I。
- Conclusion: 读作者如何概括数据加载对 large-scale HPC DNN training 的基础性作用。

## 4. 背景: 同步 mini-batch SGD 的数据路径

论文默认讨论 data-parallel synchronous mini-batch SGD。每个 learner 拿一份模型，流程是:

```text
for each training step:
    1. all learners receive the same global mini-batch index sequence
    2. each learner takes a disjoint slice
    3. each learner loads samples for its local batch
    4. each learner runs forward/backward
    5. all learners all-reduce gradients
    6. all learners update weights with the same global gradients
```

这里有两个关键概念:

- step: 训练一个 global mini-batch。
- epoch: 数据集被完整消费一轮。

普通实现通常把 global mini-batch sequence 按 block 切给各个 learner。比如 12 个样本、3 个 learner，每个 learner 固定拿 4 个。

```text
global batch: [0 1 2 3 4 5 6 7 8 9 10 11]

regular split:
    learner 0: [0 1 2 3]
    learner 1: [4 5 6 7]
    learner 2: [8 9 10 11]
```

如果这些样本都在远程文件系统中，那么每一步所有 learner 都按这个切片去读。节点越多，读请求越多，总消费速度越高，存储系统越容易被打满。

## 5. 现有 loader 为什么小规模看起来没问题

单 learner 下，如果 data loading 和 compute 串行，GPU 会等数据:

```text
load batch 0 -> compute batch 0 -> load batch 1 -> compute batch 1
```

常规优化是 prefetch，让后台 worker 提前准备后续 batch:

```text
worker:  load batch 1   load batch 2   load batch 3
main:    compute batch 0 compute batch 1 compute batch 2
```

只要 loader 准备 batch 的速度快于 GPU 消费速度，data loading overhead 就能被隐藏。

问题出在 scale-up:

- computation time 随节点数增加而下降，因为更多 GPU 分摊计算。
- preprocessing time 也会先下降，因为更多 CPU worker 参与处理。
- 但 storage system 的总 I/O rate 有上限。

于是某个规模以后，训练不再被 GPU compute 限制，而被“全局供数速度”限制。

## 6. Figure 1 的直觉

作者在 LLNL Lassen 上训练 ImageNet-1K + ResNet50，每个节点 4 learners，local batch size 固定 128，节点越多 global batch 越大。

Figure 1 展示:

- 2、4、8 nodes 时，waiting for data 很小，性能随节点数扩展。
- 到 16 nodes 以后，data waiting 开始明显。
- 节点继续增加，training time 还在下降，但 data loading cost 无法完全隐藏，最后主导 epoch cost。

这张图的含义是: 数据加载不是单机小优化，而是分布式训练扩展性问题。

## 7. Section III 的三类优化

**Multiprocessing**

PyTorch DataLoader 可以启动多个 background worker processes。主进程通过 queue 提交 batch-loading requests，worker 并行读样本和做 preprocessing。多 worker 能 overlap 多个 batch 的加载。

适合解决:

- batch 之间并行。
- I/O 请求并发。
- 多进程绕过一部分 Python GIL 限制。

**Multithreading**

在一个 worker 内，一个 batch 的多个 sample 可以并行 decode/transform。作者修改 PyTorch loader，在 worker 里用 `ThreadPoolExecutor.map()` 并行处理 samples。

注意 caveat: Python GIL 可能限制多线程，只有当 I/O 和图像变换调用 native library 并释放 GIL 时，多线程才明显有效。ImageNet JPEG decode 和 image transform 在实验中能受益。

**Caching**

mini-batch SGD 每个 epoch 都会重复消费同一个 dataset，只是顺序随机。因此有 temporal locality。

可以用:

- local DRAM cache: 最快，但容量有限。
- local SSD cache: 慢于内存，但容量更大。
- distributed cache: 所有节点 local cache 加起来，形成 aggregate cache。

普通 distributed cache 能减少从 storage system 读取，但如果每个 learner 仍然必须拿“指定切片”，它可能要从其他 learner 的 cache 拉很多样本。也就是说，它缓解 storage I/O，但不一定减少 total data movement。

## 8. 性能模型: 常规 loader 的下界

论文定义:

- `D`: dataset samples 数。
- `p`: compute nodes 数。
- `V`: 每个 compute node 的最大 training rate。
- `R`: storage system 的最大 I/O rate。
- `U`: 每个 compute node 的最大 preprocessing rate。

用 samples/sec 表示，单 epoch 的三项是:

```text
training_time = D / (p * V)
sample_io_time = D / R
preprocessing_time = D / (p * U)
data_loading_time = D / R + D / (p * U)
true_epoch_time = max(training_time, data_loading_time)
```

关键是 `D / R` 不随 `p` 下降。节点越多，`D / (p * V)` 和 `D / (p * U)` 都下降，但 `D / R` 是 storage 的常数下界。

如果假设 `U` 远大于 `V`，可以近似只比较 training time 和 sample I/O time:

```text
if D / (p * V) >= D / R:
    p <= R / V
    compute dominates
else:
    p > R / V
    storage I/O dominates
```

直觉:

```text
small p:
    true_epoch_time ~= D / (p * V)
    adding nodes helps

large p:
    true_epoch_time ~= D / R
    adding nodes no longer helps
```

这就是 Figure 1 的理论解释。

## 9. Distributed caching 的不足

设:

- `a`: aggregate cache 覆盖整个 dataset 的比例。
- `Rc`: 从 remote caches 读取样本的 I/O rate。

论文给出 distributed caching 的 sample I/O time:

```text
sample_io_time
  = (1 - a) * D / R
  + a * D / Rc * (p - 1) / p
```

第一项是 cache miss，从 storage system 读。第二项是 cached samples 但在别的节点上，需要通过节点间网络搬。

当 `p` 很大时:

```text
(p - 1) / p ~= 1
```

也就是说，即使整个 dataset 都在 aggregate cache 里，常规 distributed cache 仍然可能要在节点之间搬接近一个 dataset 的数据量。它不再压 storage，但仍然压 interconnect。

## 10. Locality-aware data loading 的核心想法

普通方法强制每个 learner 拿 global batch 的固定 block。locality-aware 方法放松这个固定切片:

> 只要一个 global mini-batch 的所有样本都参与本 step，样本在 learner 之间怎么分配并不影响同步 SGD 的全局梯度。

例子:

```text
global batch has 12 samples
target local batch per learner = 4

cache ownership:
    learner red   owns 2 samples in this batch
    learner green owns 6 samples
    learner blue  owns 4 samples

locality-aware before balance:
    red:   2
    green: 6
    blue:  4

balance:
    move 2 samples from green to red

data moved:
    2 / 12 = 16.7% of regular loading volume
```

这对应原文 Figure 4/5。重点不是“完全不搬数据”，而是“只搬为 load balance 所需的一小部分”。

## 11. 为什么梯度等价

设 global batch 是同一个样本集合。普通 Reg 方法把 batch block-distribute 给 learners；Loc 方法按 cache locality 重新分配，可能 local batch size 不同。

只要每个样本的 per-sample gradient 都被算一次，all-reduce 后的总梯度就是这些 per-sample gradients 的和。

```text
regular:
    global_grad = sum(grad(x) for x in global_batch)

locality-aware:
    global_grad = sum(grad(x) for x in permuted_global_batch)

because addition is commutative:
    both sums are equal
```

论文 Theorem 1 更形式化地说明: 在同一 random number sequence 下，distributed minibatch SGD 用 Reg 和 Loc 在相同步数后产生相同的 weights。

重要 caveat:

- 如果 batch normalization 按整个 global batch 计算统计量，等价性成立。
- 如果 batch normalization 按每个 local partition 计算 mean/variance，Loc 的 local partition 可能不同，统计量会变。作者认为影响类似换了一个 random permutation，并用实验验证 accuracy 差异很小。

## 12. Load imbalance 和 Algorithm 1

Locality-aware 方法可能让每个 learner 找到的本地样本数量不同。同步 SGD 里，local batch 太大的 learner 会成为 straggler，因此需要 balance。

论文把问题抽象成:

- 有些 learner 有 surplus。
- 有些 learner 有 deficit。
- surplus learner 给 deficit learner 发送一些样本。
- 目标是减少 message 数和数据搬运。

Algorithm 1 使用两个 heap:

- `Hs`: surplus heap，按 surplus 从大到小。
- `Hd`: deficit heap，按 deficit 从大到小。

每次取最大 surplus 和最大 deficit，移动二者较小值，然后更新 heap。

```text
while surplus exists:
    src = largest surplus learner
    dst = largest deficit learner
    moved = min(src.surplus, dst.deficit)
    schedule transfer(src, dst, moved)
    update both heaps
```

复杂度:

```text
O(p log p)
```

论文 Theorem 2 说明这是 2-approximation algorithm。直觉是最多发送 `p - 1` 条消息，而最优下界约 `p / 2` 条。

论文还用 balls-in-bins 模型和 simulation 说明 imbalance 通常不大。Figure 6 里 local batch size 32、64、128 的 median imbalance roughly 是:

- 32: 6.9%。
- 64: 4.8%。
- 128: 3.4%。

这说明 locality-aware 的额外 balance traffic 通常是 batch 的小比例。

## 13. Locality-aware 的 I/O 模型

设:

- `Rb`: load balancing data movement 的 I/O rate。
- `b`: load balancing traffic ratio，相对 dataset size。

locality-aware sample I/O time:

```text
sample_io_time
  = (1 - a) * D / R
  + a * D * b / Rb
```

和 distributed cache 比较:

```text
distributed cache:
    second term ~= a * D / Rc

locality-aware:
    second term ~= a * D * b / Rb
```

当 `b` 很小，比如 0.03 到 0.07，locality-aware 的数据搬运量就远小于普通 distributed cache。它的胜利点不是 remote cache 更快，而是根本少搬。

## 14. 本仓库的最小代码对应

这次新增:

- `learning/storage-dataops/src/data_loading_original_minimal.py`

它包含:

- `CostModel`: 对应公式 `D/(pV)`、`D/R`、`D/(pU)` 和 `max(...)`。
- `distributed_cache_io_time`: 对应论文 equation 7。
- `locality_aware_io_time`: 对应论文 equation 8。
- `sample_distribution`: 给定 batch 和 cache owner，统计每个 learner 拥有多少样本。
- `balance_transfers`: surplus-to-deficit greedy balance，对应 Algorithm 1 的 toy 版。
- `partitions_equivalent`: 用 per-sample gradient sum 验证 Reg 和 Loc 的全局梯度等价。

最小示例:

```python
batch = list(range(12))
owner = {
    0: 0, 1: 0,
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,
    8: 2, 9: 2, 10: 2, 11: 2,
}

counts = sample_distribution(batch, owner, n_nodes=3)
transfers = balance_transfers(counts, target_per_node=4)

assert counts == [2, 6, 4]
assert transfers == [(1, 0, 2)]
```

这段正好对应 Figure 5: learner 1 多 2 个样本，发给 learner 0，搬运量是 `2 / 12`。

## 15. 实验设置

作者在 Lawrence Livermore National Lab 的 Lassen 上做实验:

- up to 256 nodes。
- 每节点 2 个 IBM POWER9 processors，44 CPU cores total。
- 256 GB system memory。
- 4 Nvidia V100 GPUs，每 GPU 16 GB。
- InfiniBand EDR interconnect。
- IBM Spectrum Scale GPFS 并行文件系统。

主要数据集:

- ImageNet-1K: 约 1.28M JPEG images，总大小约 150 GB。
- UCF101-RGB: 约 2.5M images，平均 24.2 KB。
- UCF101-FLOW: 约 5M images，平均 4.6 KB。
- MuMMI: 892 GB，约 7M files，每个 131 KB，numpy array，无需额外 preprocessing。

这个设置很重要，因为不同数据格式触发不同瓶颈:

- JPEG 有 decode/transform，multithreading 有价值。
- numpy frame 不需要 preprocessing，瓶颈更纯粹是 I/O 和数据移动。

## 16. 实验证据链条

**Evidence 1: Figure 7, 单 learner loader 优化**

作者在 ImageNet-1K 上测试 worker/thread 组合。多 workers 和多 threads 通常都提高 sample loading rate。最大 loading rate 约 800 samples/sec。多线程能用较少 workers 达到更好性能，避免 worker 数太多带来的进程开销。

**Evidence 2: Figure 8, ImageNet data loading scalability**

常规 PyTorch loader 在 scale up 后加载成本不再下降，因为有效 I/O bandwidth 被打满。locality-aware loader 因为复用 cache 并减少搬运，随 learner 数增加仍然更可扩。

在 256 nodes、1,024 learners 上，locality-aware loader 对 ImageNet-1K data loading 达到接近 34x speedup。

多线程也有效:

- 常规 loader: multithreaded runs 快 24% 到 71%。
- locality-aware loader: multithreaded runs 快 105% 到 113%。

但关键结论是: multithreading 只能提高单 learner/loading pipeline 性能，不能解决常规 loader 的全局 I/O scaling 下界。

**Evidence 3: Figure 9/10, UCF101**

UCF101-RGB 和 UCF101-FLOW 的数据加载实验显示 locality-aware 方法在不同 scale 下更好:

- RGB: 2.8x 到 55.5x speedup。
- FLOW: 2.2x 到 60.6x speedup。

其中常规 loader 的曲线还会受到其他 GPFS jobs 干扰，说明共享并行文件系统环境下，数据加载性能不仅取决于自己的代码，也取决于集群整体 I/O 竞争。

**Evidence 4: Figure 11, MuMMI**

MuMMI 是 892 GB、约 7M files 的大数据集，文件已经是 numpy array，不需要 preprocessing。因此 multithreading 对它影响不大。

locality-aware loading 的 speedup 更明显:

- 16 nodes: 18x。
- 32 nodes: 35x。
- 64 nodes: 70x。
- 128 nodes: 120x。

这个实验说明，当 preprocessing 不再是瓶颈时，减少数据搬运量本身就是主要收益。

**Evidence 5: Table I, accuracy 不被明显破坏**

ImageNet-1K + ResNet50 训练 90 epochs，对比 regular loader 和 locality-aware loader:

- 16 nodes, batch 8192: 76.67% vs 76.81%。
- 32 nodes, batch 16384: 75.33% vs 75.12%。
- 64 nodes, batch 32768: 68.69% vs 69.54%。

作者强调差异小于 1%，说明 locality-aware batch 重排没有明显破坏 validation accuracy。64 nodes 绝对精度偏低是因为大 batch accuracy 本来需要 LARS 和复杂 LR tuning，作者没有把目标放在追最高精度上。

**Evidence 6: Figure 12, 实际 GPU training**

在 ImageNet-1K + ResNet50 的实际训练中:

- 16 nodes 时，GPU training time 主导，两个 loader epoch time 接近。
- 32/64 nodes 时，常规 loader 受 `D/R` 下界限制，epoch time 下降受阻。
- locality-aware loader 继续随节点数受益。
- 64 nodes、256 learners 上，per-epoch 约 1.9x speedup。

Conclusion 里还总结: 应用于实际 ImageNet classification 时，使用该 data loader 大约给 1,024 learners 带来 2x speedup，并保持可比 validation accuracy。

## 17. 局限性

第一，方法依赖重复访问同一 dataset 的 temporal locality。如果数据是严格 streaming、每个样本只看一次，cache 的意义会下降。

第二，方法依赖可查询的 cache directory。论文假设 cache location 可复制到所有 learners，并且 first epoch 后不做 cache replacement。动态数据集或频繁 eviction 会增加复杂度。

第三，locality-aware 等价性对 batch normalization 有 caveat。如果 BN 不是 global batch 统计，而是 local batch 统计，重排会改变 local statistics。

第四，locality-aware 会改变每个 learner 的 local batch size，需要 balance。虽然论文说明 imbalance 小，但极小 local batch、非均匀 sample sizes 或强 skew 分布可能增加 straggler 风险。

第五，prototype 基于 PyTorch，并非框架无关产品。作者也把未来工作放在通用软件包、SSD hierarchical cache 和更多 ML optimization methods 上。

第六，实验集中在特定 HPC 环境和数据集。GPFS、InfiniBand、V100、ImageNet/UCF101/MuMMI 的结论不能无脑外推到所有云对象存储或 LLM token 数据管道。

## 18. 现代意义

今天做 LLM 训练时，你会遇到同样问题，只是样本从 JPEG/video frame 变成 token shard、document chunk、multi-modal sample 或 RL rollout。

这篇论文给你的判断框架仍然有效:

- 先把 step time 拆成 compute、communication、data loading。
- 如果 data loading 能被 overlap 隐藏，优化 GPU/kernel 更有价值。
- 如果 data loading 暴露，先判断瓶颈是 storage I/O、decode/tokenization、augmentation、collate，还是 distributed shuffle。
- 如果扩展到更多节点后速度 plateau，检查是否出现 `D/R` 这样的全局带宽下界。
- 如果缓存命中率高但网络仍忙，检查是否是“数据已经在集群里但被搬错地方”。
- 如果要改 sampling 或 shard assignment，必须证明不改变训练语义，或者明确质量影响。

对现代 LLM dataops 的映射:

- WebDataset/tar shards: 减少小文件 IOPS，提升顺序读。
- Streaming dataset: 改善远端对象存储读取，但要管理 shuffle 和重复性。
- Local NVMe cache: 类似本文 local cache。
- Dataset sharding by rank/node: 类似 locality-aware 的精神，减少跨节点搬运。
- Async checkpoint: 同样是把存储 I/O 从训练关键路径上移开。

## 19. 本仓库学习路径

相关 lecture:

- `learning/storage-dataops/lectures/02-dataloader.md`

相关代码:

- `learning/storage-dataops/src/data_loading_original_minimal.py`
- `learning/storage-dataops/src/dataloader.py`
- `learning/storage-dataops/src/sharding.py`
- `learning/storage-dataops/src/webdataset_style.py`
- `learning/storage-dataops/src/checkpoint.py`
- `learning/storage-dataops/src/capstone_ckpt_recovery.py`

建议 30 到 60 分钟实验:

1. 运行测试:

```powershell
.venv\Scripts\python.exe learning\storage-dataops\src\tests\test_all.py
```

2. 打开 `data_loading_original_minimal.py`。
3. 把 `storage_rate` 从 `100_000` 改成 `1_000_000`，观察 regular epoch plateau 如何移动。
4. 把 `balance_ratio` 从 `0.05` 改成 `0.30`，观察 locality-aware 相对 distributed cache 的优势如何变小。
5. 改 Figure 5 toy batch 的 cache owner，让 counts 变成 `[0, 8, 4]`，观察 balance transfers 和 imbalance ratio。
6. 在 `dataloader.py` 里把 `decode_jpeg` 的时间减半，观察瓶颈 stage 是否改变。

预期观察:

- storage_rate 越高，普通 loader 能扩到更大 `p`。
- balance_ratio 越大，locality-aware 搬运越多，收益越低。
- local batch 越不均衡，load balancing traffic 越高。
- 优化非瓶颈 stage 不一定改善整体 throughput。

## 20. 常见误读

**误读 1: 加 DataLoader workers 就能解决大规模数据加载。**

不对。workers/threads 能提高单节点 loader throughput，但不能消除全局 storage I/O 下界。

**误读 2: 数据都 cache 了就没有问题。**

不对。普通 distributed cache 可能仍然要跨节点搬接近整个 dataset 的数据。locality-aware 的核心是减少 total movement。

**误读 3: Locality-aware 改了每个 learner 的样本，所以训练不等价。**

不一定。同步 SGD all-reduce 后看的是 global batch gradient sum。只要 global batch 样本集合相同，重排不改变总梯度。BN 等局部统计是 caveat。

**误读 4: 本文只适合图像，不适合 LLM。**

不对。具体 decode/augmentation 是图像场景，但 `D/R` 下界、cache locality、shard placement、数据搬运最小化对 LLM token data pipeline 仍然成立。

**误读 5: 数据加载优化一定提高最终精度。**

不对。它主要降低 time/epoch 或 time-to-quality。精度只需要不被破坏，Table I 的重点也是 comparable accuracy。

## 21. 用 AI agent 学这篇论文

推荐提示词:

```text
我正在读 Accelerating Data Loading in Deep Neural Network Training。
请一次只问一个问题，并要求我把答案映射到公式或代码:
1. 为什么普通 loader 的 epoch time 会在大规模下 plateau？
2. multiprocessing、multithreading、caching 分别解决什么？
3. 请我写出 regular、distributed cache、locality-aware 的 I/O 公式。
4. 为什么 locality-aware 不改变同步 SGD 的 global gradient？
5. Algorithm 1 的 surplus/deficit balance 如何工作？
6. 请让我在 data_loading_original_minimal.py 里改 storage_rate 或 balance_ratio 并预测输出。
```

也可以让 agent 做 debug 教练:

```text
假设我的 LLM 训练 GPU utilization 只有 55%。
请按本文思路帮我排查: storage I/O、decode/tokenization、collate、
prefetch、cache hit、rank sharding、global bandwidth plateau。
每一步都要给一个可观测指标和一个最小实验。
```

## 22. 闭卷掌握检查

1. 大规模同步 mini-batch SGD 的 step 包含哪些阶段？
2. 为什么 prefetch 可以隐藏 data loading？什么时候隐藏不了？
3. 写出 `training_time`、`data_loading_time` 和 `true_epoch_time`。
4. 为什么 `D / R` 会成为常规 loader 的扩展性下界？
5. Multiprocessing 和 multithreading 的并行粒度有什么不同？
6. Distributed cache 为什么仍然可能产生大量跨节点数据搬运？
7. Locality-aware data loading 和普通 block split 的核心区别是什么？
8. 为什么 locality-aware 方法的 global gradient 和 regular 方法相同？
9. Batch normalization 为什么是一个 caveat？
10. Algorithm 1 如何用 surplus/deficit heap 做 balance？
11. Figure 6 的 imbalance 数字说明什么？
12. ImageNet、UCF101、MuMMI 三组实验分别验证了什么？
13. Table I 为什么重要？它证明了什么，没有证明什么？
14. 在本仓库里，哪个函数对应 equation 7？哪个函数对应 equation 8？哪个函数对应 Figure 5 的搬运比例？

真正掌握的标志是: 你能看到一个训练集群 utilization 下降，就把问题拆成 compute、communication、storage I/O、preprocess、cache locality 和 load balance，而不是只盯 GPU 峰值。
