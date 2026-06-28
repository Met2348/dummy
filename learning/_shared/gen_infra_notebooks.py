"""为 7 个 nb=0 基础设施旧专题回补 notebook (导入既有 src + 跑 demo + 可视化)。
运行: python learning/_shared/gen_infra_notebooks.py"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEARNING = Path(__file__).resolve().parent.parent
def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)
MPL = """import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus']=False
for f in ['Microsoft YaHei','SimHei','DejaVu Sans']:
    try: matplotlib.rcParams['font.sans-serif']=[f]; break
    except Exception: pass"""
PRE = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np"""

def write(topic, fname, cells):
    d = LEARNING / topic / "notebooks"; d.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(); nb.cells = cells
    nbformat.write(nb, d / fname); print(f"written {topic}/notebooks/{fname}")


# 1. cuda-essentials: online softmax (flash 的核心思想)
write("cuda-essentials", "N1-online-softmax.ipynb", [
    md("""# N1 · online softmax (flash attention 的核心思想)

> 复用 `src/capstone_softmax.py` · 对比 naive softmax (两遍/存全部) vs online softmax (一遍/常数内存)。
> online softmax 是 flash attention 不爆显存的关键 (接本专题讲义)。"""),
    code(PRE + "\nimport capstone_softmax as cs\nprint('cuda-essentials src 就绪 (纯 Python 模拟 kernel)')"),
    md("## 1. naive vs online softmax: 结果一致, 但 online 单遍 + 数值稳定"),
    code("""x = [3.0, 1.0, 0.2, 5.0, 2.5, -1.0, 4.0]
naive = cs.softmax_naive(x)
online = cs.softmax_online(x)
print('naive  softmax:', [round(v,4) for v in naive])
print('online softmax:', [round(v,4) for v in online])
print('两者最大差异:', round(max(abs(a-b) for a,b in zip(naive, online)), 8))
print('→ 结果一致; online 只过一遍数据 + 在线维护 max/sum (常数内存, 数值稳定)。')"""),
    md("## 2. 为什么 online softmax 是 flash attention 的钥匙"),
    code("""print('''online softmax 机制 (本专题讲义):
  naive:  ① 求 max ② 求 sum(exp(x-max)) ③ 除  -> 要存全部 x (两/三遍)
  online: 边读边更新 running_max 和 running_sum, 遇到更大的 max 就 rescale
          -> 单遍, 常数内存, 不存全部分数

为什么关键 (flash attention):
  attention 的 softmax 在 S×S 的分数矩阵上 -> 存全部 = O(S^2) 显存爆炸
  online softmax 让 softmax 能分块 (tiling) 流式算 -> flash attention 不存全矩阵
  = kernel-engineering 专题 flash attention 的数学基础''')"""),
    md("> 本专题其余 src (`vector_add` / `gemm_tiled` / `reduce_kernel` / `shared_memory` / `coalescing` / `warp_primitives`) 是 CUDA 核心概念的纯 Python 模拟, 可在对应讲义里运行其 `_self_test()`。"),
])

# 2. gpu-architecture: roofline
write("gpu-architecture", "N1-roofline.ipynb", [
    md("""# N1 · roofline 模型: compute-bound vs memory-bound

> 复用 `src/roofline.py` + `capstone_roofline_zoo.py` · 算不同算子的算术强度, 判断它撞的是算力墙还是带宽墙。
> roofline 是 GPU 性能分析的核心工具 (接本专题讲义)。"""),
    code(PRE + "\nimport roofline as rf\nimport capstone_roofline_zoo as zoo\nprint('gpu-architecture src 就绪')"),
    md("## 1. 算子动物园: 各算子是 compute-bound 还是 memory-bound"),
    code("""results = zoo.run()
print(f\"{'算子':<22} {'强度(FLOP/B)':>12} {'利用率%':>8} {'瓶颈':>10}\")
for r in results[:12]:
    print(f\"{str(r['op']):<22} {r['ai']:>12.1f} {r['utilization_pct']:>8.1f} {str(r['bound_by']):>10}\")"""),
    md("## 2. roofline 图 (算术强度 vs 可达性能)"),
    code(MPL + """
try:
    s = zoo.summarize(results)
    print('summary:', s)
except Exception as e:
    print('summarize:', e)
# 画 roofline: 性能 = min(峰值算力, 带宽 × 算术强度)
peak_flops = 312e12; bw = 2e12   # 示意: ~A100 bf16
ai = np.logspace(-1, 3, 100)
perf = np.minimum(peak_flops, bw*ai)
plt.figure(figsize=(7,4.5))
plt.loglog(ai, perf/1e12, 'k-', lw=2, label='roofline (屋顶)')
plt.axvline(peak_flops/bw, ls='--', c='gray', alpha=0.6, label='脊点 (compute/memory 分界)')
for r in results:
    a = r.get('ai', None)
    if a: plt.scatter([a],[min(peak_flops, bw*a)/1e12], s=40, alpha=0.6, zorder=5)
plt.xlabel('算术强度 (FLOP/Byte)'); plt.ylabel('可达性能 (TFLOP/s)')
plt.title('roofline: 左=memory-bound(带宽墙), 右=compute-bound(算力墙)'); plt.legend(); plt.tight_layout(); plt.show()
print('→ 算子落在脊点左侧=memory-bound(优化带宽/融合), 右侧=compute-bound(优化算力/tensor core)。')"""),
])

# 3. kernel-engineering: flash attention 内存
write("kernel-engineering", "N1-flash-attention-memory.ipynb", [
    md("""# N1 · flash attention: 内存从 O(S²) 降到 O(S)

> 复用 `src/capstone_attn_speedup.py` · flash attention 用 tiling + online softmax (接 cuda-essentials),
> 不存 S×S 分数矩阵 → 内存线性。看内存随序列长度的对比。"""),
    code(PRE + "\nimport capstone_attn_speedup as ca\nprint('kernel-engineering src 就绪')"),
    md("## 1. naive (存全矩阵 O(S²)) vs flash (tiling O(S)) 的内存"),
    code(MPL + """
rows = ca.speedup_curve()
seq = [r['seq_len'] for r in rows]
naive = [r['naive_mb'] for r in rows]
flash = [r['flash_mb'] for r in rows]
sp = [r['speedup'] for r in rows]
print(f\"{'序列长':>8} {'naive MB':>10} {'flash MB':>10} {'省内存':>8}\")
for r in rows: print(f\"{r['seq_len']:>8} {r['naive_mb']:>10} {r['flash_mb']:>10} {r['speedup']:>7}x\")
plt.figure(figsize=(7,4.3))
plt.plot(seq, naive, 'o-', label='naive (O(S²) 存全矩阵)', color='C3')
plt.plot(seq, flash, 's-', label='flash (O(S) tiling)', color='C0')
plt.xlabel('序列长度 S'); plt.ylabel('峰值内存 (MB)'); plt.legend()
plt.title('flash attention: 内存 O(S²)→O(S) (不存 S×S 分数矩阵)'); plt.tight_layout(); plt.show()
print('→ naive 内存随 S² 爆炸; flash 用 tiling+online softmax 线性 → 长序列可行 (本专题核心)。')"""),
    md("""## 2. 反思
- flash attention = tiling (分块) + online softmax (cuda-essentials) → 不物化 S×S 矩阵。
- 内存 O(S²)→O(S), 是长上下文 (long-context) 可行的关键 kernel。
- 本专题其余 src (`fused_mlp`/`rmsnorm_kernel`/`triton_style`/`cutlass_layout`) 是 kernel 融合/布局的模拟。"""),
])

# 4. cluster-networking: allreduce 算法
write("cluster-networking", "N1-allreduce-algorithms.ipynb", [
    md("""# N1 · allreduce 算法: ring vs tree vs halving-doubling

> 复用 `src/capstone_cluster_sim.py` · 分布式训练的梯度同步 (allreduce) 是通信瓶颈。
> 不同算法/拓扑下哪个最快? 看模拟对比 (接本专题讲义)。"""),
    code(PRE + "\nimport capstone_cluster_sim as sim\nprint('cluster-networking src 就绪')"),
    md("## 1. 不同集群规模/链路下的最优 allreduce 算法"),
    code(MPL + """
rows = sim.run()
print(f\"{'GPU数':>6} {'链路':<16} {'最优算法':<16} {'最优(s)':>9} {'ring(s)':>9}\")
for r in rows:
    print(f\"{r['n_gpus']:>6} {str(r['link']):<16} {str(r['best_algo']):<16} {r['best_time_s']:>9} {r['ring_time_s']:>9}\")
n = [r['n_gpus'] for r in rows]; best=[r['best_time_s'] for r in rows]; ring=[r['ring_time_s'] for r in rows]
plt.figure(figsize=(7,4.3))
plt.plot(n, ring, 'o-', label='ring allreduce', color='C3')
plt.plot(n, best, 's-', label='最优算法 (自动选)', color='C0')
plt.xlabel('GPU 数'); plt.ylabel('allreduce 时间 (s)'); plt.legend()
plt.title('allreduce: 不同规模下最优算法 vs ring'); plt.tight_layout(); plt.show()
print('→ 小规模 ring 够好; 大规模/特定拓扑下 tree/halving-doubling 更优 (本专题: 算法随拓扑选)。')"""),
    md("> 本专题其余 src (`allreduce_algos`/`nccl_collectives`/`fabric_topology`/`sharp_inline`) 是集合通信与网络拓扑的模拟。"),
])

# 5. storage-dataops: checkpoint 策略
write("storage-dataops", "N1-checkpoint-strategies.ipynb", [
    md("""# N1 · checkpoint 策略: full vs sharded vs async

> 复用 `src/capstone_ckpt_recovery.py` · 大规模训练要定期存 checkpoint (容错)。
> 不同策略的开销 (阻塞/恢复/浪费) 怎么权衡? 看模拟 (接本专题讲义)。"""),
    code(PRE + "\nimport capstone_ckpt_recovery as ck\nprint('storage-dataops src 就绪')"),
    md("## 1. 三种 checkpoint 策略的开销对比"),
    code(MPL + """
strategies = ['full','sharded','async']
rows = [ck.total_overhead(s) for s in strategies]
print(f\"{'策略':<8} {'单次(s)':>9} {'阻塞(min)':>11} {'恢复(h)':>9} {'浪费%':>8}\")
for r in rows:
    print(f\"{r['strategy']:<8} {r['per_ckpt_s']:>9} {r['blocking_total_min']:>11} {r['recovery_total_h']:>9} {r['wasted_pct']:>7}%\")
import numpy as np
labels=strategies; block=[r['blocking_total_min'] for r in rows]; waste=[r['wasted_pct'] for r in rows]
x=np.arange(len(labels)); w=0.35
fig,ax=plt.subplots(figsize=(7,4.2))
ax.bar(x-w/2, block, w, label='阻塞时间 (min)', color='C3')
ax2=ax.twinx(); ax2.bar(x+w/2, waste, w, label='浪费比例 %', color='C0')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('阻塞 (min)', color='C3'); ax2.set_ylabel('浪费 %', color='C0')
ax.set_title('checkpoint 策略: 阻塞 vs 浪费 权衡'); plt.tight_layout(); plt.show()
print('→ full 简单但阻塞久; sharded 分片省阻塞; async 异步几乎不阻塞 (本专题: 按规模选策略)。')"""),
    md("> 本专题其余 src (`dataloader`/`sharding`/`webdataset_style`/`checkpoint`) 是数据管线与分片的模拟。"),
])

# 6. training-orchestration: 24h 集群模拟
write("training-orchestration", "N1-cluster-simulation.ipynb", [
    md("""# N1 · 集群编排: 24 小时容错训练模拟

> 复用 `src/capstone_cluster_run.py` · 大集群训练要处理节点故障/弹性/调度。
> 模拟 24 小时运行, 看容错编排的效果 (接本专题讲义)。"""),
    code(PRE + "\nimport capstone_cluster_run as cr\nprint('training-orchestration src 就绪')"),
    md("## 1. 模拟 24 小时大集群训练 (含故障/弹性)"),
    code("""res = cr.simulate_24h(n_nodes=64, gpus_per_node=8)
print('24 小时集群模拟结果:')
for k, v in res.items():
    print(f'  {k}: {v}')
print('\\n→ 容错编排 (弹性/gang 调度/故障恢复) 让大集群在节点会挂的现实下仍高利用率 (本专题核心)。')"""),
    md("> 本专题其余 src (`elastic_training`/`fault_tolerance`/`gang_scheduling`/`ray_actors`/`slurm_scheduler`) 是编排机制的模拟。"),
])

# 7. infra-graduation: 装配
write("infra-graduation", "N1-infra-capstone.ipynb", [
    md("""# N1 · 基础设施 capstone: 全栈装配

> 复用 `src/portfolio_v3.py` · 基础设施模块毕业: 把 GPU/kernel/网络/存储/编排串成训练栈。"""),
    code(PRE + "\nimport portfolio_v3 as pv\nprint('infra-graduation src 就绪')"),
    md("## 1. 运行基础设施 capstone 自检"),
    code("""# portfolio_v3 提供基础设施栈的装配/portfolio 写出
try:
    pv._self_test()
    print('\\n基础设施 capstone 自检通过 (GPU架构→kernel→网络→存储→编排 全栈)。')
except Exception as e:
    print('self_test:', e)"""),
    md("""## 2. 反思
基础设施栈全景: GPU 架构 (roofline) → kernel 工程 (flash) → 集群网络 (allreduce) → 存储/数据 (checkpoint) → 训练编排 (容错)。这是训练大模型的工程底座。"""),
])

print("done.")
