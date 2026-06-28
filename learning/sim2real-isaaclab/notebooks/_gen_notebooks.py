"""生成 11.6 notebooks (N1 DR弥合gap GPU-free / N2 IsaacLab真跑指引). 跑后 nbconvert --execute。"""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)
MPL = """import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus']=False
for f in ['Microsoft YaHei','SimHei','DejaVu Sans']:
    try: matplotlib.rcParams['font.sans-serif']=[f]; break
    except Exception: pass"""
PATHS = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import domain_rand as dr
import toy_env as env   # M11.1 共享
import numpy as np, torch"""

# ── N1 domain-randomization ──
n1 = new_notebook()
n1.cells = [
    md("""# N1 · domain randomization 弥合 sim2real gap (GPU-free)

> 配套 11.6-L1/L3 · GPU-free 玩具坐实 DR: 随机化的「环境参数」= 目标分布。
> **窄区域 (sim) 训** vs **全区域 (DR) 训**, 都在**真实全区域**评估。
> 看 DR 怎么把覆盖不足的 gap 补上 (覆盖→泛化, 同 M11.4)。"""),
    code(PATHS + "\nprint('环境 = M11.1 2D 到达; 域随机化作用在目标分布 (narrow=sim, wide=real/DR)')"),
    md("## 1. 训两个策略: 无 DR (窄 sim) vs DR (全区域)"),
    code("""torch.manual_seed(0)
# 无 DR: 只在窄区域 (右上小块) 采专家 demo 训练
S0, A0 = dr.collect_demos(n=400, region='narrow', seed=0)
m0 = dr.build_policy(seed=0); dr.train_policy(m0, S0, A0)
# DR: 随机化目标到全区域采 demo 训练
Sd, Ad = dr.collect_demos(n=400, region='wide', seed=0)
md_ = dr.build_policy(seed=0); dr.train_policy(md_, Sd, Ad)
print('两个策略训练完毕 (无DR=窄区域目标, DR=全区域目标)')"""),
    md("## 2. sim2real gap: 在 sim(窄) 和 real(全) 上分别评估"),
    code(MPL + """
res = {
 '无 DR (窄sim训)': (dr.eval_region(m0,'narrow'), dr.eval_region(m0,'wide')),
 'DR (全区域训)':   (dr.eval_region(md_,'narrow'), dr.eval_region(md_,'wide')),
}
for k,(sim,real) in res.items():
    print(f'{k:16}: sim(窄) {sim:.2f}  real(全) {real:.2f}')
x=np.arange(2); w=0.35
fig,ax=plt.subplots(figsize=(7,4.2))
ax.bar(x-w/2, [res['无 DR (窄sim训)'][0], res['无 DR (窄sim训)'][1]], w, label='无 DR', color='C3')
ax.bar(x+w/2, [res['DR (全区域训)'][0], res['DR (全区域训)'][1]], w, label='DR', color='C0')
ax.set_xticks(x); ax.set_xticklabels(['sim (窄区域目标)','real (全区域目标)']); ax.set_ylabel('成功率'); ax.set_ylim(0,1.05); ax.legend()
ax.set_title('无 DR 在 real 掉点 (sim2real gap); DR 拉回 (覆盖→泛化)')
plt.tight_layout(); plt.show()
print('→ 无 DR: sim 好但 real 掉 (没见过的远目标失败); DR: 训练覆盖全配置 → real 泛化。')"""),
    md("## 3. 看 gap 在哪: 无 DR 在哪些目标失败"),
    code(MPL + """
fig, axes = plt.subplots(1,2,figsize=(11,5))
for ax,(name,model) in zip(axes, [('无 DR (窄sim训)', m0), ('DR (全区域训)', md_)]):
    # 在全区域网格目标上测成功与否
    for gx in np.linspace(-0.85,0.85,9):
        for gy in np.linspace(-0.85,0.85,9):
            rng=np.random.default_rng(int((gx+1)*1000+(gy+1)*7))
            s=np.array([rng.uniform(-0.8,0.8),rng.uniform(-0.8,0.8),gx,gy],np.float32)
            ok=False
            for _ in range(env.MAX_T):
                a=model(torch.tensor(s[None],dtype=torch.float32)).detach().numpy()[0]
                s=env.step(s,a)
                if env.is_success(s): ok=True; break
            ax.plot(gx,gy,'o',color=('C2' if ok else 'C3'),ms=9)
    # 标出窄 sim 训练区域
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((0.3,-0.3),0.6,0.6,fill=False,ec='blue',lw=2,ls='--'))
    ax.set_title(f'{name}\\n绿=成功 红=失败 (蓝框=窄sim训练区)'); ax.set_aspect('equal'); ax.set_xlim(-1,1); ax.set_ylim(-1,1)
plt.suptitle('无 DR 只在蓝框(见过)附近成功, 框外失败; DR 全区域成功'); plt.tight_layout(); plt.show()
print('→ gap 的真相: 无 DR 只在训练覆盖的配置(蓝框)可靠, 框外=没见过=失败。DR 扩覆盖→处处成功。')"""),
    md("""## 4. 反思
你实测了 DR 弥合 sim2real gap。带走:
- **sim2real gap = 覆盖不足**: 窄 sim 训的策略只在见过的配置可靠, real 的新配置失败 (同 M11.4 分布漂移)。
- **DR = 用覆盖换泛化**: 训练随机化到覆盖真实 → real 不再"没见过" → 泛化。
- 真实 IsaacLab 里随机化的是摩擦/光照/传感器噪声等; 机制一样 (覆盖→泛化)。
下一步 N2: 真 IsaacLab 怎么跑 (需 NV GPU, 复用你 AntBot 经验)。"""),
]
nbformat.write(n1, HERE / "N1-domain-randomization.ipynb")
print("written N1")

# ── N2 isaaclab-run-guide ──
n2 = new_notebook()
n2.cells = [
    md("""# N2 · IsaacLab 真跑指引 (需 NV GPU)

> 配套 11.6-L2 · 这是一个**指引** notebook (不在 CPU 上跑真 IsaacLab, 需 NVIDIA GPU + Isaac Sim)。
> 复用本人 AntBot 实战 + WSL2 踩坑经验 (详见 `src/isaaclab_notes.md`)。
> 下面的 code cell 只做**环境检测 + 打印命令**, 不实际启动 IsaacLab (保证 nbconvert 安全)。"""),
    md("""## 1. 结论先行 (本人踩坑)
> ❌ **WSL2 + NV591 + Ubuntu22.04 + IsaacLab 5.1.0 → 失败** (Vulkan 不支持, headless 也死锁)
> ✅ **Windows 原生 + conda `env_isaaclab` + Isaac Sim 5.1 → 成功跑 AntBot**
> 教训: IsaacLab 对图形栈 (Vulkan/驱动) 极挑剔; **用 Windows 原生 / 原生 Ubuntu + 官方驱动**。"""),
    md("## 2. 环境检测 (只读, 不启动 IsaacLab)"),
    code("""import shutil, importlib.util, platform
print('平台:', platform.system(), platform.release())
# 检测 NVIDIA GPU
has_smi = shutil.which('nvidia-smi') is not None
print('nvidia-smi 可用:', has_smi, '(IsaacLab 需 NVIDIA GPU)')
# 检测 isaaclab / isaacsim (一般不在本环境)
for mod in ['isaaclab', 'isaacsim', 'omni']:
    print(f'  {mod} 模块:', '在' if importlib.util.find_spec(mod) else '不在 (正常, 真跑需专门安装)')
print('\\n→ 本课 CPU 环境不装 IsaacLab; 下面给真跑命令, 在你的 GPU 机器上执行。')"""),
    md("## 3. 真跑流程 (在 NV GPU 机器上, Windows 原生)"),
    code(r"""guide = r'''
========== IsaacLab 真跑流程 (Windows 原生, 复用 AntBot 经验) ==========

[步骤 1] 验证最小启动 (先确认环境, 别一上来就训练):
  isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
  关键确认: [INFO][AppLauncher]: Using device: cuda:0   <- GPU 在用

[步骤 2] 训练 AntBot (并行 RL):
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --headless
  --headless = 无窗口纯算(快, 训练用); 去掉则开窗可视化(慢, debug)

[步骤 3] 回放/评估训出的策略:
  isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Ant-v0

[踩坑提醒] (详见 src/isaaclab_notes.md):
  - 避开 WSL2 (Vulkan 支持差, headless 也死锁); 用 Windows 原生 / 原生 Ubuntu+官方驱动
  - 驱动版本要匹配 Isaac Sim; 卡住看 .../isaacsim/kit/logs/.../kit_*.log (日志停增=死锁)
  - 首次启动会联网同步扩展注册表; 国内配代理/镜像
'''
print(guide)"""),
    md("""## 4. 把 N1 (DR) 和真 IsaacLab 连起来

- **N1 你做的**: GPU-free 玩具, 随机化「目标分布」演示 DR 覆盖→泛化。
- **真 IsaacLab 里**: 在 `--task` 的环境配置里随机化**摩擦/质量/光照/传感器噪声/电机强度**等 (Isaac 提供 DR 接口), 机制和 N1 完全一样 (覆盖真实的不确定性)。
- **sim 内 RL** (L4): rsl_rl 在并行环境里跑 PPO; 你 AntBot 跑的就是这个。

> 你已经实际跑过 AntBot (sim 内 RL) + 趟过配置坑。加上本模块的 DR/gap 认知, 你对 sim2real 是「会跑 + 懂原理」的完整掌握。"""),
    md("""## 5. 反思 (11.6 收口)

- 真 IsaacLab 需 NV GPU; 本课用 GPU-free 玩具 (N1) 讲清 DR 机制, 真跑指引 (本 notebook) 复用你 AntBot 经验。
- **配置是第一道坎**: WSL2 Vulkan 坑 → Windows 原生 (你的一手教训, `isaaclab_notes.md`)。
- DR 机制 sim/真一致: 随机化覆盖真实不确定性 → 泛化。

> **M11.6 收口**: 仿真便宜安全并行; sim2real gap=分布偏移; DR 用覆盖弥合; sim 内 RL 喂饱样本 (AntBot)。
> **交棒 M11.7「embodied-graduation」**: Module 11 capstone — 端到端 mini-VLA 装配 + 评测 (LIBERO/CALVIN 思路) + 研究 gap。下一专题 `embodied-graduation`。"""),
]
nbformat.write(n2, HERE / "N2-isaaclab-run-guide.ipynb")
print("written N2")
