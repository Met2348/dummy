# IsaacLab 实战 checklist (吸收本人踩坑经验)

> 把本人配置/运行 IsaacLab 的一手经验整理成可复用 checklist。结论先行: **WSL2 图形栈坑多, 用 Windows 原生 (或原生 Ubuntu + 官方驱动)。**

---

## 0. TL;DR (一句话教训)

- ❌ **WSL2 + NV driver 591 + Ubuntu 22.04 + IsaacLab 5.1.0 → 失败** (Vulkan 不支持, 连 headless 都死锁)。
- ✅ **Windows 原生 + conda `env_isaaclab` + Isaac Sim 5.1 → 成功跑通 AntBot (`Isaac-Ant-v0`)。**
- 教训: **IsaacLab 对图形栈 (Vulkan/驱动) 极挑剔; 环境配置是第一道坎, 是系统问题不是算法问题。**

---

## 1. 失败案例: WSL2 配置 (踩过的坑)

**配置**: win11 + WSL 2.6.3 + NV driver 591 + Ubuntu 22.04 + IsaacLab 5.1.0

**现象**:
- 不支持的 Vulkan 库 → 无法渲染窗口。
- 连 `--headless` (无头) 也无法渲染 → 程序**死锁**。
- 运行 `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless`:
  - PhysX 降级警告。
  - Vulkan API 出现严重错误, 阻止程序继续加载。
  - 日志大小不再增长 (卡死), 无训练进程。

**根因**: WSL2 的图形/Vulkan 栈对 Isaac Sim 支持不全 (即使无头渲染也依赖底层 Vulkan)。

**结论**: 此配置不可用 → 决策改用 **Windows 原生**。

---

## 2. 成功案例: Windows 原生 (推荐路线)

**配置**: Windows 11 原生 + conda env `env_isaaclab` + Isaac Sim 5.1 + IsaacLab

**验证最小启动** (先确认环境再训练):
```bat
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```
关键确认输出:
- `[INFO][AppLauncher]: Using device: cuda:0` ← 确认用上 GPU
- `[INFO][AppLauncher]: Loading experience file: ...isaaclab.python.kit`
- Isaac Sim 启动成功 (空场景)

**训练 AntBot**:
```bat
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Ant-v0 --headless
```
- `--headless` = 无渲染窗口 (纯算, 快, 训练用)
- 去掉 `--headless` = 开窗口可视化 (慢, debug 用)

---

## 3. 通用 checklist (让 IsaacLab 跑起来)

- [ ] **平台**: Windows 原生 / 原生 Ubuntu + 官方推荐驱动。避开 WSL2 (图形坑)。
- [ ] **驱动**: NVIDIA 驱动版本匹配 Isaac Sim 要求 (版本不对直接挂)。
- [ ] **Vulkan**: 确认 Vulkan 可用 (Isaac Sim 渲染/headless 都依赖它)。
- [ ] **conda 环境**: 专用 env (如 `env_isaaclab`), 别污染 base。
- [ ] **先验证最小启动**: 跑 `create_empty.py` 确认 `cuda:0`, 别一上来就训练。
- [ ] **headless ≠ 免图形**: 无头仍需可用的渲染栈 (踩过的坑)。
- [ ] **网络**: 首次启动会同步扩展注册表 (`syncing with extension registry`), 需联网; 国内可配代理/镜像。
- [ ] **日志**: 卡住时看 `.../isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log`, 日志停增长 = 死锁。

---

## 4. 配套环境经验 (来自 MDM 环境配置)

通用 Linux 环境配置经验 (可迁移):
- **换 apt 源 + 更新**: 确保软件包最新。
- **pip 镜像** (国内加速): `~/.pip/pip.conf`:
  ```
  [global]
  index-url = https://pypi.tuna.tsinghua.edu.cn/simple
  trusted-host = pypi.tuna.tsinghua.edu.cn
  ```
- **版本核对**: `cat /etc/os-release | grep -E "VERSION_ID|VERSION_CODENAME"` 确认系统版本无误。
- **git 代理** (拉取慢/墙): 配 git proxy。

---

## 5. 把踩坑变成认知 (接 L2)

- **环境配置是 sim 的第一道坎** —— 很多人 (包括我) 卡在这, 不是算法问题。
- **图形栈 (驱动/Vulkan) 是命门** —— Isaac Sim 重度依赖, 版本/平台不对全盘皆挂。
- **先小后大** —— 先验证最小启动 (cuda:0), 再上训练。
- 趟过这些坑 = 你的优势: 真做过 sim2real 的人, 配置经验是稀缺的。
