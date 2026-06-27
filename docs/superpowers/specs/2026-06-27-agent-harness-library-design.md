# Agent Harness 文献报告库 — 设计文档（spec）

- 日期：2026-06-27
- 分支：`ERIC-3080Ti/paper-guides`
- 状态：**已批准，建设中**
- 对标：已完成的 `learning/auto-research-frontier/` 74 篇报告库

## 1. 背景与目标
用户的老师（2026-06）在 auto-research 方向之外，提出**第二个新方向：Agent Harness**——
把 LLM 变成 agent 的软件层（控制循环、工具接口、上下文/记忆、评测、可靠性）。
要求**大规模文献调研**并建成同等深度的精读报告库。

## 2. 范围决定（已与用户确认）
| 维度 | 决定 |
|---|---|
| 范围边界 | **通用 harness 全景**（E/T/C/L/O/V 六层），非仅编码 |
| 规模 | **≥70**（实际 **74**，与 auto-research 对等） |
| 交付物 | **仅报告库**（老师要的"文献调研"）；可跑教学模块留作 phase 2 |
| 流水线 | **全复用** auto-research 已验证的工业化流水线 |
| canon 配比 | 约 30 篇 2022–2024 基石 + 约 44 篇 2025–2026 前沿（已告知用户，可调） |

## 3. 贯穿论点（全库一句话）
**Agent = Model + Harness**——能力/可信度有一大半压在 harness 上。
对应 auto-research 的「独立验证收口」。实证锚点：同模型换 scaffold，CORE-Agent 42%→78%、Cursor 46%→80%、
Vercel 砍工具 80%→100%；反方（regime 依赖）：METR/Scale 发现部分模型族 harness 选择在误差内。

## 4. 分类法（8 组 = E/T/C/L/O/V 映射，74 篇）
| 组 | 主题（层） | 篇数 | 锚点 |
|---|---|---:|---|
| A | 综述/框架与定义（跨层） | 8 | Natural-Language Agent Harnesses、ReCreate(φ+A)、Inside-the-Scaffold |
| B | 控制循环/推理-行动（L） | 10 | ReAct、Reflexion、ToT、LATS、General Modular Harness |
| C | 工具接口/ACI（T） | 8 | SWE-agent ACI、Toolformer、ToolLLM、Gorilla |
| D | 上下文工程/记忆（C） | 16 | MemGPT、Mem0、MEM1、A-MEM、IterResearch、AgentFold |
| E | 编码 Agent 集成系统 | 10 | OpenHands SDK、CodeAct、Agentless、SWE-Fixer |
| F | Web/计算机使用/GUI（E） | 7 | WebArena、OSWorld、UI-TARS、Mind2Web |
| G | Harness 评测/scaffold-aware（V/O） | 9 | Harness-Bench、SWE-bench、Terminal-Bench、τ-bench、GAIA |
| H | 可靠性/安全/可观测（O） | 6 | LlamaFirewall、AgentDojo、AgenTracer、容错沙箱 |

## 5. 目录与产物
```
learning/agent-harness-frontier/
├── papers/
│   ├── download_papers.py        # 74 篇，幂等可重下，%PDF 校验
│   └── INDEX.md                  # 清单/统计（收尾写）
└── paper-reports/
    ├── _STYLE-GUIDE-harness.md   # harness 专属增量 Θ1–Θ5（引用 v1+v2）
    ├── PROGRESS.md               # 建设账本（source of truth）
    ├── README.md                 # 链接索引（收尾写）
    ├── CATALOG-by-type.md        # 按类型情况简报（收尾写）
    └── <id>-<slug>.md × 74       # 报告本体
```

## 6. 规范（三层叠加）
v1 硬规范（公式前直觉+先定义符号、指标定义式、§/Table 出处、~20 页）
+ v2（Why 三连 + 强制 Inspires-Us）
+ harness 专属 Θ1–Θ5（E/T/C/L/O/V 分层、回扣 Agent=Model+Harness、Inspires-Us 打到自己 harness、canon/前沿坐标、regime 诚实）。

## 7. 执行计划
1. 基建：目录 + 下载脚本 + 规范 + 账本 + spec（本批）。
2. ID 核验：4 子代理并行核验 74 真实 arXiv ID，去重现有 74（**已完成**）。
3. 下载：跑脚本，核验 ≥70 PDF 落盘。
4. 标杆：亲写 1 篇 v2 harness 标杆（Harness-Bench）。
5. 报告：分批派子代理并行真读 PDF + 写报告，抽检 + 分批提交。
6. 收尾：README/CATALOG/INDEX/PROGRESS 同步 + 记忆更新。

## 8. 安全护栏（不可破）
- `learning/agent-foundations/lectures/02-react.md` 不碰/不暂存/不提交。
- 子进程 `--json-out/--md-out` 指 temp，不覆盖基线。
- PDF 走 `.gitignore`，只提交 `.md`。
- **不 push**，除非用户明说。

## 9. 风险与缓解
| 风险 | 缓解 |
|---|---|
| arXiv ID 幻觉 | 子代理逐一核验 + 下载器 `%PDF`+>10KB 校验（404 即 FAIL） |
| canon 比例偏高于"足够新"诉求 | 已透明告知用户；可砍基石补前沿 |
| API/socket 中断 | 产物落盘 + PROGRESS 账本 + 分批提交（auto-research 已验证抗中断） |
| 跨组重复 / 与旧库重复 | 建库前用 74 个旧 ID 做 diff；SWE-agent 跨组重复已归 C |
