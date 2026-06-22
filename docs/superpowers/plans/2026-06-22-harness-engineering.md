# harness-engineering Implementation Plan ⭐ (生产工程 + 研究入口)

**Goal**: 新建 `learning/harness-engineering` 专题 —— `agent-harness-design` 的进阶续作, 把用户从「理解 harness 零件」推到「生产级 harness engineering + 研究前沿」。14 讲 + Capstone, 混合重心 (工程 + 研究桥)。

**Architecture**: 4 Part = 护城河/定义(L01-02) + 升级生产级(L03-08) + 成熟度三件套(L09-12) + 研究入口(L13-14)。Capstone 升级用户已有 mini-harness, 跑跨窗口长任务。运行默认 MockProvider (无 key 可跑), 真 provider 可选。

**Tech Stack**: Python 3.13 Windows native / stdlib + nbformat + pandas + matplotlib (复用 critical-reading-gap 环境)。无 GPU、无强制 API key。

**Design 文档**: `docs/superpowers/specs/2026-06-22-harness-engineering-design.md`
**复用**: `learning/agent-harness-design/src/harness` (概念骨架) + `learning/critical-reading-gap/templates/idea-card.md` (Part IV 产出)

---

## Phase 1: 骨架 + 环境 + README
- 1.1 `learning/harness-engineering/{papers,lectures,notebooks,src,templates,environment}/`
- 1.2 environment: requirements.txt + verify_env.py (utf-8 reconfigure)
- 1.3 README: 总览 14 讲 + 与已有专题边界 + 运行说明 + 研究桥

## Phase 2: Part I 课件 (护城河/定义)
- 2.1 L01-moat-and-commoditization.md (88%/65% 数据, 学科诞生)
- 2.2 L02-constitutive-definition.md (4 要素 inclusion test, 鉴定真实 harness)

## Phase 3: Part II 课件 (升级生产级)
- 3.1 L03-real-model-provider.md (provider 抽象/流式/tool 协议)
- 3.2 L04-progressive-compaction.md (5 阶段 + context folding + 落盘)
- 3.3 L05-long-horizon-autonomy.md (loop-with-hook + 文件系统状态)
- 3.4 L06-subagent-firewall.md (context firewall + debate)
- 3.5 L07-tool-mcp-control-plane.md (schema/MCP gateway/agentgateway)
- 3.6 L08-safety-and-control.md (权限门/destructive hooks/预算/RBAC)

## Phase 4: Part III 课件 (成熟度三件套)
- 4.1 L09-observability-otel.md (OTel + LLM span)
- 4.2 L10-evaluating-the-harness.md (同模型换 harness 分差, SWE-Bench-Pro)
- 4.3 L11-architecture-patterns.md (70 系统 5 模式 + 调度框架)
- 4.4 L12-portable-harness.md (harness 散落问题, NL harness 前沿)

## Phase 5: Part IV 课件 (研究入口)
- 5.1 L13-research-gaps.md (用 6 类 gap 雷达扫 harness 开放问题)
- 5.2 L14-capstone.md (升级 mini-harness 全流程说明)

## Phase 6: src 生产级组件 (stdlib, mock 默认, 可测)
- 6.1 `provider.py` — Provider 抽象 + MockProvider (确定性流式+tool-call) + 真 provider 接口
- 6.2 `compaction.py` — 5 阶段渐进式 compaction
- 6.3 `long_horizon.py` — loop-with-hook runner + filesystem state store
- 6.4 `otel_trace.py` — span/child-span 式 trace (reasoning/tool)
- 6.5 `harness_eval.py` — 跑同任务两配置, 出成功率/成本对照
- 6.6 `tests/test_all.py` — stdlib unittest 全绿

## Phase 7: notebooks (nbformat 生成 + nbconvert 跑通)
- 7.1 N1-compaction-in-action.ipynb (填满窗口看 5 阶段触发 + token 曲线)
- 7.2 N2-long-horizon-task.ipynb (loop-with-hook 跨 3 窗口完成一个长任务)
- 7.3 N3-harness-eval.ipynb (开/关 compaction 两 harness 成功率+成本对照, pandas+图)

## Phase 8: Capstone + 收尾
- 8.1 capstone: 串联 provider+compaction+long_horizon+otel+eval 跑通长任务
- 8.2 产出 2-3 张 harness 方向 idea 卡 (复用 idea-card 模板)
- 8.3 portfolio 更新 (Module 7 进阶 / 工程⨯研究交叉)
- 8.4 verify + commit

## 成功标准
- [ ] 14 讲齐全, 研究生级 (图/公式逐项/2026 实况+来源)
- [ ] src 可 import, test_all 全绿
- [ ] ≥3 notebook nbconvert 跑通
- [ ] Capstone 跨窗口长任务跑通 + idea 卡产出
- [ ] Part IV 真接上 critical-reading-gap (工程→研究桥)
