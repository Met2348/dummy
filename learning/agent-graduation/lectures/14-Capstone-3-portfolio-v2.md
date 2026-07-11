# L14 · Capstone-3 — 39-Topic Portfolio v2 ⭐⭐⭐⭐⭐⭐⭐

## 整个 39 专题马拉松的"毕业作品"

不止 1 个 markdown，是**完整学习历程的 ID 卡**。

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"; python learning/agent-graduation/src/portfolio_v2.py
```

`__main__` 会先跑 `_self_test()`（写一个相对临时文件、断言内容后立即删除，不落 repo），再真实生成一份完整 Portfolio v2 并写到系统临时目录（`%TEMP%/agent_graduation_portfolio_v2.md`，同样不写入 repo），打印前 1500 字符预览 + 完整落盘路径。

若要在自己的脚本 / notebook 里生成到指定路径（例如真的要产出简历附件）：

```python
import sys
sys.path.insert(0, "learning/agent-graduation/src")
from portfolio_v2 import write_portfolio_v2

# 显式传一个你想要的路径；bare 相对路径（如 "portfolio_v2.md"）从 repo 根运行会直接
# 落在 repo 根，请勿在仓库工作区里裸调用——写到 repo 外部或系统临时目录更安全。
path = write_portfolio_v2("portfolio_v2.md")  # 或换成你自己的绝对路径
print(f"Portfolio v2 written to {path}")
```

## Portfolio v2 内容

```markdown
# 39-topic LLM Learning Portfolio (v2)

> 2026-06-05 · Module 7 收官，全系列马拉松完成 ⭐⭐⭐⭐⭐⭐⭐

## Section 1: Timeline
### Module 1 PEFT (3 专题)
   1-3. prompt / lora / adapter

### Module 3 造大模型 (8)
   4-11. data / transformer / moe / ssm / long-ctx / scaling / pretraining / small-grad

### Module 4 改大模型 (7)
   12-18. rl-found / rlhf / dpo / process / r1 / sota-2026 / multimodal

### Module 5 用大模型 (7)
   19-25. engine / sglang / spec / quant / distrib / prod / grad

### Module 6 评测/安全 (7)
   26-32. eval-found / reason / agent-code / judge / red-team / safety / grad

### Module 7 Agent 应用层 (7)
   33-39. agent-found / rag / tool-mcp / multi-agent / memory / framework / grad

## Section 2: 6-ckpt zoo + DRA
[7-row table: vanilla / lora / dpo / r1_tiny / phi_tiny / dra_v1]

## Section 3: All Capstones (6 + DRA + τ-bench)
- Module 3 small-model-grad (Phi-tiny 270M)
- Module 4 五线综合 (5 ckpt × 1 GSM8K 题)
- Module 5 serving-grad (mini vLLM serving)
- Module 6 mini-HELM / mini-Arena / 红队 / 防御 (4-axis)
- Module 7 DRA / τ-bench / Portfolio v2 ⭐⭐⭐⭐⭐⭐⭐

## Section 4: Selection trees
- Bench 选型 (Module 6)
- Inference engine 选型 (Module 5)
- RL 算法选型 (Module 4)
- **Agent framework 选型 (Module 7)** ⭐ NEW

## Section 5: 6 大画像
1. 造模型 — 从 0 训 GPT-2 / Phi-tiny
2. 改模型 — LoRA / Adapter / DPO / R1-Zero
3. 用模型 — vLLM / SGLang / 量化 / 分布式
4. 评模型 — 25 bench × judge × Arena
5. 守模型 — 红队 + 4 层防御 + Constitutional Cls
6. **造 agent 产品** — ReAct/RAG/MCP/multi-agent/memory ⭐ NEW

= 2026 LLM 全栈工程师 ID 卡 v2

## Section 6: What I can do (cover letter)
- "I can train Llama-7B / Phi-3 from scratch with curated data"
- "I can serve LLM at scale with vLLM / SGLang"
- "I can build deep research agents from scratch"
- ...

## Section 7: 5 Career paths
- LLM Infra Engineer ($250k-$500k)
- AI Application Engineer ($150k-$300k)
- ML Research Engineer ($300k-$1M)
- AI Safety Engineer ($200k-$500k)
- AI Product Manager ($150k-$400k)
```

## 整个 portfolio_v2 约 300 行 markdown，10 分钟读完。

## 用法

1. **简历附件**：直接 PDF
2. **GitHub README**：repo 主页 markdown
3. **面试讲解**：每 section 1 分钟，10 分钟总览
4. **LinkedIn post**：6 画像作 thread

## 退出条件

- [ ] portfolio_v2.md 文件生成
- [ ] 含 39 topic enumerated
- [ ] 含全部 capstone summary
- [ ] 含 4 selection tree
- [ ] 含 6 画像 + "What I can do"
- [ ] 含 5 career paths

## git tag 收官

```bash
git tag 应用-graduation
git tag module7-complete
git tag series-complete   # ⭐ 整个 39 专题马拉松完成
```

最后 3 个 tag 完结整个学习马拉松。

## 一句话

> Capstone-3 ⭐⭐⭐⭐⭐⭐⭐ = 把 39 专题写成 1 份 portfolio v2 — 2026 年 LLM 全栈工程师 ID 卡 v2。
