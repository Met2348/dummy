# Topic 6: Safety Defense（安全防御 / Guardrails）

> Module 6「评」第 6 专题 · 12 lectures · 12 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 防御 4 层结构 | `common.py` |
| L02 | Llama Guard 3 (Meta) | `llama_guard_mock.py`（+ `llama_guard_original_minimal.py` 论文算法形状变体） |
| L03 | ShieldGemma (Google) | (lecture only) |
| L04 | WildGuard (AI2) | `wildguard_mock.py` |
| L05 | NeMo Guardrails (NVIDIA) | `nemo_guardrails_mock.py` |
| L06 | Constitutional Classifiers (Anthropic 2025) ⭐ | `constitutional_classifier.py` |
| L07 | Prompt Injection 防御 | `prompt_injection_defense.py` |
| L08 | Content moderation 传统 | (lecture only) |
| L09 | PII 检测 + 脱敏 | `pii_redaction.py` |
| L10 | 监控 + 事件响应 | (lecture only) |
| L11 | 安全 bench | `safety_eval_runner.py` |
| L12 | **Capstone: 4-layer pipeline** | `defense_pipeline.py` |

## Tag

- `safety-defense` — Topic 6 完结

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V1 验证通过（**纯 stdlib，无需 torch/transformers，无需改代码**）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules safety-defense
> ```

9 个 lecture 直跑 demo（均无需传参，mock guard/rails 上跑分类-防御逻辑，CPU 秒级出结果）：

```powershell
python learning/safety-defense/src/common.py                          # L01 防御4层结构：harm关键词分类 + confusion matrix工具
python learning/safety-defense/src/llama_guard_mock.py                # L02 Llama Guard 3 mock：input/output 9类harm关键词分类器
python learning/safety-defense/src/llama_guard_original_minimal.py    # L02 Llama Guard 论文形状：O1-O6原始taxonomy + safe/unsafe\nOi输出格式
python learning/safety-defense/src/wildguard_mock.py                  # L04 WildGuard mock：关键词 + 对抗模式（ignore previous等）双通道检测
python learning/safety-defense/src/nemo_guardrails_mock.py            # L05 NeMo Guardrails mock：rule-based rails（refuse/redirect/sanitize/allow）
python learning/safety-defense/src/constitutional_classifier.py       # L06 Constitutional Classifiers mock：constitution规则 + 对抗覆盖检测
python learning/safety-defense/src/prompt_injection_defense.py        # L07 Prompt Injection防御：隐藏文本剥离 + 注入检测 + privilege-sandwich构造
python learning/safety-defense/src/pii_redaction.py                   # L09 PII检测+脱敏：email/phone/SSN/信用卡/IPv4正则 + toy姓名NER
python learning/safety-defense/src/safety_eval_runner.py              # L11 安全bench：任意guard函数的precision/recall/F1
```

**Capstone：4 层防御流水线（PII 脱敏 → NeMo rails → constitutional 分类 → 注入检测 → output guard），no_defense vs 4-layer 对照**：

```powershell
python learning/safety-defense/src/defense_pipeline.py
```

> 注（demo 性质，非 bug）：全部 10 个脚本无 argparse（runbook 内标 `v0: false`，跳过 `--help` 探针，直接 smoke 直跑到完成）；均为教学 mock（关键词/规则分类器），非真实微调的 Llama Guard / WildGuard / Constitutional Classifiers 权重。`llama_guard_original_minimal.py` 是 L02 的补充脚本：还原 2023 论文的原始 O1-O6 taxonomy + prompt/response 分任务 + `safe`/`unsafe\nOi` 输出格式 + 1-vs-all 逐类打分，而 `llama_guard_mock.py` 是 4 层 pipeline 里实际调用的 input/output 分类器版本——两者互补。Capstone 的 no_defense/4-layer 对照数字（precision/recall/F1 由 0.00 升到 1.00）由真实分类逻辑逐条推出（含一例 input 层漏检、output 层补漏的纵深防御演示），非硬编码。

**测试（V2）**：`test_defense.py` 聚合跑 10 个模块的 `_self_test()`：

```powershell
python learning/safety-defense/src/tests/test_defense.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules safety-defense --tests
```

预期：`10/10 modules passed`。（`test_defense.py` 无 pytest `test_*` 函数，`pytest` 会 collect-0 → 审计 harness 自动回退直接跑脚本。）

## 关键文献

- Inan et al. 2023 Llama Guard
- Meta 2024 Llama Guard 3
- Google 2024 ShieldGemma
- Han et al. 2024 WildGuard (AI2)
- NVIDIA NeMo Guardrails (2023)
- **Sharma et al. 2025 Constitutional Classifiers ⭐**
- Anthropic 2022 Constitutional AI
- OWASP LLM Top 10 (2024)

## 与 Topic 5/7 关系

```
Topic 5: 红队攻击
   ↓ ASR 高 → 需要防御
Topic 6 (本): 4 层防御
   ↓ ASR 大幅下降
Topic 7: 五线综合 mini-HELM 跑分（含防御 ASR 对照）
```

## 一句话

> 防御 = 4 层纵深，Constitutional Classifiers 2025 最强 — ASR 降 20×。
