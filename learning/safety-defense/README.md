# Topic 6: Safety Defense（安全防御 / Guardrails）

> Module 6「评」第 6 专题 · 12 lectures · 12 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 防御 4 层结构 | `common.py` |
| L02 | Llama Guard 3 (Meta) | `llama_guard_mock.py` |
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

## 跑测试

```powershell
python learning/safety-defense/src/tests/test_defense.py
```

预期：`9/9 modules passed`。

## 跑 capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/safety-defense/src'); from defense_pipeline import run_capstone, to_md; print(to_md(run_capstone()))"
```

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
