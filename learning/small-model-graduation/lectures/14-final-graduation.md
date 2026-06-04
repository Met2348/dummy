# L14 · 最终毕业 — Module 3 收官

> 24 slides | 80 min ⭐⭐⭐⭐⭐⭐ (六星)

## Slide 1 · 仪式感

```
================================================
       Module 3「造大模型」毕业典礼
================================================

8 topic · 107 lectures · 118 方法 · ~ 122h

完成 capstone E = 毕业证书
tag: 造改-graduation
```

## Slide 2 · 你已学会

```
[v] 数据 pipeline (CommonCrawl → tokenized shard)
[v] transformer 现代架构 (RoPE/GQA/SwiGLU/MLA)
[v] MoE (GShard/Switch/Aux-Loss-Free)
[v] SSM (Mamba/RWKV/Jamba)
[v] 长 ctx (PI/NTK/YaRN/Ring)
[v] scaling laws + parallel (FSDP/Megatron)
[v] 推理 infra (vLLM/SGLang/量化)
[v] Phi-tiny 270M 从零训练
[v] 五部曲对照实验
```

## Slide 3 · 你能做什么

```
- 从 scratch 训出 270M Phi-tiny (单 5090 1d)
- 复刻 Llama-3 配方 (思路, 不是真训 405B)
- 调 ckpt 长 ctx 扩展 (8k → 32k)
- 评估 + benchmark
- 写完整 tech report
```

## Slide 4 · 五部曲 ckpt 集

```
ckpt_A: 124M vanilla
ckpt_B: 124M + Cosmopedia
ckpt_C: 270M Phi-tiny
ckpt_D: 270M + 8k YaRN
ckpt_E: 270M + curriculum  ⭐ Master ckpt
```

## Slide 5 · 报告结构

```
report/
  README.md            (一页概览)
  experiment_setup.md  (5 ckpt detail)
  curves/              (5 loss curves)
  benchmarks.csv       (主数据)
  benchmarks_radar.png (一图看 ablation)
  generations.md       (5 ckpt 同 prompt)
  ablation_table.md    (五部曲贡献)
  bag_of_tricks.md     (10 大 trick)
  reflections.md       (个人感想)
```

## Slide 6 · graduation_capstone.py 入口

```python
def graduate():
    # 1. 训 5 ckpt
    for v in ["A","B","C","D","E"]:
        train_variant(v)
    # 2. 评测
    results = run_all_benchmarks_per_ckpt()
    # 3. 报告
    write_report_md(results, "report/")
    # 4. 可视化
    save_all_plots(results)
    # 5. 生成对照
    save_generations_md()
    print("[GRADUATION] complete!")
```

## Slide 7 · Final tag

```bash
git add report/ src/graduation_capstone.py
git commit -m "graduation: Module 3 capstone complete"
git tag 造改-graduation
git tag 造改-graduation-final
```

## Slide 8 · 自检 checklist

```
[ ] 5 ckpt 全部训完
[ ] 6 metric × 5 = 30 数据点
[ ] 5 个 prompt × 5 ckpt = 25 生成样本
[ ] 4 张图 (curve / bar / radar / heatmap)
[ ] report.md 完整 (10 个 section)
[ ] git tag 造改-graduation 打上
[ ] README 自豪
```

## Slide 9 · 学习时间分配

```
T1 data:        ~ 12 h
T2 transformer: ~ 24 h (最重)
T3 MoE:         ~ 16 h
T4 SSM:         ~ 12 h
T5 long-ctx:    ~ 22 h
T6 scaling:     ~ 20 h
T7 recipe:      ~ 24 h
T8 graduation:  ~ 18 h (本)
                ≈ 148 h
```

## Slide 10 · 自己的感想 (留白)

```
What surprised you?
What was the hardest part?
What clicked for you?
What would you do differently?
What's next for you?
```

## Slide 11 · 致谢

```
- 原书《大模型算法》(电子工业 9787121500725)
- Llama-3 / DeepSeek-V3 / Phi tech reports
- nanoGPT (Karpathy)
- vLLM / SGLang teams
- Cosmopedia / FineWeb / DCLM datasets
```

## Slide 12 · 数字成就

```
118 methods learned
107 lectures attended
~150 hours invested
1 ckpt trained from scratch
5 ckpt ablation experiment
1 graduation report ✓
```

## Slide 13 · 与同行对比 (主观)

```
完成 Module 3 后:
  超过 90% 应届 ML 学生
  与 LLM 公司 junior 相当
  够当 LLM training engineer junior
  尚需 PEFT/RLHF (Module 4) 才能算 mid
```

## Slide 14 · 在 GitHub 展示

```markdown
## Pretraining from scratch
- Trained Phi-tiny 270M on Cosmopedia + web
- Implemented YaRN, RoPE, GQA, SwiGLU
- FSDP + WSD + bf16
- HellaSwag 0.50, MMLU 0.35, NIAH 80%
- [Full report](link)
```

简历亮点.

## Slide 15 · 给下一程的建议

```
Module 4 RL 系列:
  - 先把 RL 基础打牢 (CartPole)
  - 别跳 PPO 直接 GRPO
  - R1 复现要降级 (GPT-2 教学 + Qwen-1.5B 挑战)
  - DAPO 4 件套消融实验
  - 别忘了五线综合
```

## Slide 16 · 别犯的错

```
- 跳过 Topic 6 直接训大 (会 OOM)
- 跳过 Topic 5 训长 ctx (NIAH 0)
- 不写 report (3 月后忘光)
- 不打 tag (找不到 milestone)
- 完美主义 (报告 80% 就发)
```

## Slide 17 · 大方向 (2026+)

```
当前 (2026.06):
  - MoE 主流化 (DeepSeek-V3)
  - 长 ctx 普及 (32k → 128k)
  - 推理 RL (R1 类)
  - 商业 thinking model
  
未来 1-2 年趋势:
  - 1M+ ctx 标配
  - 多模态融合
  - Agentic LLM
  - 边缘端侧 ML
```

## Slide 18 · 自己的研究问题

```
留 1-2 个开放问题:
- 270M 模型为什么 emergence 差?
- WSD 真比 cosine 好吗?
- 长 ctx middle 怎么救?
- 小模型 + 高质数据极限在哪?
```

## Slide 19 · 总结

```
你完成了 LLM 训练从 0 到 1 全程
你训过 5 个 ckpt 对照实验
你写过 1 份 tech report
你打了 1 个 graduation tag
```

## Slide 20 · 一句话

```
"造大模型" = 数据 + 架构 + 训练 + 部署
不可怕, 跑通了就懂
```

## Slide 21 · 仪式结束

```
================================================
  congratulations
  你已是 LLM training engineer (junior)
================================================
```

## Slide 22 · 答辩问题 (可自答)

```
Q: 为什么 D 和 E 区别是什么?
A: D 是 C 续训 (后期加长 ctx)
   E 是从头训 (课程含长 ctx)
   benchmark 接近, E 路线工程更简

Q: ckpt B vs A 差 0.3 loss, 数据为啥?
A: Cosmopedia 平均 perplexity 更低
   分布更窄, 模型更易学

Q: 为啥不直接训 7B?
A: 5090 24G 训不动 7B
   需 8 卡 H100 80G 至少
```

## Slide 23 · 推荐继续

```
今天: 完成 Capstone E + 报告
明天: tag 造改-graduation
下周: Module 4 Topic 1 (RL 基础)
下月: Module 4 Topic 5 (R1 双轨)
```

## Slide 24 · 最后

```
祝你训练顺利
祝你 loss 平滑下降
祝你 grad_norm 不爆
祝你 ckpt 顺利保存

—— Module 3 系列结业
```

## 参考
- 系列内 Topic 1-7
- 本 capstone 报告
