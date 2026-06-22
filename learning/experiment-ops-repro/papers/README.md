# papers/ — experiment-ops-repro 参考源

本专题教**工程纪律**, papers/ 收可复现性的方法论与社区规范。

## 可复现性方法论 / 危机
- **Reproducibility in Machine Learning** — Joelle Pineau et al. (ML Reproducibility Checklist 的来源, NeurIPS 推行)。本专题 `repro_check.py` 的清单蓝本。
- **Deep RL that Matters** — Henderson et al. 2018 (种子/超参敏感、单次结果不可靠; 也见 9.4)。
- **Troubling Trends in ML Scholarship** — Lipton & Steinhardt 2018 (含复现/对照规范)。
- **Improving Reproducibility in ML Research** — Pineau et al. 2021 (NeurIPS reproducibility program 总结)。

## 工具文档 (照着用)
- Weights & Biases: https://docs.wandb.ai/  (业界标准实验追踪, 推荐学)
- Hydra (配置管理): https://hydra.cc/  (config as code 的工业标准)
- MLflow: https://mlflow.org/  · DVC (数据版本): https://dvc.org/
- NeurIPS Reproducibility Checklist / Paper Checklist (官网每年更新)

## 规范 / checklist
- ACM 的 Repeatable/Reproducible/Replicable 定义 (L1 三层级的来源)。
- 各顶会的 reproducibility checklist / artifact evaluation 说明。

## 为什么 papers/ 这么轻
本专题知识在**可跑的工具** (`exp_tracker` / `repro_check`) 和工程清单里, 不在论文 PDF。
最好的练习: 拿这套去审你自己某个 `learning/` 复现专题, 看它的可复现性体检能得几分。
