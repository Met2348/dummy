<!-- 复现 checklist (9.5-L4) · 投稿前必过的硬关卡. NeurIPS/ML reproducibility checklist 精简版. -->
# 复现 Checklist: ____________

> 投稿 / 开源前逐项勾选. 任一未勾, 写明原因或整改. repro_check.py 自动查前 6 项.

## A. 单 run 级 (repro_check.py 覆盖)
- [ ] 1. seed 已固定 (含 PYTHONHASHSEED), 代码里 seed_everything
- [ ] 2. git_sha 已记且 **非 -dirty** (正式结果来自已提交的代码版本)
- [ ] 3. config 完整, 无硬编码魔法数字
- [ ] 4. 环境指纹已记 / 附 lockfile (python + 包版本)
- [ ] 5. 数据集来源 + 版本 + 哈希已记
- [ ] 6. metrics 结构化记录 (非截图)

## B. 项目级
- [ ] 7. README 有明确「如何复现」段落
- [ ] 8. 有一键复现脚本 (scripts/reproduce_*.sh), 非手动跟论文跑
- [ ] 9. 关键结果有多种子 + error bar (接 9.4-L5)
- [ ] 10. 报告了算力 (GPU 型号 + 小时数), 让人估成本
- [ ] 11. 大文件/密钥已 .gitignore, 不进库
- [ ] 12. 不可公开数据有获取说明 + 可公开样例

## C. 诚实性 (成熟度的标志)
- [ ] 13. 已知不可复现点诚实写出 (如「GPU 非确定性致 ±0.3」)
- [ ] 14. limitations 段落不回避 (呼应 9.3 批判式阅读: 你希望别人怎么诚实待你, 就怎么待读者)

---
体检得分 (repro_check): ___/6 · 项目级补齐: ___/8 · 裁决: ____
