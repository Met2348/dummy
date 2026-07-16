# WebShop-small 交互式 PK 实验记录（2026-07-14）

## 一句话结论

本机已经把 WebShop-small 从“静态商品选择投影”推进到**真实 text environment 交互实验**：在 1000 个 WebShop synthetic goals、1000 个 indexed products、9 个非 LLM baseline 上，`TRACE-H constraint+query ledger` 同时在两个指标上排名第一，并且对所有 baseline 都达到 paired sign test `p < 0.05`。这显著缓解了“29 个样本太少”的问题，但它仍然不是完整论文终局，因为当前真实数据源只有 ALFWorld 与 WebShop-small 两个，还缺第三个真实 benchmark。

## 本机环境修复

WebShop text env 原先不能启动，主要阻塞点已经修复：

- WSL 安装 JDK 21 后，`java` 与 `javac` 可用。
- 为新版 `pyserini` import 设置本地 dummy `OPENAI_API_KEY`，避免无关的 OpenAI encoder 初始化报错。
- 安装 `thefuzz` / `python-Levenshtein`。
- 安装 `en_core_web_sm==3.8.0`，供 WebShop reward 模块加载 spaCy。
- 将 `items_shuffle_1000.json`、`items_ins_v2_1000.json`、`items_human_ins.json` 同步到 WebShop 源码期望的 `data/` 目录。
- 使用 WebShop 自带 `convert_product_file_format.py` 和 `pyserini.index.lucene` 构建 `search_engine/indexes_1k`，成功索引 1000 个商品。

smoke 测试已经通过真实动作链：`reset -> search[...] -> click[asin] -> Buy Now`。

## 实验设置

实验脚本：

- `scripts/run_webshop_interactive_pk.py`

正式报告：

- `local-dev/reports/L4-webshop-interactive-pk-goals1000-20260714.json`

总 dashboard：

- `local-dev/reports/L5-local-paper-pk-dashboard-webshop-interactive-20260714.json`
- `local-dev/reports/L5-local-paper-pk-dashboard-webshop-interactive-table-20260714.tsv`
- `local-dev/figures/L5-local-paper-pk-webshop-interactive-primary-20260714.svg`
- `local-dev/figures/L5-local-paper-pk-webshop-interactive-secondary-20260714.svg`

每个 method 都真实执行 WebShop text env action：搜索、必要时翻页、点击商品、选择可匹配 option、购买。所有方法共享同一个 option selector，避免把选项点击能力混入主要 PK；主要差异集中在搜索 query 与候选商品选择机制。

## Baseline

本轮有 9 个 baseline：

- `random_top10_full_search`
- `bm25_full_instruction`
- `bm25_attribute_query`
- `bm25_core_query`
- `title_overlap_full_search`
- `description_overlap_full_search`
- `all_text_overlap_full_search`
- `attribute_overlap_attr_search`
- `rarest_attribute_anchor`

Ours：

- `traceh_constraint_query_ledger`

Ours 不使用目标 ASIN；它使用 instruction 中可解析的属性、option、价格与 query/category 线索，形成 constraint ledger 与 query ledger，再在 top-50 搜索候选中做商品选择。

## 主要结果

指标一：`mean_webshop_reward`。指标二：`exact_purchase_rate`。

| method | reward | exact |
|---|---:|---:|
| TRACE-H constraint+query ledger | 0.951 | 0.932 |
| all-text overlap after full search | 0.813 | 0.562 |
| attribute overlap after attr search | 0.787 | 0.610 |
| rarest-attribute anchor | 0.787 | 0.610 |
| BM25 full instruction | 0.784 | 0.551 |
| description overlap after full search | 0.667 | 0.322 |
| title overlap after full search | 0.423 | 0.106 |
| BM25 attribute query | 0.416 | 0.136 |
| random top-10 after full search | 0.376 | 0.090 |
| BM25 core query | 0.278 | 0.061 |

Ours vs 最强 reward baseline `all_text_overlap_full_search`：

- reward delta: `+0.138`
- paired wins/losses/ties: `394 / 44 / 562`
- p-value: `2.14e-71`
- exact delta: `+0.370`
- exact wins/losses: `405 / 35`
- p-value: `6.17e-81`

Ours vs 最强 exact baseline `attribute_overlap_attr_search` / `rarest_attribute_anchor`：

- reward delta: `+0.164`
- paired wins/losses/ties: `304 / 4 / 692`
- p-value: `1.43e-84`
- exact delta: `+0.322`
- exact wins/losses: `322 / 0`
- p-value: `2.34e-97`

## 解释

这组结果支持一个更强版本的机制 claim：在 WebShop 这类“搜索-选择-配置-购买”的长链任务中，单纯 BM25 或 all-text lexical overlap 很容易把 instruction 里的所有词混成一个软匹配目标；attribute-only 方法又会丢失商品类型与 query/category 约束。`TRACE-H constraint+query ledger` 的优势来自把 instruction 分成不同约束层：硬属性、商品类型/类别、option、价格、候选排序位置。它不是诊断性分析，而是一个实际可执行的 action selection 机制，最终体现在 WebShop reward 和 exact purchase 同时上升。

## 证据边界

这不是 LLM agent 端到端结果；本轮 policies 是 deterministic non-LLM local baselines，用于低成本机制迭代。它证明的是：在真实 WebShop text environment 和真实商品索引上，constraint/query ledger 机制相对一组强检索/重排 baseline 有稳定收益。后续论文级还需要：

- 换成 LLM agent 或至少 LLM parser + deterministic executor 的端到端版本；
- 增加第三个真实 benchmark；
- 做 sealed target split，避免所有机制选择都在同一批目标上迭代；
- 把静态 WebShop projection 降为补充材料，主文优先使用交互 WebShop。
