# 9.2 literature-mapping — 文献综述 + 领域地图

> **Module 9「科研技能」· 阶段: 输入 (intake)**
> 教你在 **2 周内系统地摸清一个陌生子领域**: 找到奠基作、画出方法谱系、定位 SOTA 演进线和当前争论 —— 产出一张你自己的「领域地图」。

---

## 这个专题要解决的真问题

你进组第一周, 导师扔给你一个方向: 「去看看 preference optimization 这块」。然后呢?

- **新手做法**: arxiv 搜关键词, 从第一页开始一篇篇读, 读了 30 篇还是一团乱, 不知道谁重要、谁过时、现在做到哪了。两周过去, 一张图都画不出来。
- **熟手做法**: 先找 1-2 篇 survey 和 1-2 篇奠基作当锚点, **滚雪球**摸出网络骨架, 按方法族分流派, 排出演进时间线, 两周后能在白板上给导师画出整张领域地图 + 指出当前前线在哪。

> **核心区别: 摸领域不是「读得多」, 是「读得有结构」。** 文献不是一个列表 (list), 是一张网 (graph)。会读网的人, 用 1/5 的论文量得到 5 倍清晰的地图。

```
   新手: 线性扫           熟手: 读网络拓扑
   论文1 → 论文2 → ...     先找奠基作(网络中心) → 滚雪球(顺引用爬) → 分流派 → 排时间线
   读完仍是一团乱            两周后: 一张能上白板的领域地图
```

这张地图是后续一切的前提: 没有它, 9.3 找的 gap 可能早被人填了 (你不知道), 9.4 设计的实验可能没有对的 baseline (你不知道 SOTA 是谁)。**摸清领域 = 给你的研究装上 GPS。**

---

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-why-systematic.md` | 为什么要系统摸领域 + 三类综述 + 找四种入口锚点 | 子领域的 4 个锚点 |
| L2 | `lectures/L2-snowballing.md` | 滚雪球检索: 后向找奠基、前向找前沿 + 引用网拓扑读法 | 一张引用网 |
| L3 | `lectures/L3-building-the-map.md` | 从文献网到领域地图: 建 taxonomy + 排 SOTA 演进线 | 领域地图 (流派+时间线) |
| L4 | `lectures/L4-two-week-sop.md` | 2 周摸清子领域的 day-by-day SOP + mini-survey 产出 | 一份 mini-survey |

## 动手 (2 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-build-citation-graph.ipynb` | 用 `src/snowball.py` 给「偏好优化」子领域建引用网, 用 PageRank 中心度自动找出奠基作与前沿作, 画出引用网图 |
| `notebooks/N2-map-a-subfield.ipynb` | 用 `src/field_map.py` 把这张网整理成**领域地图**: 按流派分类、排演进时间线、出一张时间线图, 落成 markdown 地图 + mini-survey |

> 两个 notebook 用的「偏好优化 / DPO 家族」子领域, 正好对应你已有的 `learning/dpo-family` 复现 —— 你摸的是你已经动过手的领域, 体会最深。

## 可复用模板 (`templates/`)

- `subfield-map.md` — 领域地图模板 (流派 / SOTA 线 / 关键 benchmark / 争论 / 我的位置)
- `mini-survey.md` — 2 周 mini-survey 输出模板 (给导师看的那一页)

## 工具 (`src/`)

- `snowball.py` — 引用网建图 + 滚雪球 + PageRank 中心度找奠基/前沿 (networkx, 内置离线数据集)
- `field_map.py` — 把网整理成 taxonomy + timeline, 出 markdown 地图与时间线图

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native。两个 notebook 离线可跑 (内置子领域数据集)。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 拿到一个陌生子领域, 30 分钟内找到它的 survey + 奠基作 + 关键 benchmark
- [ ] 用滚雪球 (前向+后向) 摸出领域骨架, 而不是线性扫 arxiv
- [ ] 用引用中心度区分「必读奠基作」和「可跳过的边缘工作」
- [ ] 把文献整理成 taxonomy (流派) + timeline (演进线), 画成图
- [ ] 2 周内产出一份能上白板、给导师讲的 mini-survey + 领域地图

---

## 在 Module 9 中的位置

```
Module 9 科研技能
  地基   9.1 research-knowledge-mgmt   ✅ (地图/卡都落进它)
  输入   9.2 literature-mapping        ◄── 你在这里
        9.3 critical-reading-gap       ✅ (在地图前线上找 gap)
  执行   9.4 experiment-design          (地图告诉你 baseline 是谁)
  ...
```
> 9.2 → 9.3 是连续动作: 9.2 摸出**整片领域的地图**并定位「前线在哪」, 9.3 在那条前线上**对具体论文找 gap**。先有地图 (9.2), 才知道去哪找洞 (9.3)。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`
