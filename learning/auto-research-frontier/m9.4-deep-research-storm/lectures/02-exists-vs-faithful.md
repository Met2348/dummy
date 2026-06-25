# L02 · 引用存在 ≠ 引用忠实（本模块核心）

## 1. 两种完全不同的"引用对不对"

- **存在性**：你引的 `[2504.08066]` 是不是一篇真论文？（9.2 守的那关，防的是凭空编一个 id。）
- **忠实度**：那篇 `[2504.08066]` 到底**有没有说**你这句话主张的事？

这两件事天差地别。一个引用可以**完美地存在、却完全不忠实**：
引了真论文，但那论文压根没说这个论断。这是真实综述 agent 最体面、最难抓的错误
（[OpenScholar](https://arxiv.org/abs/2411.14199) 专门量化"引用准确率"就是冲它去的）。

## 2. 本模块抓的两句

跑 `python src/run.py`，看这两句 `[!!]`：

```
[!! ] AI Scientist v2 的结果已经过湿实验验证 [2504.08066]
      → 引的 [2504.08066] 真实存在，但它并不支持 'wet-lab-validated'。
[!! ] STORM 实现了高引用准确率 [2402.14207]
      → 引的 [2402.14207] 真实存在，但 'citation-accuracy' 是 OpenScholar 的事。
```

第一句尤其阴险：**"湿实验验证"是真事**——只不过是 **co-scientist** 干的，不是 v2。
把一个领域里真实存在的成就，安到**另一篇**论文头上。读者若只想"v2 是不是真论文"，
答案是"是"，于是放行。

## 3. 忠实度怎么判：claim ⊆ 被引文献的事实账本

`faithfulness.py` 的核心一句：

```python
return "faithful" if sentence.claim_tokens <= doc.supports else "unfaithful"
```

每篇文档有个 `supports` 集合（它真支持的论断）；每句话有个 `claim_tokens`（它主张的事）。
**这句的主张必须被它引的那篇真包含。** v2 的 `supports` 里没有 `wet-lab-validated`，
所以"v2 经湿实验验证"判 unfaithful。`existence_only` 则只问 id 在不在库——被骗。

对照 `test_existence_check_is_fooled_by_unfaithful`：

```python
assert existence_only(s) is True            # 存在性：通过（被骗）
assert check_sentence(s) == "unfaithful"    # 忠实度：抓住
assert a["existence_pass"] == a["total"]    # naive 检查认为"全 5/5 合规"
```

## 4. 教训

> **核对"引用是否存在"是必要的，但远远不够。** 真正的核查要回到被引内容本身，
> 一句一句问"它真这么说了吗"。在你自己写论文时，这一关同样适用——
> 很多"引错"不是编了假文献，而是把对的事引到了错的出处。

## 5. 动手

1. 把不忠实句①的引用从 `2504.08066`（v2）改成 `2502.18864`（co-scientist），
   重跑——它还判 unfaithful 吗？为什么？（因为 co-scientist 的 `supports` 里**真有** wet-lab。）
   这说明 unfaithful 是"claim 和**这个**出处"的关系，不是 claim 本身对错。
2. 加一句"半真半假"的话（claim 一半被支持一半不被支持），看现在的"全包含"判它什么——
   这暴露了集合包含作为忠实度代理的局限（真实里要用 NLI/蕴含打分，见 L03）。
