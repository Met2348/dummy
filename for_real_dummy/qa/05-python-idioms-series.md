# Q&A —— TensorFlow 100 项系列 + Pythonic 写法系列(2026-07-08)

## Q:接着是 tensorflow 类似的 100 个函数专题一口气做完,还要一些 pythonic 的写法专辑,包括但不限于 one-liner 等等

## Claude 的回答(要点)

1. **TensorFlow 系列暂缓,原因是环境问题,不是不做。** 排查发现 `.venv` 里 tensorflow/keras/jax 一个都没装,而且 TensorFlow 从 2.11 起官方砍掉了 Windows 原生 GPU 支持——要和现有 torch(cu128,GPU 可用)同等验证深度,必须走 WSL2(装 Linux 发行版 + NVIDIA WSL 驱动 + CUDA toolkit + 独立 venv)。这是一次独立的系统级环境搭建工程,和"一口气做完"内容写作不是一回事,已经和你确认过,单独排期,采用 torch-deep-dive 的面试深度格式。

2. **本轮先做 Pythonic 写法系列**,新增 [python-idioms/](../python-idioms/00-roadmap.md),是 [python-advanced/](../python-advanced/00-roadmap.md) 的姊妹篇——那边讲语言特性,这边讲表达习惯。4 个分类、26 个知识点:推导式与函数式基础(含收尾的 one-liner 取舍)、解包与迭代惯用法、容器与标准库惯用法(含 EAFP vs LBYL 哲学)、字符串与现代语法惯用法。

3. **分类设计经过了两轮探索 + 一轮 Plan agent 校验**,不是拍脑袋定的:确认了"pythonic/one-liner/itertools/推导式"这些关键词在现有内容里零重复;挖到了博士学长原仓库(`learning/`)里的真实代码素材(`bpe_trainer.py` 的 Counter/zip/字典推导、`mini_arena.py` 的 itertools.combinations、`inference-engine-core` 的 deque 调度队列等),优先用这些真实例子而不是编造玩具例子;体量预期从最初低估的"2000+行"修正为符合实测数据的"3500-4200行"。

4. **执行方式:4 个 agent 一对一分类并行撰写,每个都要求真实在 `.venv` 里独立验证。** 全部完成后,我又用一个独立脚本把全部 158 个代码块(44+53+32+29)逐个起子进程重新跑了一遍,不是只信 agent 自报——过程中 agent 自己就抓到了不少真实问题(手算多级排序 tie-break 顺序算错、`Counter.most_common()` 的插入序依赖坑、非连续代码引用误标成可执行块导致独立跑不通、`match-case` 编译期特殊报错发现),修复后全部通过。

5. **安全网:动手前先单独提交了一次 `for_real_dummy/`。** 发现这个目录(当时约240个知识点、17000+行)完全没有 git 版本历史,和你确认后先提交了一次快照,再开始批量生成。

## 详细内容

分类明细和每篇覆盖的知识点见 [python-idioms/00-roadmap.md](../python-idioms/00-roadmap.md)。
