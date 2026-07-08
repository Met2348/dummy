# Q&A — 怎么开始动手练习、验证结果、留笔记(2026-07-07)

## Q:如果自己开始练习,怎么在 IDE 当中开始做,且让代码可以跑起来,且让结果可以验证是正确或错误的,而且还能留存为自己的笔记?是 Jupyter 吗?

## Claude 的回答(要点)

1. **环境已经装好了,不用配置。** 仓库自带的 `.venv` 里已经有 jupyter/ipykernel/numpy/torch(2.11.0+cu128),不需要 `pip install` 任何东西,只需要学会怎么操作。

2. **是 Jupyter,核心概念是 cell + kernel。** kernel 类似于一个不退出的交互式进程(类比:一直挂着的 gdb 会话),cell 分 markdown(写笔记)和 code(写代码)两种,代码执行顺序由你自己控制,不是自动从上到下。

3. **验证结果对不对的核心方法是 `assert`,不是肉眼看输出。** 两种情况:有确定手算答案的直接断言等于那个值;没有固定答案的(比如随机训练)断言趋势/性质(loss 下降、shape 正确等)。浮点数比较要用 `np.allclose`/`abs(a-b)<1e-6`,不能用 `==`。

4. **notebook 本身就是笔记。** markdown cell 写"我在验证什么/我的理解",code cell 写代码+assert,两者穿插,就是持续增长的练习记录。建议目录:`practice/` 下一个练习一个 `.ipynb`。

5. **新建了一个可以直接打开就跑的起步文件** [practice/00-getting-started.ipynb](../practice/00-getting-started.ipynb):包含环境自检 cell、一个完整验证范例(softmax 的两条性质),以及对应 02 教程练习 1/2/3 的三个留空练习模板(TODO 注释 + 验证提示)。

## 详细内容

完整教程见 [04-how-to-practice-with-jupyter.md](../04-how-to-practice-with-jupyter.md),包括 VSCode 里选 kernel、运行快捷键、常见报错排查表等。
