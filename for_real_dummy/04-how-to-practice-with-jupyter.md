# 用 Jupyter 开始练习 —— 跑代码、验证结果、留笔记三合一

> 前提：已看完 [01-numpy-for-c-programmers.md](01-numpy-for-c-programmers.md)、[02-pytorch-basics.md](02-pytorch-basics.md)
> 目标：知道怎么在 VSCode 里打开、运行、验证一个 notebook，并把它当成自己的练习笔记

---

## 0. 好消息：环境已经装好了

检查过你的 `.venv`(这个仓库自带的虚拟环境),下面这些全部已经安装好,**不需要你 `pip install` 任何东西**:

| 包 | 作用 |
|---|---|
| `jupyter` / `jupyterlab` / `ipykernel` | 让 notebook 能跑起来 |
| `numpy` 2.4.6 | 01 教程用的库 |
| `torch` 2.11.0(cu128) | 02 教程用的库,且带 GPU 支持 |

你只需要学会"怎么点",不需要配环境。

---

## 1. Jupyter 到底是什么

一个 `.ipynb` 文件由若干个 **cell(单元格)** 组成,两种类型:

- **code cell**:写 Python 代码,可以单独运行
- **markdown cell**:写文字说明(普通 Markdown,标题、列表、代码块都能用)

关键概念:**kernel**。

**C 语言类比:** 你写 C 的流程是"改代码 → 重新编译 → 重新运行整个程序",每次都从头开始,变量清零。
Jupyter 的 kernel 更像一个你 attach 上去、一直不退出的进程(类似开着不退出的 gdb 会话)——你一段一段地喂代码进去,它记得你之前定义过的所有变量。这就是为什么你可以先运行"定义一个函数"的 cell,再运行"调用这个函数"的 cell,而不用把所有代码写在同一个 cell 里。

**这也是新手最容易踩的坑:** cell 的运行顺序不是自动保证从上到下的——是你自己按的顺序。如果跳着运行、或者改完某个变量的定义忘了重新运行前面的 cell,kernel 里可能留着"幽灵变量",导致代码"能跑"但其实逻辑已经不对了。解决方法见第 4 节。

---

## 2. 在 VSCode 里具体怎么操作

### 2.1 打开一个 notebook

双击任意 `.ipynb` 文件(比如待会要用的 `practice/00-getting-started.ipynb`)。

如果打开后看到的是一堆 JSON 文本而不是一格一格的界面,说明 VSCode 还没装 **Jupyter 扩展**(微软官方)——去左侧扩展商店图标,搜索 "Jupyter" 装上,重新打开文件。

### 2.2 选择 kernel(重要,通常只需做一次)

notebook 界面右上角有一个 **"Select Kernel"**(选择内核)按钮。点开,选择:

```
Python Environments → .venv (Python 3.13.9)
```

如果列表里没看到 `.venv`,选 "Select Another Kernel" → "Python Environments" → 手动浏览到 `e:\Workspace\dummy\.venv\Scripts\python.exe`。

选错 kernel 最常见的后果:import numpy/torch 报"找不到模块"——遇到这个报错先检查 kernel 选对了没有。

### 2.3 运行一个 cell

鼠标悬停在 cell 左边会出现一个 ▷ 按钮,点它运行。或者选中 cell 后:

| 快捷键 | 效果 |
|---|---|
| `Shift + Enter` | 运行这个 cell,跳到下一个 |
| `Ctrl + Enter` | 运行这个 cell,留在原地 |

运行完,code cell 下面会直接显示输出(print 的内容、最后一行表达式的值、报错信息等)。

### 2.4 增加 / 删除 cell

鼠标移到两个 cell 之间会出现 "+ Code" / "+ Markdown" 按钮,点哪个加哪种。删除:点 cell 右边的垃圾桶图标。

---

## 3. 怎么验证结果是对是错(这是重点)

**眼睛看输出、觉得"好像对了",不算验证。** 真正的验证是:写出一个你确定成立的事实,用 `assert` 让程序自己检查。

### 3.1 基本模式

```python
result = 某个计算
assert result == 你手算/推理出的期望值, "不满足时打印的提示信息"
```

`assert` 后面的条件如果是 `False`,程序会直接报错并停下——这比"盯着数字发呆"可靠得多。

### 3.2 浮点数的陷阱:不能用 `==`

```python
0.1 + 0.2 == 0.3        # False!浮点数精度问题,不是 bug
```

比较浮点数要留一点误差空间:

```python
abs(a - b) < 1e-6                    # 单个数字
np.allclose(arr1, arr2)              # numpy 数组
torch.allclose(tensor1, tensor2)     # torch tensor
```

### 3.3 两类不同的验证目标

| 情况 | 怎么验证 | 例子 |
|---|---|---|
| 有确定的手算答案 | 直接 assert 等于那个值 | softmax 输出总和 = 1;autograd 算出的导数 = 手算的 5.0 |
| 没有固定答案(比如随机数据训练) | assert **趋势/性质**,而不是具体数字 | loss 应该下降;输出 shape 应该符合矩阵乘法规则 |

不知道该断言什么的时候,问自己:"如果这段代码写错了,会有什么现象?"——那个现象的反面就是你的 assert。

---

## 4. 怎么保证"从头跑一遍"也是对的

写了一堆 cell、乱序运行、中途改了某个变量——很容易出现"当前能跑,但从头跑一遍就炸"的情况,因为 kernel 里混进了"幽灵状态"。

**养成习惯:** 写完一个 notebook 后,点菜单栏的 **"Restart"**(重启 kernel)再 **"Run All"**(从头到尾依次运行所有 cell)。这样跑完所有 assert 都不报错,才算真的验证通过——等价于 C 里"改完代码重新完整编译一遍,而不是相信内存里的旧状态"。

---

## 5. 怎么把这些练习留存成你自己的笔记

建议的组织方式:

```
for_real_dummy/
└── practice/
    ├── 00-getting-started.ipynb   ← 现成的起步文件,见下一节
    ├── 01-numpy-basics.ipynb      ← 你自己建的,对应 01 教程练习
    ├── 02-pytorch-basics.ipynb    ← 对应 02 教程练习
    └── ...
```

每个 notebook 内部的写法:**markdown cell 写"我现在要验证什么/我的理解是什么",紧跟一个 code cell 写代码 + assert。** 这样几个月后回头看,你能看懂当时自己在想什么,而不是只有一堆没头没尾的代码。

这跟 [my-cheatsheet.md](my-cheatsheet.md) 是互补关系:cheatsheet 记"函数怎么用"的速查笔记,practice 笔记本记"我验证过这个概念/这段代码是对的"的过程记录。

---

## 6. 现在就试试

打开 [practice/00-getting-started.ipynb](practice/00-getting-started.ipynb):

1. 选 kernel(`.venv`)
2. 从上到下依次运行每个 cell(`Shift+Enter`)
3. 第一个 code cell 会打印 Python/numpy/torch 版本——能正常输出说明环境没问题
4. 后面有一个完整写好的验证范例(softmax),以及三个留空的练习(对应 02 教程练习 1/2/3),照着 TODO 注释填

---

## 7. 顺带一提:仓库里其他 notebook 长得不一样,正常

如果你点开比如 `learning/agent-foundations/notebooks/02-react.ipynb`,看到的是大段 markdown 理论 + 一个引用 `src/` 代码的空 cell——那是博士学长写的"课程讲义"风格,重点是理论笔记,代码在旁边的 `src/` 文件夹里。

你在 `for_real_dummy/practice/` 里写的 notebook 是另一种用途:代码为主、assert 为主,是你自己动手验证理解的地方。两种风格没有冲突,服务的目的不一样。

---

## 8. 遇到报错怎么办(常见情况)

| 报错/现象 | 大概率原因 |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | kernel 选错了,回第 2.2 节重选 `.venv` |
| 变量未定义,但明明写了 | 定义它的 cell 还没运行过,或者顺序反了——Restart + Run All |
| 输出一堆 tensor 元数据,很乱 | 忘了 `.item()`,见 [03 教程地雷 4](03-how-to-look-up-not-memorize.md) |
| Kernel 直接挂掉/无响应 | 点 "Restart Kernel",通常是某个 cell 死循环或者显存爆了 |

---

*更新:2026-07-07*
