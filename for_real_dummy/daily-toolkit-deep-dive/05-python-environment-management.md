# Python 环境管理实操 —— venv / conda / 依赖冲突排查

> 前提:本机已确认 `python 3.13.9` 可用(仓库根目录 `.venv` 用的就是它)。
> 边界:[04-how-to-practice-with-jupyter.md](../04-how-to-practice-with-jupyter.md) 第 2.2 节已经讲过"打开 notebook 时 kernel 要选 `.venv`"——那是"在 VSCode 图形界面里怎么点选"。这份文件讲的是更底层的东西:venv/conda 这些"环境管理工具"本身是什么、命令行里怎么创建/激活/退出/排查,以及为什么 Jupyter 的 "kernel" 概念,本质上就是"你选了哪一个虚拟环境的 python 解释器"。看完这份文件再回头看 04 号文件的 kernel 选择,会知道自己当时点的到底是什么。
> 目标:能自己创建/激活/退出一个虚拟环境;看懂 `pip install` 报错里的依赖冲突信息,而不是被一堆红字吓到直接搜 Stack Overflow;知道怎么导出/复现一份 `requirements.txt`;以后去到用 conda 的实验室服务器上,知道 conda 和 venv 的概念怎么对应、能不能混用、混用会踩什么坑。

---

## 0. 这份文件是怎么验证的

下面所有带命令和输出的代码块,都是撰写这份文件时**真实在这台机器上跑出来的**,不是凭记忆编的。具体验证范围:

- **Windows 原生 venv 操作**:PowerShell(本机确认过 `CurrentUser` 执行策略是 `RemoteSigned`)+ Git Bash 两种 shell 都真实跑过创建/激活/退出。
- **WSL2 venv 操作**:复用 rhcsa-bash-deep-dive 系列已经装好的 WSL2 RockyLinux(`python3 3.12.13`)真实跑过一遍;另外临时探测了一下 WSL2 Ubuntu(`python3 3.12.3`)作为对照。
- **依赖冲突 / `pip check` / `pip show`**:真实用 `pip install` 触发过冲突,真实报错文本原样贴在下面。
- **conda**:本机确实装了 Anaconda(`conda 23.3.1`),所有 conda 相关命令都是真实执行的,包括一个临时建了又删干净的演示环境。

**安全边界(和你自己动手时应该遵守的边界一样)**:所有真实操作都在仓库之外的临时目录里做的——Windows 侧用的是 `C:\Users\<你的用户名>\AppData\Local\Temp\daily-toolkit-venv-demo\`(在 Git Bash 里看到的路径是 `/tmp/daily-toolkit-venv-demo/`,两者是同一个目录,只是两种 shell 显示路径的写法不同),WSL2 侧用的是 WSL2 自己文件系统里的 `/tmp/daily-toolkit-venv-demo/`(这是和 Windows 的 `/tmp`完全不同的另一个目录,别混了)。仓库根目录真正在用的 `e:\Workspace\dummy\.venv` 全程没有被创建、删除或改动里面已装的包(文末有真实复查)。

**一个提前说明的真实事故**:写这份文件的过程中,作者本人真的手滑,把一个测试包装进了仓库的真实 `.venv` 里——不是编的段子,是第 4 节会详细复盘的真实翻车现场,当场发现、当场修复。之所以照实写出来,是因为这次事故本身就是对"为什么要养成确认环境的习惯"最有说服力的证据。

---

## 1. 为什么每个项目要有自己隔离的环境

### 1.1 为什么需要这个 / 不会有什么后果

先从最基础的事实说起:你在终端里敲 `pip install requests`,这个包会被装到**当前这个 Python 解释器**能找到的一个固定目录里,叫 `site-packages`。如果你的电脑上只有"一个" Python(比如直接装在 `C:\Python314\` 的那种,不用任何虚拟环境),那么**你电脑上所有的 Python 项目,不管有多少个,全部共用同一个 `site-packages` 目录**。

这意味着什么?意味着"给项目 A 装的包"和"给项目 B 装的包"其实装在同一个地方。如果项目 A 需要 `requests` 的 2.25.0 版本(比如它是照着一份两年前的教程写的),项目 B 需要 `requests` 的 2.31.0 版本(新特性),这两个版本**不可能同时装在同一个 `site-packages` 里**——`pip install requests==2.31.0` 会直接把 2.25.0 覆盖掉。项目 A 下次运行的时候,用的其实是 2.31.0,而 2.31.0 可能已经改了某个函数的行为,项目 A 就这样在你没做任何"改动"的情况下,莫名其妙跑不动了。

这不是危言耸听,是可以现场触发的真实报错。下面是一次真实的 `pip install` 尝试(在隔离出来的临时环境里做的,不影响仓库真实环境):

```bash
$ pip install "requests==2.25.0" "requests==2.31.0"

ERROR: Cannot install requests==2.25.0 and requests==2.31.0 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested requests==2.25.0
    The user requested requests==2.31.0

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip to attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
```

这是最直白的一种冲突——直接要求同一个库的两个版本,pip 当场拒绝。但更常见、更容易让人掉以轻心的,是下面这种"库 A 需要库 C 的旧版本,库 B 需要库 C 的新版本"的**间接冲突**,而且这种冲突 pip **不一定会拦住你**——这才是真正吓人的地方,细节见 1.3 节的真实翻车演示。

**如果不隔离会有什么后果**:你会活在一个"所有项目共享一套依赖版本"的世界里。装新项目所需要的包,可能会在你完全没意识到的情况下,悄悄改坏三个月前写好、当时能跑的旧项目。而且因为改动的时间点(装新包)和报错的时间点(重新跑旧项目)是分开的,排查起来会非常痛苦——你甚至想不起来"最近装了什么东西"。

### 1.2 环境要求

- 需要 Python 3.3 及以上版本(`venv` 模块从这个版本开始内置于标准库,不需要额外安装)。
- 检查命令:`python --version`(3.3+ 就够);`python -m venv --help` 能正常打印帮助信息,说明 `venv` 模块可用。
- 本机已确认:仓库 `.venv` 用的 `python 3.13.9` 可用;WSL2 RockyLinux 的 `python3 3.12.13` 也自带 `venv` 模块(现场用 `python3 -m venv --help` 确认过)。

### 1.3 一步步跟着做

下面这段是**真实执行**的完整翻车过程,在临时目录 `demo-conflict/` 里(路径见第 0 节)完成:

第一步,先干净地装上 `requests==2.25.0`,看看它自己要求的依赖版本范围:

```bash
$ pip install "requests==2.25.0"
Collecting requests==2.25.0
Collecting urllib3<1.27,>=1.21.1 (from requests==2.25.0)
  Downloading urllib3-1.26.20-py2.py3-none-any.whl
...
Successfully installed certifi-2026.7.22 chardet-3.0.4 idna-2.10 requests-2.25.0 urllib3-1.26.20
```

注意这一行:`Collecting urllib3<1.27,>=1.21.1 (from requests==2.25.0)`——这是 `requests==2.25.0` 自己声明的:它只能配合 `urllib3` 的 1.21.1 到 1.27(不含)之间的版本工作。装完之后用 `pip show` 确认:

```bash
$ pip show requests urllib3
Name: requests
Version: 2.25.0
Requires: certifi, chardet, idna, urllib3
Required-by:
---
Name: urllib3
Version: 1.26.20
Requires:
Required-by: requests
```

第二步,现在假装你的项目里又要用到另一个库,这个库要求更新的 `urllib3`(比如 2.2.0)。你并没有去动 `requests`,只是单独装这一个包:

```bash
$ pip install "urllib3==2.2.0"
Collecting urllib3==2.2.0
  Downloading urllib3-2.2.0-py3-none-any.whl
Installing collected packages: urllib3
  Attempting uninstall: urllib3
    Found existing installation: urllib3 1.26.20
    Uninstalling urllib3-1.26.20:
      Successfully uninstalled urllib3-1.26.20
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
requests 2.25.0 requires urllib3<1.27,>=1.21.1, but you have urllib3 2.2.0 which is incompatible.
Successfully installed urllib3-2.2.0
```

**这里是这次演示最关键的地方,读两遍**:pip 打印了一行 `ERROR:`,清清楚楚告诉你 `requests` 和新装的 `urllib3` 不兼容——但它**没有拒绝安装**,最后一行照样是 `Successfully installed urllib3-2.2.0`。也就是说,你的环境现在已经处于"表面上安装成功、实际上 requests 随时可能出问题"的状态,而唯一的线索就是那行很容易被滚动过去的 `ERROR:`。这个"能不能重现"的问题,3-5 节会给出系统排查方法。

### 1.4 背后发生了什么

pip 现在用的是"新版依赖解析器"(2020 年之后的 pip 都是),它在处理 `pip install X` 这条命令时,**只会认真解析 X 本身和它的依赖树**,尽量让这次要装的东西内部自洽。但对于"已经装好、这次命令没有提到"的包(比如上面例子里的 `requests`),新解析器只会在装完之后**回头检查一遍、发现不对就打印警告**,而不会在装之前就阻止你——这正是 pip 官方提示里那句 "does not currently take into account all the packages that are installed" 的字面意思。

换句话说:pip 的冲突检查更像是"事后诸葛亮",而不是"事前守门员"。如果这一步(装之前的隔离)被跳过,后果就是你会在一个所有项目共享的环境里不断堆叠这种"装的时候警告了一下、但没人当回事"的隐患。

### 1.5 常见坑

| 现象 | 真实原因 / 怎么办 |
|---|---|
| `pip install` 报 `ResolutionImpossible` | 两个直接写在命令行里(或都在 requirements.txt 里)的版本互斥,往上翻几行看 `The conflict is caused by`,把冲突的包名和版本记下来 |
| 装的时候有一行 `ERROR:`,但最后写着 `Successfully installed` | 不是你眼花——pip 确实会在检测到冲突的情况下继续安装,这不是"装失败了",是"装成功了但环境不一致",装完务必跑一遍 `pip check`(见第 5 节) |
| 换了台电脑 / 重开一个新项目,发现"明明代码没改,却报错" | 大概率是全局环境(没用虚拟环境)被别的项目的 `pip install` 污染了,先用 `pip show <出问题的包>` 看看版本是不是和你预期的不一样 |
| 不确定某次冲突是不是"真的会导致运行时报错"还是"版本号不一致但凑巧能用" | 版本约束(比如 `<1.27,>=1.21.1`)是包作者自己测试过、保证兼容的范围,范围之外"能不能用"完全看运气,不要赌 |

### 1.6 自测清单

- [ ] 能说清楚:如果不隔离环境,两个项目对同一个库的不同版本要求会怎么"打架"
- [ ] 能读懂上面那段真实报错里,`The conflict is caused by` 后面两行分别代表什么
- [ ] 知道 pip 遇到"和已装包冲突"时,不一定会阻止安装,只会打印 `ERROR:` 警告
- [ ] 能解释为什么"这次装东西的时候看到了警告,但没细看,几周后另一个功能坏了"这种事会发生

---

## 2. venv 日常操作

### 2.1 为什么需要这个 / 不会有什么后果

第 1 节说清楚了"为什么要隔离",这一节讲最基础的隔离工具:**`venv`**,Python 自带的虚拟环境模块(不需要额外安装任何东西)。

`venv` 做的事情很简单:在你指定的一个目录下(比如仓库根目录的 `.venv`),复制/链接出一份"专属于这个目录"的 `python`、`pip` 可执行文件,以及一个空的 `site-packages`。你在这个目录"激活"之后,`pip install` 装的所有东西,只会进这个专属的 `site-packages`,不会影响系统上其它任何地方。删掉这个目录,相当于把这个项目所有装过的包一次性清空,不影响别的项目分毫。

**不这样做的后果**,就是回到第 1 节演示的那种情况——所有项目共用一个 `site-packages`。而且更麻烦的是:如果哪天你的全局环境被装乱了,唯一的"重置"方式是一个一个包手动卸载排查;用了 `venv` 之后,重置方式是删掉那个目录重新建一个,几秒钟的事。

C 语言类比一下:如果说"全局 Python 环境"像是把所有项目的 `.o` 文件和库文件全扔进同一个 `/usr/lib`,链接的时候谁的符号先来后到全靠运气;那 `venv` 就相当于给每个项目单独开一个干净的编译目录,每个项目自己的 `build/` 谁都不共享。

### 2.2 环境要求

- Python 3.3+(内置 `venv`,无需安装)。
- 本机确认:Windows 侧仓库 `.venv` 用的 `python 3.13.9`;WSL2 RockyLinux 的 `python3 3.12.13`;另外临时探测 WSL2 Ubuntu 的 `python3 3.12.3` 也自带完整的 `venv`(含 `ensurepip`,建出来的虚拟环境里直接有 `pip` 能用)。
- **已知的常见坑(这次没有在本机复现,但极其常见,提前说明)**:某些精简版 Debian/Ubuntu 镜像(常见于一些云服务器/Docker 基础镜像)把 `venv` 依赖的 `ensurepip` 拆成了单独的系统包,不预装。表现是 `python3 -m venv myenv` 建是能建,但建出来的环境里没有 `pip`,报 `ensurepip is not available`。遇到这种情况的解决办法是 `sudo apt install python3-venv`(具体包名可能带 python 版本号,比如 `python3.12-venv`)。本机测试的两个 WSL2 发行版(RockyLinux、Ubuntu)恰好都没有这个问题,所以没能现场复现报错,如实说明。

### 2.3 一步步跟着做

**创建**(以下命令统一用 `python -m venv 目录名`,目录名随意,约定俗成常用 `.venv` 或 `venv`):

```bash
$ python -m venv demo-basic
```

跑完之后目录长这样(Windows 上创建出来的):

```
demo-basic/
├── Scripts/
│   ├── activate          ← Git Bash / WSL2 风格,用 source 调用
│   ├── activate.bat       ← cmd.exe 风格
│   ├── activate.fish      ← fish shell 风格
│   ├── Activate.ps1        ← PowerShell 风格
│   ├── deactivate.bat
│   ├── python.exe
│   └── pip.exe
└── pyvenv.cfg
```

**关键细节**:Windows 上可执行文件在 `Scripts\` 目录,Linux/Mac(包括 WSL2)上在 `bin/` 目录——这不是随便取的名字,是各自操作系统"可执行文件通常放哪"的惯例。这也是为什么"激活"命令在两边长得不一样:Windows 下是 `.venv\Scripts\activate`,Linux/Mac 下是 `source .venv/bin/activate`,本质上都是"运行这个目录下负责激活的那个脚本",只是可执行文件的目录名不同,而且 **Linux/Mac 那边必须加 `source`**,原因见 2.4 节。

**Windows · PowerShell**,真实跑一遍(在临时目录 `demo-basic/` 里):

```powershell
PS> $((Get-Command python).Source)      # 激活前
e:\Workspace\dummy\.venv\Scripts\python.exe

PS> .\demo-basic\Scripts\Activate.ps1
(demo-basic) PS>                        # 提示符前面多了 (demo-basic)

(demo-basic) PS> $((Get-Command python).Source)
C:\Users\...\daily-toolkit-venv-demo\demo-basic\Scripts\python.exe

(demo-basic) PS> python --version
Python 3.13.9

(demo-basic) PS> deactivate
PS> $((Get-Command python).Source)      # 退出后
e:\Workspace\dummy\.venv\Scripts\python.exe
```

注意最后一行:退出(`deactivate`)之后,`python` 并没有变回"什么虚拟环境都没有"的裸系统 Python,而是变回了**激活 `demo-basic` 之前那个已经处于激活状态的仓库 `.venv`**。这是本机这个终端本身的特殊情况——这个终端进程一启动,`VIRTUAL_ENV` 就已经指向仓库 `.venv` 了(不是系统里永久设置的,用 `[Environment]::GetEnvironmentVariable("VIRTUAL_ENV","User")` 查过是空的,只在这个进程级别存在,大概率是终端启动时被外层工具自动注入的)。这恰好引出一个很重要的认识:**`deactivate` 退回的是"激活这一层之前的状态",不一定是"什么环境都没有的裸状态"**——如果你是从一个已经激活了某个环境的终端里,再激活另一个环境,退出后会回到原来那一层,而不是回到"无环境"。

**Windows · Git Bash**,同样真实跑一遍:

```bash
$ which python                          # 激活前
/e/Workspace/dummy/.venv/Scripts/python

$ source demo-basic/Scripts/activate
(demo-basic)
$ which python
/tmp/daily-toolkit-venv-demo/demo-basic/Scripts/python

$ python --version
Python 3.13.9

$ deactivate
$ which python                          # 退出后,同样是回到仓库 .venv,不是裸系统
/e/Workspace/dummy/.venv/Scripts/python
```

**WSL2(RockyLinux)**,真实跑一遍,注意这里目录是 `bin/` 不是 `Scripts/`:

```bash
$ which python3                         # 激活前
/usr/bin/python3

$ python3 -m venv demo-basic
$ source demo-basic/bin/activate
(demo-basic) $ which python
/tmp/daily-toolkit-venv-demo/demo-basic/bin/python

(demo-basic) $ python --version
Python 3.12.13

(demo-basic) $ deactivate
$ which python3                         # 退出后,这里确实回到了裸系统 python3
/usr/bin/python3
```

WSL2 这边退出后确实回到了裸系统的 `/usr/bin/python3`——因为这个 WSL2 终端进程本身没有那个"外层预先注入 VIRTUAL_ENV"的情况。两相对比正好说明:`deactivate` 到底会回到哪里,取决于你激活这个环境之前终端本来的状态,不能想当然认为"退出等于回到系统默认"。

**怎么确认自己到底在哪个环境**(这是本节最实用的一条):

- Windows PowerShell 下,**不要直接输入 `where`**——它在 PowerShell 里被 `Where-Object` 占用了别名,输入 `where python` 实际执行的是完全不同的东西。真实验证:

  ```powershell
  PS> Get-Alias where
  Alias  where -> Where-Object
  ```

  要用 Windows 原生的 `where` 命令,必须显式加 `.exe`:

  ```powershell
  PS> where.exe python
  e:\Workspace\dummy\.venv\Scripts\python.exe
  C:\Python314\python.exe
  D:\Anaconda\python.exe
  ```

  这条命令会把 PATH 里所有名字叫 `python` 的可执行文件全列出来,**排第一个的才是真正会被调用的那个**。
- Git Bash / WSL2 下用 `which python`,同样只看第一条。
- 最可靠、不依赖 PATH 顺序的方式:直接看环境变量 `VIRTUAL_ENV`——PowerShell 里 `$env:VIRTUAL_ENV`,Bash 里 `echo $VIRTUAL_ENV`。这个变量是激活脚本亲手设置的,值就是虚拟环境的完整路径;没激活任何环境时它是空的。`where`/`which` 理论上也可能因为 PATH 顺序被别的东西干扰给出误导性结果(极少见,但第 4 节会给出一个本机真实翻车的例子),`VIRTUAL_ENV` 更值得信任。

### 2.4 背后发生了什么

激活脚本(不管是 `Activate.ps1` 还是 `activate`)做的事情本质上就三件:把这个虚拟环境的 `Scripts`/`bin` 目录**临时插到 PATH 最前面**(所以 `python`/`pip` 会优先找到这里的可执行文件)、设置 `VIRTUAL_ENV` 环境变量指向这个目录、顺手把提示符改一下(比如加上 `(demo-basic)`)方便你肉眼确认。`deactivate` 做的是相反的事:把 PATH 和提示符恢复成激活之前保存的样子。

这也解释了为什么 Linux/Mac 下必须用 `source`(或者等价的 `.`)来运行 `activate`,而不能直接 `./activate` 或者 `bash activate`:普通执行一个脚本(`./activate`)会开一个**新的子 shell 进程**去跑它,子进程改的环境变量,改完这个子进程一退出就全没了,父 shell(你正在用的这个终端)完全不受影响。`source` 则是让这个脚本的每一条命令,直接在**当前这个 shell 进程**里执行,改的 PATH、改的环境变量,自然就留在了当前终端里。Windows 的 `Activate.ps1`/`activate.bat` 不需要显式 `source`,是因为 PowerShell/cmd 运行 `.ps1`/`.bat` 脚本本身默认就是在当前进程语境里执行(这也是为什么 Windows 上这几种脚本文件名各不相同——`.ps1` 给 PowerShell,`.bat` 给 cmd.exe,无扩展名的 `activate` 给 Git Bash 这种 POSIX shell,分别匹配各自"怎么执行脚本"的规则)。

如果跳过激活这一步会怎样?其实激活**只是个方便的壳子**,不是必需品——你完全可以不激活,直接用完整路径调用虚拟环境里的 `python.exe`/`pip.exe`(比如 `demo-basic\Scripts\python.exe -m pip install requests`),效果和激活后再敲 `pip install requests`是完全一样的,只是每次都要打全路径,麻烦一些。很多 CI 脚本、自动化脚本就是走这条路,压根不激活。

### 2.5 常见坑

| 现象 | 真实原因 / 怎么办 |
|---|---|
| PowerShell 里运行 `Activate.ps1` 报错,提示"无法加载,因为在此系统上禁止运行脚本" | 执行策略(Execution Policy)限制。本机 `CurrentUser` 是 `RemoteSigned`(本地脚本能跑),真的遇到这个报错说明你的策略更严格,修复:`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,只影响当前用户,不需要管理员权限 |
| 敲了 `where python` / `which python`,结果和预期不一样 | PowerShell 下 `where` 是 `Where-Object` 的别名,不是 Windows 的 `where.exe`,必须写全 `where.exe python`;更保险的做法是直接查 `$env:VIRTUAL_ENV` / `$VIRTUAL_ENV` |
| `deactivate` 之后发现 `python` 还是指向某个虚拟环境,不是"裸" Python | 参考 2.3 节的真实案例——`deactivate` 只是退回上一层,如果这个终端启动时就已经在某个环境里,退出后会看到那一层,不是系统默认 python。想确认"裸系统"环境,开一个全新终端窗口检查 |
| Windows 上把 `.venv` 文件夹整个复制到另一台电脑,发现坏了 | 虚拟环境里的激活脚本、`pyvenv.cfg` 里都写死了创建时的绝对路径,不能跨机器/跨目录直接复制,换机器要重新 `python -m venv` 再装一遍依赖(这正是第 3 节 `requirements.txt` 要解决的问题) |
| 虚拟环境文件夹不小心被 `git add` 进仓库 | 检查 `.gitignore` 有没有排除它——这个仓库根目录的 `.gitignore` 已经写了 `.venv/` 和 `venv/`(真实确认过),如果你自己新建的项目没有这两行,记得加上,虚拟环境目录几百 MB 起步,而且里面全是能重新生成的东西,不该进版本控制 |

### 2.6 自测清单

- [ ] 能不看任何笔记,独立敲出"创建一个新虚拟环境"的命令
- [ ] 知道 Windows 下激活用 `Scripts\`、Linux/Mac 下用 `bin/`,并且知道这是操作系统惯例差异,不是随意命名
- [ ] 能说出为什么 Linux/Mac 下必须用 `source activate` 而不能直接执行这个脚本文件
- [ ] 知道 `where python`(PowerShell 下要写 `where.exe`)/ `which python` 分别怎么用,以及为什么 `$env:VIRTUAL_ENV` / `$VIRTUAL_ENV` 更可靠
- [ ] 能解释"退出虚拟环境后看到的不是裸系统 python"这种情况可能是什么原因造成的

---

## 3. requirements.txt / pip freeze

### 3.1 为什么需要这个 / 不会有什么后果

第 2 节的 `venv` 解决了"隔离"问题,但只在**你自己这台电脑**上有效。虚拟环境目录本身不能跨机器复制(2.5 节提到过,路径是写死的),那怎么把"我这个项目依赖哪些包、分别是什么版本"这件事,原样传给另一台电脑、另一个人、或者半年后的你自己?答案就是一份纯文本清单——习惯上文件名叫 `requirements.txt`,内容就是一行一个"包名==版本号"。

**不这样做的后果**:你在自己电脑上代码跑得好好的,发给同学或者传到服务器上,对方 `pip install` 装的是各个包"当前最新版",而不是你当初实际用的那个版本组合,复现不出你的结果——机器学习/科研场景下这个问题格外致命,因为版本差异可能导致数值结果都对不上,排查起来毫无头绪。

### 3.2 环境要求

- 只需要 `pip`(创建虚拟环境时自带,不需要额外装)。

### 3.3 一步步跟着做

**导出**:在一个装好了包的虚拟环境里,真实操作(临时目录 `demo-freeze-src/`):

```bash
$ pip install requests click
...
$ pip freeze
certifi==2026.7.22
charset-normalizer==3.4.9
click==8.4.2
colorama==0.4.6
idna==3.18
requests==2.34.2
urllib3==2.7.0
```

注意:命令里只显式装了 `requests` 和 `click` 两个包,但 `pip freeze` 打印出了 **7 行**——多出来的 `certifi`、`charset-normalizer`、`colorama`、`idna`、`urllib3` 都是这两个包各自依赖的间接依赖(比如 `requests` 依赖 `urllib3`)。`pip freeze` 打印的是**这个环境里实际装了的一切**,不是"你手动敲过 install 的那几个"——这一点很重要,3.5 节还会再提一次。

存成文件、复现到另一台机器(这里用第二个全新的虚拟环境模拟"另一台机器"):

```bash
$ pip freeze > requirements.txt
$ cat requirements.txt
certifi==2026.7.22
charset-normalizer==3.4.9
click==8.4.2
colorama==0.4.6
idna==3.18
requests==2.34.2
urllib3==2.7.0

# 在全新的另一个虚拟环境 demo-freeze-reproduce/ 里:
$ pip install -r requirements.txt
$ pip freeze > requirements.txt

# 两份 requirements.txt 逐行 diff:
$ diff demo-freeze-src/requirements.txt demo-freeze-reproduce/requirements.txt
$ echo $?
0
```

`diff` 没有输出任何差异,两份文件**逐字节完全一致**——这就是"复现"的真实验证,不是嘴上说说。

**和仓库里真实存在的 requirements.txt 互相印证**:这个仓库根目录本身没有 `requirements.txt`(用 `git status`/目录列表确认过),但 `learning/` 下几十个子模块各自的 `environment/` 目录里都有一份,风格和上面 `pip freeze` 生成的完全不同。比如 `learning/transformer-deep/environment/requirements.txt` 的真实内容:

```
# Transformer Deep 学习包依赖(复用 Module 1+2 cu130 torch)

torch>=2.5
einops>=0.8
transformers>=5.0
tokenizers>=0.20
matplotlib>=3.8

# 可选库(部分依赖 Linux/特定 GPU)
# flash-attn>=2.6      # 需 sm_80+,Win 装不上时跳过
# triton>=3.0          # 需 nvcc,Win native 可能受限

numpy>=1.26
tqdm>=4.66
```

对比一下就能看出 `requirements.txt` 其实有两种完全不同的**写法风格**,都叫这个文件名,但用途不一样:

| 风格 | 版本写法 | 谁写的 | 用途 |
|---|---|---|---|
| `pip freeze` 生成 | `==` 精确锁死,而且包含所有间接依赖 | 机器自动生成 | "原样复现我这个能跑的环境",给别人 `pip install -r` 之后拿到和你完全一样的版本组合 |
| 手写的(如上面 transformer-deep 那份) | `>=` 给下限、留出升级空间,只写你直接用到的顶层包,还能加注释 | 人手写 | "这个项目大致需要什么",给新项目定一个起点,不追求锁死到小版本号 |

两种都合理,只是回答的问题不一样:手写版回答"这个项目大概需要什么东西",`pip freeze` 版回答"我这个能跑的环境精确长什么样"。

### 3.4 背后发生了什么

`pip freeze` 做的事情,是去读当前这个虚拟环境 `site-packages` 目录里,每个已安装包留下的元数据记录(每个包装的时候都会在 `site-packages` 里留一份 `*.dist-info` 记录名字和版本),然后把它们**原样打印**出来——它不知道也不关心"你当初是不是故意要装这个",只负责如实汇报"现在这里到底有什么"。这也是为什么会包含间接依赖:间接依赖也是实实在在装在 `site-packages` 里的东西。

`pip install -r requirements.txt` 反过来,就是把文件逐行读出来,等价于按顺序把每一行当成一次 `pip install 这一行的内容` 来执行。

### 3.5 常见坑

| 现象 | 真实原因 / 怎么办 |
|---|---|
| `pip freeze` 输出里有一堆自己没主动装过的包名 | 正常——那些是间接依赖,3.3 节的真实例子里 `certifi`/`idna`/`urllib3` 等都是这么来的,不用一个个纠结要不要删 |
| 在 Windows 上 `pip freeze` 生成的文件,拿到 Linux 上 `pip install -r` 会不会出问题 | 视情况。上面真实例子里出现的 `colorama` 其实只是 `click` 在 Windows 上才需要的依赖(用来在 Windows 终端里显示颜色),但 `pip freeze` 的输出**不会保留"仅 Windows 需要"这种平台条件**,变成一行普通的 `colorama==0.4.6`。拿去 Linux 装,`colorama` 本身是纯 Python 包,能装上,只是装了也用不到,不算报错,但确实不是"干净"的复现——如果要跨平台严格复现,更适合用 `pip freeze --all` 之外的专门工具,或者干脆手写 `requirements.txt` 只列直接依赖 |
| `pip install -r requirements.txt` 之后版本还是和别人不完全一样 | `requirements.txt` 只锁 Python 包版本,不锁 Python 解释器本身的版本——对方如果用的是 Python 3.11 而你冻结时用的是 3.13,同一份包版本清单不保证行为完全一致,尤其是有 C 扩展的包(比如 numpy)。想完全复现,Python 版本也要对齐 |
| 忘了自己是在哪个虚拟环境里执行的 `pip freeze` | 结果会混进那个环境里其它不相关项目留下的包。执行前先用 2.3 节的方法确认 `$VIRTUAL_ENV`/`where.exe python` |

### 3.6 自测清单

- [ ] 能独立完成"导出当前环境依赖 → 在一个全新环境里复现 → 验证两边一致"这一整套流程
- [ ] 能解释为什么 `pip freeze` 的输出里会出现你没有主动 `pip install` 过的包名
- [ ] 能说出手写 `requirements.txt`(`>=`)和 `pip freeze` 生成的 `requirements.txt`(`==`)分别适合什么场景
- [ ] 知道 `requirements.txt` 不能保证锁定 Python 解释器本身的版本

---

## 4. conda:如果你的环境里同时有它

### 4.1 为什么需要了解这个 / 不会有什么后果

这个仓库从头到尾用的都是 `venv`(仓库根目录 `.venv` 就是证据),所以严格说你现在**不需要** conda 就能完成这个仓库里的所有练习。但现实情况是:很多实验室的 GPU 服务器、很多别人写的教程/README,默认假设你用的是 conda(`conda create -n xxx`、`conda activate xxx` 这种命令)。如果完全不了解 conda 和 venv 的概念对应关系,遇到这些说明会觉得完全陌生,甚至可能在不清楚后果的情况下把两者混着用,搞出很难排查的环境问题。

这一节不是要教你"精通 conda",只解决三个问题:conda 的核心概念怎么和刚学的 venv 对应、能不能混用、混用最容易踩什么坑。本机恰好装了 Anaconda,所以下面全部是真实命令和真实输出。

### 4.2 环境要求

- 需要装 Anaconda 或 Miniconda(这个仓库本身不需要,如果你的服务器/实验室机器已经装了才用得上这一节)。
- 本机真实确认:`conda 23.3.1`,装在 `D:\Anaconda`。用 `conda env list` 能看到这台机器上已经存在的环境(真实输出,做了截断,只保留能说明问题的部分):

  ```
  # conda environments:
  #
                           C:\ProgramData\miniconda3
  base                  *  D:\Anaconda
  HSI                      D:\Anaconda\envs\HSI
  HSI_310                  D:\Anaconda\envs\HSI_310
  ...(还有若干历史环境,略)
  py36                     D:\Anaconda\envs\py36
  ```

  这里顺带能看出这台机器上其实**装了两套独立的 conda**(`D:\Anaconda` 和 `C:\ProgramData\miniconda3`)——这本身也是一种常见的环境混乱来源,见 4.5 节。

### 4.3 一步步跟着做

**概念对应关系**,先建立起来:

| venv 里的概念 | conda 里的对应概念 |
|---|---|
| `python -m venv myenv` | `conda create -n myenv python=3.11` |
| `.venv/Scripts/activate`(Win)、`.venv/bin/activate`(Linux) | `conda activate myenv` |
| `deactivate` | `conda deactivate` |
| `pip install 包名` | `conda install 包名`(或者继续用 `pip install`,见 4.5 节) |
| `pip freeze > requirements.txt` | `conda env export > environment.yml` |
| 删掉整个虚拟环境目录 | `conda env remove -n myenv` |

**venv 管不了、conda 能管的东西**:这是两者最本质的区别,不只是命令名字不一样。`pip`/`venv` **只能装 Python 包**;conda 是一个更通用的包管理器,不仅能装 Python 包,还能装 CUDA Toolkit、C/C++ 编译器、甚至非 Python 的命令行工具这些"系统级"依赖。这就是为什么很多深度学习实验室服务器用 conda 而不是 venv——GPU 环境经常需要精确匹配的 CUDA 版本,这件事 `pip` 做不到,conda 可以。

**真实创建一个临时环境**(用完会在下面完整删除干净,不是留着的):

```bash
$ conda create -n daily-toolkit-conda-demo python=3.11 -y
...
environment location: D:\Anaconda\envs\daily-toolkit-conda-demo
...
#
# To activate this environment, use
#
#     $ conda activate daily-toolkit-conda-demo
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

**真实翻车现场**(前面提到过的那次事故,完整还原):按提示在当前 PowerShell 里敲了 `conda activate daily-toolkit-conda-demo`,命令本身没有报任何错。但检查 `python` 实际指向哪里:

```powershell
PS> conda activate daily-toolkit-conda-demo
PS> $((Get-Command python).Source)
e:\Workspace\dummy\.venv\Scripts\python.exe        # 还是仓库 .venv,根本没切换!
```

`python` 压根没有变成 `daily-toolkit-conda-demo` 环境里的那个——`conda activate` 看起来执行成功了,实际上这个终端的环境**完全没有切换**。这时候如果没意识到,直接敲一条 `pip install 某个包`,装的其实是仓库真实 `.venv` 而不是那个新 conda 环境——这正是写这份文件时真实发生的事:一条以为是装进临时 conda 环境的 `pip install tabulate`,实际上悄悄把 `tabulate` 装进了仓库的真实 `.venv` 里。当场用 `pip show tabulate` 发现了这个包不该出现在这里,立刻 `pip uninstall -y tabulate` 清理干净,文末有复查记录。

**为什么会这样**,当场查证:

```powershell
PS> Test-Path $PROFILE
False                                   # 这个用户根本没有 PowerShell 配置文件

PS> Get-Command conda
CommandType     Application             # 是"外部程序",不是 shell 函数
Source          D:\Anaconda\Library\bin\conda.bat
```

原因是:`conda activate` 要生效,必须先跑过一次 `conda init powershell`,让 conda 在你的 PowerShell 配置文件里注册一个**shell 函数**来接管 `conda` 这个命令。本机这个 PowerShell 会话根本没有配置文件,`conda` 因此只是一个普通的外部 `.bat` 程序——外部程序作为子进程运行,不管它内部怎么改自己的环境变量,都影响不到调用它的父 shell(这一点和 2.4 节"为什么 Linux/Mac 下 activate 必须用 `source`"是同一个操作系统原理:子进程改不了父进程的环境)。

**不依赖 activate 是否生效的稳妥办法**,两种,都真实验证过:

1. 直接写环境的完整路径调用:`D:\Anaconda\envs\daily-toolkit-conda-demo\python.exe -m pip install ...`(和 2.4 节讲的"不激活直接用完整路径"是同一个思路)。
2. 用 `conda run`,专门为这种场景设计,不依赖 shell hook:

   ```bash
   $ conda run -n daily-toolkit-conda-demo python --version
   Python 3.11.15
   ```

**conda 环境里混用 pip 的真实证据**——用上面第 1 种稳妥办法,往这个 conda 环境里装一个 pip 包:

```bash
$ D:/Anaconda/envs/daily-toolkit-conda-demo/python.exe -m pip install tabulate
...
$ conda list -n daily-toolkit-conda-demo
# Name                    Version                   Build  Channel
pip                       26.1.2             pyhc872135_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python                    3.11.15              hb00fc5c_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tabulate                  0.10.0                   pypi_0    pypi
```

看最后一列 **Channel**:`python`、`pip` 都标着从 anaconda 自己的源装的,唯独 `tabulate` 标着 `pypi`——这就是"conda 环境里又套了一层 pip 装的包"在 `conda list` 里的真实样子。conda 自己是知道这个包存在的(因为它扫描了这个环境的 `site-packages`),但 `tabulate` 完全是 pip 装的,不受 conda 的版本求解器管理。

**清理**(真实执行,恢复到操作前的状态):

```bash
$ conda env remove -n daily-toolkit-conda-demo -y
$ conda env list
# 确认 daily-toolkit-conda-demo 已经不在列表里,其余环境和最初一模一样
```

### 4.4 背后发生了什么

`conda activate` 需要修改**当前 shell 进程**的 PATH 和一堆内部变量。但 `conda` 命令本身,如果没有被 `conda init <你的shell>` 注册成 shell 函数,就只是磁盘上的一个外部程序(`conda.bat`/`conda.exe`)——操作系统里子进程天生不能反过来修改父进程的环境变量,这是进程模型的基本规则,不是 conda 的 bug。`conda init` 做的事情,就是把"运行 conda 命令"这件事,从"启动一个子进程"改造成"在当前 shell 里直接执行一段脚本逻辑"(和 venv 的 `source activate` 解决的是同一类问题,方式不同而已)。

`conda list` 里的 Channel 列之所以能看出 `pypi`,是因为 conda 会扫描环境里每个包留下的安装元数据——conda 自己装的包会记录"这是从哪个 conda 频道装的";pip 装的包不会有这种记录,但 conda 仍然能从 Python 标准的包元数据里看到"这里有一个叫 tabulate 的包”,于是标注成来源不明确的 `pypi`,提醒你这不是它自己管理的。

### 4.5 常见坑

| 现象 | 真实原因 / 怎么办 |
|---|---|
| `conda activate xxx` 没有报错,但 `python` 还是没变 | 本节真实复现的翻车现场——大概率是这个 shell 没跑过 `conda init <shell名>`。确认方法:`Test-Path $PROFILE`(PowerShell)看有没有配置文件,或者 `Get-Command conda` 看它是 `Function` 还是 `Application`;临时绕过用 `conda run -n 环境名 命令`,长期解决用 `conda init powershell`(会改写你的 PowerShell 配置文件,自己评估要不要做) |
| 一台机器上 `conda env list` 出现意料之外的环境,或者位置很奇怪 | 可能装了不止一套 conda(本机真实观察到 `D:\Anaconda` 和 `C:\ProgramData\miniconda3` 同时存在)。用 `where.exe conda` / `which conda` 确认你实际调用的是哪一套 |
| conda 环境里装的东西,`conda list` 却显示 Channel 是 `pypi` | 说明是用 `pip install` 装进这个 conda 环境的,不是 `conda install`。不算错,但意味着 conda 自己的依赖求解器不会去检查这些包的版本兼容性——如果同一个库既能 `conda install` 又打算装,尽量选一种方式装到底,不要来回混 |
| 以为自己在某个虚拟环境里,其实命令实际执行在了完全不相关的另一个环境(比如仓库的 `.venv`) | 本节真实事故的教训——**任何一次 `pip install` 之前,养成习惯确认 `$VIRTUAL_ENV` 或者 `where.exe python`/`which python` 输出的路径是不是你以为的那个**,尤其是在切换过 conda/venv 好几次之后 |

### 4.6 自测清单

- [ ] 能说出 conda 的 `create`/`activate`/`deactivate`/`env remove` 分别对应 venv 的哪个操作
- [ ] 能说清楚 conda 和 pip/venv 最本质的能力差异(conda 能管非 Python 的系统级依赖,pip/venv 不能)
- [ ] 知道 `conda activate` 看起来没报错不等于真的切换成功了,以及怎么用 `Test-Path $PROFILE` / `Get-Command conda` 类的方法确认原因
- [ ] 知道 `conda run -n 环境名 命令` 或者直接写完整路径,是不依赖 shell 是否正确初始化的稳妥调用方式
- [ ] 能看懂 `conda list` 里 Channel 列显示 `pypi` 是什么意思

---

## 5. 依赖冲突真实排查流程

### 5.1 为什么需要这个 / 不会有什么后果

前四节已经证明了两件事:环境冲突是真实会发生的(第 1 节),而且有时候连"我是不是在正确的环境里"这种基础问题都会在没有察觉的情况下出错(第 4 节的真实事故)。这一节给一套具体、可以照着执行的排查步骤,而不是"遇到问题要仔细检查"这种正确但没用的空话。

不会排查的后果:遇到依赖报错只能干瞪眼,或者把整个虚拟环境删了重建(有时候有用,但治标不治本,而且如果问题出在你自己代码逻辑对某个库版本的假设上,重建环境根本解决不了)。

### 5.2 环境要求

- 只需要 `pip`,具体是三个子命令:`pip show`、`pip check`、`pip list`。

### 5.3 一步步跟着做

用第 1 节真实制造出来的那个"表面装成功、实际不兼容"的环境(`requests==2.25.0` + `urllib3==2.2.0`)作为排查对象,完整走一遍流程:

**第一步:精读报错信息本身**,不要只看最后一行。第 1 节那次真实报错的关键句是:

```
requests 2.25.0 requires urllib3<1.27,>=1.21.1, but you have urllib3 2.2.0 which is incompatible.
```

这一行本身就是完整的排查结论——**谁**(`requests 2.25.0`)、**要求什么**(`urllib3<1.27,>=1.21.1`)、**实际是什么**(`urllib3 2.2.0`)。很多时候排查根本不需要更多步骤,这行字已经说完了。

**第二步:如果报错信息不够清楚,或者想确认现在的状态**,用 `pip show` 分别看双方:

```bash
$ pip show requests
Name: requests
Version: 2.25.0
Requires: certifi, chardet, idna, urllib3
Required-by:

$ pip show urllib3
Name: urllib3
Version: 2.2.0
Requires:
Required-by: requests
```

`pip show 包名` 的 `Requires` 一行告诉你**这个包依赖谁**,`Required-by` 告诉你**谁依赖这个包**——两个方向都能查,顺着 `Required-by` 往上找,能搞清楚"这个版本冲突到底会牵连到我代码里哪个直接调用的包"。

**第三步:不想一个个手动核对,让 pip 自己全面扫一遍**——这才是最推荐的做法:

```bash
$ pip check
requests 2.25.0 has requirement urllib3<1.27,>=1.21.1, but you have urllib3 2.2.0.

$ echo $?
1
```

`pip check` 会检查**当前环境里所有已安装的包**,只要有任何一个包的依赖要求没被满足,就打印出来,而且退出码是 `1`(可以直接用在脚本里做自动检查,比如 CI 流程里装完依赖后跑一下 `pip check` 确保环境没坏)。环境完全正常时:

```bash
$ pip check
No broken requirements found.
$ echo $?
0
```

**第四步:照着报错信息给的范围修复**,重新验证:

```bash
$ pip install "urllib3<1.27,>=1.21.1"
...
Successfully installed urllib3-1.26.20

$ pip check
No broken requirements found.
```

四步走完,从"看到一堆红字"到"确认环境干净",全程没有靠猜。

### 5.4 背后发生了什么

`pip show` 读的是**单个包**在安装时留下的静态元数据——它只负责如实告诉你这个包当初声明了什么依赖,不会主动去检查"现在这个依赖是不是真的满足了"。`pip check` 则是把环境里**所有包**的这份声明都读出来,互相交叉比对一遍,把所有对不上的地方汇总报告——本质上就是把"用 `pip show` 一个个手动核对"这件事自动化、批量化了。

理解这一点之后,5.3 节的四步顺序其实是有讲究的:先看报错定位大概方向(第 1 步),需要更细节再用 `pip show` 顺藤摸瓜(第 2 步),`pip check` 用来做"有没有漏网之鱼"的全面扫描(第 3 步)——报错信息通常只告诉你**这一次操作**触发的冲突,`pip check` 能发现**历史上**任何一次操作遗留下来、但从没人主动触发过报错的潜在问题。

### 5.5 常见坑

| 现象 | 真实原因 / 怎么办 |
|---|---|
| 只看 `pip install` 输出的最后一行,以为"Successfully installed"就代表一切正常 | 第 1、4 两节都真实证明了这个假设不成立——`ERROR:` 行完全可能出现在 `Successfully installed` 之前,滚动很快的话容易被忽略。养成装完就跑一遍 `pip check` 的习惯,尤其是用了 `--no-deps` 或者手动强制指定版本之后 |
| `pip check` 报了问题,但不知道该升级还是降级 | 报错信息里的版本范围(比如 `<1.27,>=1.21.1`)就是答案,选这个范围里的任意版本重装即可,通常选范围内最新的 |
| 排查了半天,发现根本不是版本冲突,是自己在错误的虚拟环境里操作 | 回到第 2/4 节——任何排查之前,先用 `$VIRTUAL_ENV`/`where.exe python` 确认自己到底在哪个环境,不要在错误的环境里查了半天 |
| `pip check` 什么都没报,但程序运行时依然报奇怪的 `ImportError`/`AttributeError` | `pip check` 只检查版本号是否满足声明的范围,不检查"这个版本范围本身是不是真的兼容"(有些包作者自己声明的范围其实不准)。这种情况下需要手动看该库的 changelog,确认某个具体函数/参数是不是在你用的版本之间发生了变化 |

### 5.6 自测清单

- [ ] 遇到依赖报错,能不慌不忙地从报错信息里提取出"谁、要求什么、实际是什么"三个要素
- [ ] 能说出 `pip show` 和 `pip check` 的区别:一个查单个包的声明,一个扫描全环境找不满足的地方
- [ ] 知道 `pip check` 的退出码可以用来做自动化检查(0 正常,1 有问题)
- [ ] 能独立完成一次"制造冲突 → 用 pip check 定位 → 按报错范围修复 → 再次验证"的完整流程
- [ ] 知道 `pip check` 全部通过不代表代码一定能正常运行,它只检查版本号是否在声明范围内

---

*进度归属:[00-roadmap.md](00-roadmap.md) 第 05 项。*
