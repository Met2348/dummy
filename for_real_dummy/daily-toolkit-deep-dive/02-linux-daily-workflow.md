# 02 · Linux 日常生产力操作 —— Shell 导航、环境变量、tmux、后台任务、日志排查、磁盘空间

> 总览见 [00-roadmap.md](00-roadmap.md)
> **和 [`rhcsa-bash-deep-dive/`](../rhcsa-bash-deep-dive/00-roadmap.md) 的边界(先说清楚,避免重复劳动)**:那条系列已经用 RHCSA 考试视角讲过大量 Linux 命令——`grep` 的正则语法(BRE/ERE)在它的 [01-essential-tools.md 第 4 点](../rhcsa-bash-deep-dive/01-essential-tools.md)、`ps`/`kill` 信号在它的 [02-process-and-boot.md 第 5-6 点](../rhcsa-bash-deep-dive/02-process-and-boot.md)、`export`/环境变量的子进程可见性机制在它的 [09-bash-scripting.md 第 1 点](../rhcsa-bash-deep-dive/09-bash-scripting.md)——**这篇完全不重复这些内容**,需要用到时只给一句话回顾 + 链接过去,不重新展开推导。这篇要讲的是那条系列**完全没有覆盖**的另一整块东西:不是"这个命令怎么用",而是"科研人员每天打开终端到关掉终端之间,那些真正会用到、决定你干活顺不顺手的操作习惯"——目录怎么跳得快、环境变量怎么设置才能不用每次重来、训练任务怎么保证你断网/合笔记本也死不掉、任务丢进后台怎么管、log 里的异常怎么用工具筛出来而不是拿眼睛翻、磁盘写满了怎么查。这些在 rhcsa-bash 的 100 个知识点里一条都没有。

---

## 0. 这篇文章是怎么验证的(先说清楚)

- **验证环境**:复用 [rhcsa-bash-deep-dive](../rhcsa-bash-deep-dive/00-roadmap.md) 已经修复好的 WSL2 **Rocky Linux 10.2**(root 会话,systemd 正常运行)。现场确认 `tmux`(`tmux-3.3a-13.20250207gitb202a2f.el10.x86_64`)和 `screen`(`screen-5.0.1-6.el10_2.x86_64`)**都已经装好**,不需要再跑 `dnf install`——如果你在自己的新机器上发现这两个命令不存在,装的命令就是 `dnf install -y tmux` 或 `dnf install -y screen`,本文第 7 节收尾实操会再提一次。
- **涉及改 `.bashrc` 的所有实验**,全部在专门为验证这篇文章新建的、和你的真实工作无关的 Linux 用户(`dtdemo`、`freshdemo`)上做,验证完 `userdel -r` 删掉,**没有碰过 root 自己真实在用的 `.bashrc`**。
- **tmux "断开连接后任务还活着"这个核心结论**,不是靠猜的,也不是靠"看起来应该这样"——专门给 WSL2 里的 Rocky Linux 实例配了一把临时 SSH 密钥,**用一条真实的 `ssh root@localhost` 连接**把训练循环丢进 tmux、真实断开连接、再用一条全新的 SSH 连接回来验证任务是否还在跑,验证完这把临时密钥已经从 `authorized_keys` 里删除。这是这篇文章里验证强度最高的一节,原因是它是整篇最重要的结论。
- **两个真实撞到、如实记录、不遮掩的环境限制**(不是"应该发生"而是真的发生了,过程写在第 3 节"常见坑"里):① 这个 WSL2 虚拟机本身在没有任何进程连着的时候会很快自动休眠/重启(这是 WSL2 宿主机制本身的特性,rhcsa-bash-deep-dive 的环境说明里也记录过,和真实远程服务器的行为无关,真实服务器不会因为你没连着就自己关机);② 这次验证用到的这个具体 tmux 构建版本(`tmux -V` 显示内部版本号是 `next-3.4`,是官方 3.3a 到下一个正式版之间的 git 快照构建),在 `tmux capture-pane` 命令上真实触发过服务端 SIGABRT 崩溃(内核日志原文能证实,见第 3 节),所以本文所有验证都**不依赖 `capture-pane` 的输出**作为证据,改用日志标记文件 + `ps` + `list-sessions` 这些没有踩到这个问题的手段。
- **需要真正的按键交互才能验证的部分**(比如 `Tab` 键触发补全的那一下敲击、`Ctrl+Z` 挂起前台任务的那一下按键):这两者本身都无法像执行一条命令那样被脚本化验证——但它们各自触发的底层机制是可以验证的:Tab 补全背后调用的候选列表生成函数是 `compgen`,可以直接调用验证;`Ctrl+Z` 发送的信号是 `SIGTSTP`,可以用 `kill -TSTP` 直接发送验证效果完全一致。这两处都如实标注了"用等价机制验证,不是真的模拟了一次按键"。

---

## 1. Shell 导航肌肉记忆 —— `cd -`、Tab 补全、目录栈

**为什么需要这个 / 不会有什么后果:**

你现在这个仓库,从根目录到一个具体的实验脚本,路径经常有五六层深。以后写论文代码、跑实验,情况只会更深:代码在一个目录,数据在另一个目录,checkpoint 存第三个目录,log 存第四个目录,而且大概率是在一台你要 SSH 上去的远程 GPU 服务器上,那台机器上的目录结构你还没熟悉。

如果每次换目录都手打完整路径,会有两个实际后果,而不是"不够优雅"这种空泛的说法:第一,**慢**——一条 `cd /data/experiments/run_2026_07/checkpoints/epoch_120` 这种路径,手打一遍再检查有没有打错,比起一个操作习惯要多花好几倍时间,一天下来的路径切换次数不会少;第二,**容易打错却不容易发现**——路径打错最常见的后果不是报错(报错至少你知道错了),而是"打成了另一个真实存在但不是你想去的目录",这种情况下你以为自己在 A 目录,操作的却是 B 目录,轻则命令找不到文件,重则如果紧接着是一条 `rm -rf ./*` 或者把训练输出写到相对路径,后果可能不可逆。本节要讲的几个习惯,核心目的就是**减少手动输入路径的次数**,输入越少,打错的机会越少。

**环境要求:**

- 任意 bash(WSL2 Rocky Linux、Git Bash、任何 Linux 服务器都一样——这是 bash 本身的功能,不是某个发行版特有的)。本文用 WSL2 Rocky Linux 验证,`bash --version` 确认是 `GNU bash, version 5.2.26(1)-release`。

**一步步跟着做:**

先建一个和后面几节共用的实验目录结构:

```
$ mkdir -p /root/dtd02-verify/{code,data,checkpoints,logs} && cd /root/dtd02-verify
$ pwd
/root/dtd02-verify
```

1. **`cd` 基础复习(很快,不是重点)**:`cd 目录名` 进子目录,`cd ..` 回上一级,`cd` 不带任何参数直接回家目录,`cd ~` 效果一样。

   ```
   $ cd code
   $ pwd
   /root/dtd02-verify/code
   $ cd ../data
   $ pwd
   /root/dtd02-verify/data
   $ cd
   $ pwd
   /root
   $ cd ~
   $ pwd
   /root
   ```

2. **`cd -`:回到"上一个你待过的目录",不用记路径。** 这是本节最值得养成的一个习惯——你经常需要在两个目录之间来回切(比如代码目录和 log 目录),`cd -` 就是"来回跳"专用的:

   ```
   $ cd /root/dtd02-verify/code
   $ cd /root/dtd02-verify/data
   $ cd -
   /root/dtd02-verify/code
   $ pwd
   /root/dtd02-verify/code
   $ cd -
   /root/dtd02-verify/data
   ```

   注意 `cd -` 会**把它跳转到的目录打印出来**——这是它和普通 `cd` 唯一的输出差异,故意的,因为"我到底跳到哪去了"这件事光看提示符不一定够明显,`cd -` 主动告诉你一声。

3. **`cd` 失败时,你的位置不会变。** 这是个容易被忽略但重要的安全属性:

   ```
   $ pwd
   /root/dtd02-verify/data
   $ cd /root/dtd02-verify/checkpoints/does/not/exist/yet
   bash: line 14: cd: /root/dtd02-verify/checkpoints/does/not/exist/yet: No such file or directory
   $ pwd
   /root/dtd02-verify/data
   ```

   路径打错、目录不存在,`cd` 会报错并且**原地不动**,不会把你扔到某个奇怪的地方——这意味着"cd 失败"本身不危险,危险的是没注意到报错、以为自己已经换过去了,接下来的命令全部在错的目录里执行。

4. **Tab 补全——用工具帮你打路径,而不是自己一个字一个字敲。** 这是效率提升最直接的一项:输入路径的前几个字符,按 `Tab` 键,shell 会自动把剩下部分补全(如果唯一匹配),或者如果有多个可能,再按一次 `Tab` 列出所有候选。

   **诚实说明**:真正按一下 `Tab` 键这个动作,没法在脚本里自动化验证(和按鼠标一样,是真实的按键交互)。但 Tab 补全背后调用的候选生成机制是 `compgen`,可以直接调用,效果和 Tab 键触发的候选列表完全一致——下面全部是真实跑出来的:

   ```
   $ compgen -d c
   checkpoints
   code
   ```

   这就是你在 `/root/dtd02-verify` 下输入 `cd c` 再按 `Tab` 会看到的效果:因为 `checkpoints` 和 `code` 都以 `c` 开头,补全没法唯一确定,会把两个候选都列出来,等你多打一个字符(比如 `ch` 或者 `co`)区分开。

   ```
   $ compgen -c tm
   tmux
   ```

   这是命令名补全(以 `tm` 开头、PATH 里能找到的命令)。

   ```
   $ touch logs/train_run_2026-07-20.log logs/train_run_2026-07-21.log
   $ compgen -f logs/train_run
   logs/train_run_2026-07-20.log
   logs/train_run_2026-07-21.log
   ```

   这是文件名补全——两个 checkpoint/log 文件名字很长又很像,手打全名极容易打错某个数字,Tab 补全能省掉这类风险。

5. **目录栈:`pushd`/`popd`/`dirs`,比 `cd -` 更进一步的场景(超过两个目录来回跳)。** `cd -` 只能记住"上一个"目录,如果你要在三四个目录之间轮流切换,目录栈更合适:

   ```
   $ cd /root/dtd02-verify
   $ pushd code
   ~/dtd02-verify/code ~/dtd02-verify
   $ pwd
   /root/dtd02-verify/code
   $ pushd ../data
   ~/dtd02-verify/data ~/dtd02-verify/code ~/dtd02-verify
   $ pwd
   /root/dtd02-verify/data
   $ dirs -v
    0  ~/dtd02-verify/data
    1  ~/dtd02-verify/code
    2  ~/dtd02-verify
   $ popd
   ~/dtd02-verify/code ~/dtd02-verify
   $ pwd
   /root/dtd02-verify/code
   $ popd
   ~/dtd02-verify
   $ pwd
   /root/dtd02-verify
   ```

   `pushd 目录` 相当于"`cd` 过去,同时把去之前的位置记到一个栈里";`popd` 是"弹出栈顶,回到那个位置";`dirs -v` 随时能看当前栈里记了哪些位置。日常科研工作流里,如果你发现自己在超过两个目录之间反复横跳(比如"改代码 → 看 log → 看 checkpoint 目录 → 改代码"),这一组比反复 `cd -` 更不容易迷路。

**背后发生了什么:**

- **`cd -` 依赖两个 shell 变量**:`$PWD`(当前目录)和 `$OLDPWD`(上一个目录)。每次 `cd` 成功后,shell 会把旧的 `$PWD` 存进 `$OLDPWD`,再把新目录写入 `$PWD`。`cd -` 其实就是 `cd "$OLDPWD"` 的简写,可以自己验证:

  ```
  $ cd /root/dtd02-verify/code
  $ cd /root/dtd02-verify/data
  $ echo "OLDPWD=$OLDPWD  PWD=$PWD"
  OLDPWD=/root/dtd02-verify/code  PWD=/root/dtd02-verify/data
  ```

- **Tab 补全的原理**:bash 内建一个叫 `readline` 的输入库,负责处理你在终端里敲的每一个字符,`Tab` 键被 `readline` 绑定为"触发补全"。补全候选的生成逻辑,正是 `compgen` 命令背后调用的同一套函数——`compgen -f` 对应文件名补全,`compgen -d` 对应目录名补全,`compgen -c` 对应命令名补全。这也是为什么 `compgen` 能作为 Tab 键效果的真实验证手段,而不是一个"看起来差不多"的替代品。
- **`pushd`/`popd` 操作的是一个真正的栈(后进先出)**:`pushd` 把新目录压栈顶,`popd` 弹出栈顶。`dirs -v` 打印的编号 `0` 永远是当前目录,编号越大表示越早压进去的。

**常见坑:**

| 现象 | 大概率原因 |
|---|---|
| `cd -` 跳到了一个意料之外的目录 | `$OLDPWD` 记的是"上一次 cd 成功前"的位置,如果中间有别的脚本/命令悄悄 `cd` 过(比如某个你 `source` 的脚本内部有 `cd`),`$OLDPWD` 会被那次覆盖,不是你以为的那次 |
| Tab 补全没反应,或者补出一堆不相关的东西 | 输入的前缀不够精确(比如目录下有十几个同前缀文件);也可能不小心切到了不支持补全的极简 shell(比如某些容器里的 `sh` 而不是 `bash`) |
| 打 `cd` 到一个目录,回车之后似乎"什么都没变" | 提示符没有显示当前路径,容易产生"是不是没生效"的错觉——不确定的时候直接 `pwd` 一下,不要凭提示符长相猜 |
| `pushd`/`popd` 用着用着栈"乱了" | 每次 `pushd` 都会往栈里加一层,忘记配对 `popd` 会越叠越多——`dirs -v` 随时检查当前栈的真实内容,不要凭记忆猜 |

**自测清单:**

- [ ] 能说出 `cd -` 依赖的两个 shell 变量是什么,并且能现场 `echo` 出来验证
- [ ] 知道 `cd` 失败时当前目录会不会变(不会),这一点为什么重要
- [ ] 能说出 Tab 补全和 `compgen` 之间的关系(同一套底层候选生成逻辑)
- [ ] 能用 `pushd`/`popd`/`dirs -v` 在三个目录之间来回切换,而不是每次都手打完整路径

---

## 2. `.bashrc` / 环境变量 / `PATH` 实操 —— 临时设置 vs 永久生效

**为什么需要这个 / 不会有什么后果:**

先明确边界:`export` 本身怎么让子进程看到一个变量、shell 变量和环境变量的区别,[rhcsa-bash-deep-dive 的 09-bash-scripting.md 第 1 点](../rhcsa-bash-deep-dive/09-bash-scripting.md)已经讲过,这里不重复推导,需要的话直接跳过去看。这一节要解决的是那条系列没碰过的一个更贴近日常的问题:**同一件"设置一个环境变量/自定义一个命令"的事,做的方式不同,效果能不能撑过"关掉终端、重新登录"这个边界,天差地别。**

具体后果:你今天在服务器上手动 `export` 了一个变量(比如某个数据集路径、某个自己写的工具脚本所在目录),当场好用,但下班关了终端;第二天重新连上去,同样的命令突然"command not found"或者报"找不到文件"——不是操作又失败了,而是昨天设置的东西根本没有被保存下来,你在一个全新的、什么都没设置过的 shell 里。更隐蔽的后果:提交到 SLURM/后台队列的训练脚本,和你手动在终端跑的时候用的不是同一个环境——如果脚本依赖某个你只在交互式终端里 `export` 过的变量,提交上去的作业会用一套不同的(缺失的)环境,行为和你手动跑时不一致,而且这类问题经常要等作业跑到中途才暴露。

**环境要求:**

- 建议不要拿你正在用的真实账号做下面的实验(尤其是要故意制造"影子命令"这种有风险的操作)。本文全程用专门新建的 Linux 用户 `dtdemo` 验证,跟着做的话也建议开一个类似的临时账号,`useradd -m dtdemo` 即可(需要 root/sudo)。
- 先看一眼一个全新用户拿到的、系统给的默认 `.bashrc` 长什么样(Rocky Linux 10.2 真实内容,不是编的):

  ```
  # .bashrc

  # Source global definitions
  if [ -f /etc/bashrc ]; then
      . /etc/bashrc
  fi

  # User specific environment
  if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]; then
      PATH="$HOME/.local/bin:$HOME/bin:$PATH"
  fi
  export PATH

  # User specific aliases and functions
  if [ -d ~/.bashrc.d ]; then
      for rc in ~/.bashrc.d/*; do
          if [ -f "$rc" ]; then
              . "$rc"
          fi
      done
  fi
  unset rc
  ```

  注意系统默认就已经在往 `PATH` 里加 `~/.local/bin` 和 `~/bin`——这两个目录不需要你自己额外加,后面第 2 步演示用的是**一个不在这个默认名单里**的目录,才能看出"没在 PATH 上"和"在 PATH 上"的真实差别。

**一步步跟着做:**

**第一部分:同一件事,临时做 vs 永久做,效果差在哪。**

先造一个"自己写的科研小工具"(内容不重要,能证明"被找到、被执行"就行),故意放在 `~/tools/`(不在默认 PATH 里):

```
$ mkdir -p ~/tools
$ cat > ~/tools/mytrainstatus <<'EOF'
#!/bin/bash
echo "training job: running, step 4200/10000, loss=1.83"
EOF
$ chmod +x ~/tools/mytrainstatus
```

**E1:基线——工具目录不在 PATH 上,直接找不到。**

```
$ mytrainstatus
-bash: mytrainstatus: command not found
$ echo $?
127
```

（退出码 `127` 是 shell 的约定:专门表示"没找到这个命令"。）

**E2:临时设置——只在当前这一个 shell 会话里生效。**

```
$ export PATH="$HOME/tools:$PATH"
$ mytrainstatus
training job: running, step 4200/10000, loss=1.83
```

**E3:换一个全新会话(相当于重新登录一次),临时设置消失了。**

```
$ mytrainstatus
-bash: mytrainstatus: command not found
$ echo $?
127
```

`export` 只改了**这一个 shell 进程自己内存里的变量**,这个 shell 一退出,改动跟着消失——不会自动写回任何文件。

**E4:把它写进 `.bashrc`,但是——光编辑文件本身,不会影响"已经在运行的"这个 shell。** 这一步分解开做,故意在同一个 shell 会话里连续验证:

```
$ mytrainstatus
-bash: line 3: mytrainstatus: command not found
$ echo 'export PATH="$HOME/tools:$PATH"' >> ~/.bashrc
$ tail -1 ~/.bashrc
export PATH="$HOME/tools:$PATH"
$ mytrainstatus
-bash: line 13: mytrainstatus: command not found
```

**关键的一点**:`.bashrc` 文件已经改了(`tail` 能看到新加的那一行),但**这个正在运行的 shell 完全没反应**——还是找不到命令。原因很直接:`.bashrc` 只在 shell **启动的那一刻**被读一次,之后这个文件再怎么改,这个已经活着的 shell 都不会自动重新读。

**E5:`source` 让"已经在运行"的这个 shell 重新读一遍 `.bashrc`,不需要重开终端。**

```
$ source ~/.bashrc
$ echo $PATH
/root/tools:/home/dtdemo/.local/bin:/home/dtdemo/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
$ mytrainstatus
training job: running, step 4200/10000, loss=1.83
$ echo $?
0
```

**E6:从此以后,新开的会话不用手动 `source`,自动就有——因为新会话本来就会在启动时读一遍 `.bashrc`。**

```
$ mytrainstatus
training job: running, step 4200/10000, loss=1.83
```

把 E1-E6 串起来看,是一条完整的因果链:临时 `export` 只活在当前进程里 → 想让它"以后每次登录都有",要写进 `.bashrc` 这个每次新开 shell 都会被自动读一遍的文件 → 但文件被改动这件事,不会主动通知"已经在运行"的旧 shell,你必须用 `source` 手动让它重读,或者干脆开一个新的会话。

**第二部分:PATH 的查找顺序,以及一个真实存在的风险——"影子命令"。**

`PATH` 是一份**目录列表**,冒号分隔,顺序敏感——当你敲一个命令名,shell 按这份列表**从左到右**依次找,**找到第一个匹配就停,不会继续找后面的**。这个"谁排在前面谁说了算"的规则,平时感觉不到,但一旦你的个人工具目录里,不小心出现一个和系统命令**同名**的脚本,而你的个人目录又排在 PATH 前面(常见,因为很多人习惯把自己的 `~/bin` 或者 conda/venv 的路径塞到 PATH 最前面),后果就是**系统命令被你自己的同名脚本"影子"覆盖了**,而且不会有任何警告。真实复现一次:

```
$ which -a python3
/usr/bin/python3
$ type -a python3
python3 is /usr/bin/python3
```

一切正常,`python3` 目前只有一个来源。现在模拟一个真实会发生的意外——`~/tools` 里有个以前调试用的、忘记删除的脚本,恰好也叫 `python3`:

```
$ cat > ~/tools/python3 <<'EOF'
#!/bin/bash
echo "!! this is NOT the real python3, this is a leftover script from an old experiment !!"
EOF
$ chmod +x ~/tools/python3
$ which python3
/home/dtdemo/tools/python3
$ python3
!! this is NOT the real python3, this is a leftover script from an old experiment !!
```

因为 `~/tools` 这次实验里被排在了 PATH 最前面(第一部分加进去的),`which python3` 直接返回了这个冒牌货,以后所有"跑一下 python3"的地方,悄悄换成了这个假脚本——**不会报错,只会算出完全错误的结果**,这种坑比"报错"隐蔽得多,真实场景里更常见的版本是:conda/venv 环境里手滑装了一个同名的自定义脚本,或者两个不同环境的 `bin` 目录顺序排反,用错了 Python 解释器版本自己都没发现。

**诊断和修复:**

```
$ type -a python3
python3 is /home/dtdemo/tools/python3
python3 is /usr/bin/python3
$ which -a python3
/home/dtdemo/tools/python3
/usr/bin/python3
```

`type -a` / `which -a` 会把 PATH 上**所有**匹配到的候选都列出来,顺序就是 shell 真正查找的顺序——第一条就是"当前生效的那一个"。确认问题后,要么删掉/改名那个冒牌脚本,要么用**完整路径**临时绕开 PATH 查找,强制用你想要的那一个:

```
$ /usr/bin/python3 --version
Python 3.12.13
$ rm ~/tools/python3
```

**背后发生了什么:**

- **`PATH` 查找是"从左到右,找到即止"**,不是"综合所有匹配结果"。这就是为什么顺序敏感,以及为什么"个人目录该不该排在系统目录前面"是一个需要谨慎对待的选择,不是随手加。
- **`.bashrc` 只在 shell 启动那一刻被读**,严格来说是"交互式非登录 shell 启动时读 `~/.bashrc`"(登录 shell 读的是 `~/.bash_profile`/`~/.profile`,这类细节 rhcsa-bash 的 bash 脚本部分有更系统的讲法,这里只强调和本节相关的一点:**改文件本身不等于改正在运行的进程状态**,这是理解 `source` 为什么存在的关键——`source 文件`(等价写法 `. 文件`)不是"运行这个脚本",而是"把这个文件里的命令,当成是你自己在当前 shell 里一行行手打的",所以它执行的 `export` 语句,改的就是当前这个 shell 自己的变量,不像直接 `bash 文件.sh` 那样是开一个新的子进程去执行、改动出了子进程就没了。
- **为什么新开的 shell 不用手动 `source` 也能生效**:纯粹是因为"新开一个交互式 shell"这个动作本身,就包含了"自动执行一遍 `.bashrc`"这一步,不是什么特殊机制,只是"从头启动"和"已经在运行、想中途重新读一遍"这两种情况,天然需要不同的操作。

**常见坑:**

| 现象 | 大概率原因 |
|---|---|
| 明明写进了 `.bashrc`,当前终端还是不生效 | 忘了 `source ~/.bashrc`,或者干脆开一个新终端 |
| `source` 之后还是不生效 | 检查有没有拼写错误(比如 `PATH` 打成 `Path`——变量名区分大小写);或者 `.bashrc` 里这一段代码写在了某个 `if` 条件分支里,条件没成立 |
| 换了新服务器,以前顺手的命令/别名全没了 | `.bashrc` 是每台机器、每个账号各自独立的文件,不会跨机器同步,换机器要么重新配一遍,要么用 dotfiles 仓库管理(这篇不展开,提一句存在这类方案) |
| 某个命令突然"行为不对但没报错",查了半天是版本不对 | 大概率是本节讲的 PATH 顺序 + 同名影子问题——`type -a`/`which -a` 是第一时间该做的排查动作,不要先怀疑代码逻辑 |
| 提交到 SLURM/后台的脚本,和手动跑的时候环境不一样 | 手动 `export` 的变量只在你当前登录的这个 shell 里,后台提交的作业未必继承同一份环境——需要的变量应该写进 `.bashrc`(每次登录都有)或者显式写进提交脚本本身,不要依赖"我手动 export 过了" |

**自测清单:**

- [ ] 能不看教程,复述一遍"临时 export → 写进 .bashrc → source 生效 → 新会话自动生效"这条完整链路,并说清楚每一步解决的是什么问题
- [ ] 能解释为什么"改了 `.bashrc` 文件"不等于"当前终端立刻生效",以及 `source` 具体做了什么
- [ ] 知道 `type -a`/`which -a` 是干什么用的,遇到"命令行为对不上预期"会第一时间想到用它排查
- [ ] 能说出 PATH 是"从左到右第一个匹配就停",而不是某种更聪明的综合判断

---

## 3. tmux(screen 备选)—— 让训练任务不因为断网/合笔记本就死掉

**为什么需要这个 / 不会有什么后果:**

先搞清楚一件事:你 SSH 登录到远程服务器后,敲的每一条命令,都是在**这次 SSH 连接开出来的那个终端(pty)**上跑的。这个终端和你的 SSH 连接绑在一起——**连接一断,这个终端就没了**,而在这个终端上直接跑起来的前台进程(比如你直接敲 `python train.py` 跑起来的训练脚本),**默认情况下会跟着一起被杀掉**(内核会给它发送 `SIGHUP`,"挂断"信号,默认处理方式就是终止进程)。

后果非常具体,而且科研场景里天天会撞上:笔记本合盖睡眠、家里/学校网络抖动断线、SSH 客户端软件意外关闭、甚至只是电脑重启——只要这次 SSH 连接断了,你直接在终端里跑起来的训练任务,大概率跟着没了,而且**很可能没有任何 checkpoint 保存,几个小时的 GPU 时间白跑**。这不是"操作失误",是终端和进程之间与生俱来的绑定关系导致的,不管你多小心敲命令都躲不开——除非任务本身跑在一个"不依赖你这次连接是否存活"的地方。这正是 `tmux`(或者 `screen`)存在的意义。

**环境要求:**

- 检查是否已装:`command -v tmux`(或 `command -v screen`)。本机 Rocky Linux 10.2 现场确认两者都已装好:

  ```
  $ tmux -V
  tmux next-3.4
  $ screen -v
  Screen version 5.0.1 (build on 2026-01-05 00:01:00)
  ```

  （`tmux -V` 打出来的 `next-3.4` 不是版本号乱掉——`rpm -q tmux` 能看到真实包版本是 `tmux-3.3a-13.20250207gitb202a2f.el10`,这是官方 3.3a 正式版之后、下一个正式版发布之前的一个 git 快照构建,构建时自报的内部版本字符串就是 `next-3.4`,不是你机器上装错了东西。）
- 如果两个都没装:`dnf install -y tmux` 或者 `dnf install -y screen`(需要 root/有 sudo 权限,复用 rhcsa-bash-deep-dive 05 号文件已经讲过的 dnf 用法,这里不重复)。

**一步步跟着做:**

**第一步:最基本的会话生命周期——创建、查看、往里面塞命令。**

```
$ tmux new-session -d -s trainjob
$ tmux list-sessions
trainjob: 1 windows (created Thu Jul 23 07:50:19 2026)
```

`-d` 表示"创建了就直接放到后台,不要把我的当前终端切换进去"——这一步之后,`trainjob` 这个会话已经在跑了,只是你现在还"看不见"它。往里面塞一条命令,用 `send-keys`(效果等价于你手动切进这个会话、在里面敲这行命令再按回车,只是这里用脚本的方式注入,方便不需要真人操作也能验证):

```
$ tmux send-keys -t trainjob 'for i in 1 2 3 4 5; do echo "epoch $i" >> ~/dtd02-verify/logs/final_progress.txt; sleep 1; done; echo FINAL_DONE > ~/dtd02-verify/logs/final_marker.txt' Enter
```

**第二步(这是整节最重要的证据):真实断开一次 SSH 连接,再真实重新连一次,验证任务到底死没死。**

下面这组命令,`ssh` 的部分是从 Windows 侧真实敲的两条完全独立的 SSH 连接(用专门为这次验证生成的临时密钥),不是同一个会话里假装"断开"——第一条连接建好会话、塞进训练循环、看一眼中途进度,然后连接主动结束;隔几秒钟之后,第二条全新连接进来确认结果:

```
### 连接 1(几秒后主动断开)###
$ ssh root@<这台机器> "
    tmux new-session -d -s finaltest
    tmux send-keys -t finaltest 'for i in 1 2 3 4 5; do echo epoch \$i >> ~/dtd02-verify/logs/final_progress.txt; sleep 1; done; echo FINAL_DONE > ~/dtd02-verify/logs/final_marker.txt' Enter
    sleep 1
    tmux list-sessions
    cat ~/dtd02-verify/logs/final_progress.txt
    cat ~/dtd02-verify/logs/final_marker.txt   # 这时候应该还不存在
  "
finaltest: 1 windows (created Thu Jul 23 07:50:19 2026)
epoch 1
epoch 2
cat: .../final_marker.txt: No such file or directory
```

（连接 1 到这里结束,SSH 连接真实关闭——此刻循环只跑了 2/5,`finaltest` 会话所在的这个"登录会话"已经不存在了。）

```
### 连接 2(全新的、和连接 1 毫无关系的一次 SSH 连接)###
$ ssh root@<这台机器> "
    sleep 5
    tmux list-sessions
    cat ~/dtd02-verify/logs/final_progress.txt
    cat ~/dtd02-verify/logs/final_marker.txt
    ps -ef | grep -E 'tmux|final' | grep -v grep
  "
finaltest: 1 windows (created Thu Jul 23 07:50:19 2026)
epoch 1
epoch 2
epoch 3
epoch 4
epoch 5
FINAL_DONE
root   698     1  0 07:50 ?  00:00:00 tmux new-session -d -s finaltest
```

**这就是结论的真实证据**:会话 `finaltest` 在两次完全独立的连接之间**依然存在**;5 个 epoch 的循环**在没有任何人连着的这段时间里自己跑完了**(`FINAL_DONE` 标记文件是在两次连接之间的空档期被写出来的);`ps -ef` 显示这个 tmux 服务进程的父进程 ID 是 `1`(也就是 `init`)——它已经不挂在你任何一次登录会话下面,是一个独立运行的常驻进程。

作为对照,同样的"断开重连"场景,**不用 tmux、直接裸跑一个后台任务**,第 4 节会给出对照实验——先在这里记住结论:tmux 里的任务几乎肯定活得下来,裸跑的不一定,取决于很多你不一定能控制的细节,不要去赌。

**第三步:真正的日常用法——`attach`/`detach`,以及为什么这两步没法在这里自动化验证。**

上面的 `send-keys`/`list-sessions` 是为了能脚本化验证才用的手段;你真实坐在电脑前操作,应该用的是:

- `tmux attach -t trainjob`(或简写 `tmux a -t trainjob`):把当前终端"接入"这个会话,就像打开一个窗口看到里面正在发生的一切,可以正常在里面敲命令、看输出。
- 接入之后,按 `Ctrl+B` 然后松开再按 `D`(这是 tmux 的默认前缀键组合,不是同时按三个键):**分离(detach)**,回到你原来的终端,里面的任务继续跑,不受影响。

**诚实说明**:这两步是真实的按键交互,没法像上面的 `send-keys`/`list-sessions` 那样脚本化验证——但它们和本节已经验证过的核心机制是同一件事:`attach` 只是把你的终端接到已经存在的会话上(会话本身早就在跑,不是这一步才开始跑),`detach` 只是断开这次接入,会话依然继续跑,和第二步验证过的"SSH 连接断开后会话依然存在"是同一个底层事实,不会因为你是主动按快捷键分离还是被动断网而有任何差别。

**第四步:screen 作为备选,一句话对比。** `screen` 解决的是同一个问题,命令风格不同:`screen -S trainjob` 创建并进入一个新会话,`Ctrl+A` 然后 `D` 分离,`screen -r trainjob` 重新接入,`screen -ls` 列出所有会话——和 tmux 是同一层的东西,选一个用顺手就行,现在大多数教程和默认安装更偏向 tmux(窗口/面板功能更丰富),但如果你接手的服务器只装了 screen 没装 tmux,不用现装,screen 一样能完成"断线不丢任务"这个核心目的。

**背后发生了什么:**

tmux 由两部分组成:一个常驻的 **server** 进程(真正跑着你的会话、窗口、面板的那个),和你每次 `attach` 时临时连上去的 **client**。`tmux new-session -d` 会先检查有没有 server 在跑,没有就启动一个,这个 server 启动后会**脱离**创建它的那个 shell(不是它的子进程那种依赖关系,而是通过标准的"守护进程化"手法——脱离控制终端、重新设置会话),之后不管是你主动 `detach` 还是 SSH 连接意外断掉,受影响的都只是 client 这一端,server 和它管理的所有会话/窗口/面板完全不受影响,里面的进程该怎么跑还怎么跑,它们的"控制终端"是 tmux 自己内部虚拟出来的伪终端,不是你那次 SSH 连接的终端,所以你那次连接的 `SIGHUP` 根本传不到它们身上。这就是为什么"断线不丢任务"不是运气,是这个架构设计出来的效果。

**常见坑:**

| 现象 | 大概率原因/怎么办 |
|---|---|
| 敲 `tmux` 相关命令,报 `no server running on ...` | 还没有任何会话被创建过,`tmux new-session` 先建一个 |
| 杀掉 server(`tmux kill-server`)之后**立刻**又 `new-session`,偶尔报 `server exited unexpectedly` | 本机验证过程中真实撞到过:杀掉旧 server 和启动新 server 抢占同一个 socket 文件之间有一个短暂的竞态窗口。**这不是真正的故障**,`sleep 1` 一下或者重试一次就正常——这条经验和 rhcsa-bash-deep-dive 03 号文件里"`losetup -d` 之后立刻查询有竞态,要轮询重试"是同一类问题,不是这篇文章独有 |
| `attach` 之后发现窗口显示错乱、内容花掉 | 通常是终端窗口大小和 tmux 记录的不一致,先试试 `Ctrl+B` 然后 `:` 输入 `refresh-client` 强制刷新;或者是这次验证里如实记录到的更底层问题——见下一条 |
| **本机验证时真实撞到的、不遮掩的问题**:`tmux capture-pane` 命令有一定概率触发 tmux 服务端崩溃 | 内核日志能直接证实:`kernel: tmux: tmux: server: potentially unexpected fatal signal 6`(信号 6 是 `SIGABRT`,程序自己触发的中止,不是被谁杀的),同时伴随 `coredump: ...: systemd-coredump pipe failed`。反复定位后确认这是**这个具体 WSL2 环境 + 这个 git 快照版本 tmux 之间的兼容性问题**,不是标准操作会踩到的坑,也是为什么本文所有验证证据都改用日志标记文件/`ps`/`list-sessions`,完全不依赖 `capture-pane` 的输出——如果你自己环境里也遇到类似的崩溃,大概率不是你的操作有问题,换成直接 `attach` 用眼睛看,或者把命令的输出重定向到文件里再 `cat`,都能绕开这个特定命令 |
| 忘记自己开过哪些会话,越攒越多 | `tmux list-sessions` 随时看一眼全部有哪些;不需要的 `tmux kill-session -t 名字` 及时清掉,不要放着不管 |

**自测清单:**

- [ ] 能说清楚"SSH 连接断开为什么会杀死前台任务"这条因果链(终端消失 → 内核发 `SIGHUP` → 默认处理是终止)
- [ ] 能说出 tmux 的 server/client 结构,以及为什么 server 不受你这次连接死活的影响
- [ ] 能独立完成"新建会话 → 塞一个长时间任务进去 → detach → 重新登录 → attach 验证任务还在跑"整个流程,不看教程
- [ ] 知道 `tmux list-sessions` 和 `tmux kill-session` 分别是干什么的
- [ ] 能说出至少一条本节"常见坑"表格里真实记录的问题,以及对应的排查思路

---

## 4. 后台任务管理 —— `&`、`nohup`、`disown`、`jobs`、`fg`、`bg`

**为什么需要这个 / 不会有什么后果:**

tmux 是"从一开始就计划好这个任务要长期跑、要能重新接入看"的场景该用的工具。但现实中还有两类更轻量的情况:① 你压根忘了开 tmux,已经在终端里直接跑起来一个任务,现在想"补救",让它不要因为你接下来要断开连接就死掉;② 只是想让当前这条命令别占着终端、你好继续敲别的命令,还没到需要一个完整持久化会话的程度。这两种情况,`&`/`jobs`/`fg`/`bg`/`nohup`/`disown` 这一组 shell 内建的轻量级任务管理机制,比专门开一个 tmux 会话更直接。

**环境要求:** 同上,任意 bash。

**一步步跟着做:**

**第一步:`&` 把命令丢到后台,`jobs` 看当前 shell 名下有哪些后台任务。**

```
$ sleep 100 &
[1] 751
$ jobs -l
[1]+   751 Running                 sleep 100 &
```

`&` 后面 shell 立刻返回给你一个新提示符,不用等 100 秒;`[1]` 是这个后台任务在**当前 shell 里**的编号(不是系统级 PID,系统级 PID 是后面那个 `751`),后面 `kill %1`、`fg %1` 都用这个编号,比记 PID 方便。

**第二步:`Ctrl+Z` 挂起一个前台任务,`bg` 让它转后台继续跑,`fg` 把后台任务拉回前台。**

`Ctrl+Z` 是真实按键交互,这里用它触发的信号 `SIGTSTP` 直接验证,效果完全一致:

```
$ sleep 100 &
$ kill -TSTP %1
$ jobs -l
[1]+   787 Stopped                 sleep 100
```

注意状态变成了 `Stopped`——**挂起不是杀死**,进程还在,只是被暂停调度,不占用 CPU,随时能恢复。恢复到后台继续跑:

```
$ bg %1
[1]+ sleep 100 &
$ jobs -l
[1]+   787 Running                 sleep 100 &
```

拉到前台(用一个短任务演示,这样能很快看到 `fg` 返回):

```
$ sleep 2 &
[2] 794
$ fg %2
sleep 2
```

`fg` 之后 shell 会等这个任务结束(或者你再按一次 `Ctrl+Z`)才把提示符还给你——这也是为什么"忘记开 tmux、任务已经在前台跑着了"这种情况的标准补救流程是:`Ctrl+Z` 挂起 → `bg` 转后台继续跑 → 该干嘛干嘛,不用干等着。

**第三步:为什么只有 `&` 不够——`SIGHUP` 机制,以及一个如实记录的、比想象中更微妙的真实测试结果。**

第 3 节讲过,连接断开时内核会给终端上的前台进程发 `SIGHUP`。但 `&` 扔到后台的任务,**已经不是前台进程组的一员**,理论上不会直接收到这个信号——不过 bash 自己还有一层逻辑:文档记载,一个**交互式**的 bash 在自己收到 `SIGHUP` 快要退出之前,会**主动把 `SIGHUP` 转发给它 jobs 列表里记着的所有后台任务**,不管这些任务是不是已经转入后台。

如实记录一个真实验证结果:本机用一条真实 SSH 连接,起一个不加任何保护的后台任务,连接结束后再连回来检查——**这个任务其实跑完了**,并没有被杀死。这不代表"不用 nohup 也没事"是通用结论——上面那条"交互式 shell 会转发 SIGHUP 给后台任务"的机制是 bash 官方文档写明的真实行为,会不会触发取决于你这次连接的 shell 到底算不算" bash 自己认定的交互式",这本身就是一个容易被环境细节影响、不值得去赌的判断。**稳妥的习惯是:任何你想保留的后台任务,无论这次连接看起来会不会杀死它,都主动加保护**,而不是走一步算一步。

**第四步:`nohup`——让进程从一开始就"拒收" `SIGHUP`。**

```
$ nohup sleep 100 > nohup_test.out 2>&1 &
[1] 829
$ ps -p 829 -o pid,ppid,cmd
    PID    PPID CMD
    829     826 sleep 100
$ grep SigIgn /proc/829/status
SigIgn:	0000000000000001
```

`/proc/PID/status` 里的 `SigIgn` 是这个进程当前**明确选择忽略**的信号位图,第 0 位(对应信号 1,也就是 `SIGHUP`)是 `1`——`nohup` 在把自己换成真正要跑的程序(这里是 `sleep`)**之前**,先把 `SIGHUP` 设成"忽略",这个忽略状态会跟着保留下来。直接验证效果:

```
$ kill -HUP 829
$ kill -0 829 ; echo "still alive: $?"
still alive: 0
```

`kill -HUP` 打过去,进程原地不动,`kill -0`(不发送任何信号,只用来测试进程存在与否)确认它确实还活着。对比不加 `nohup` 的裸后台任务:

```
$ sleep 100 &
[1] 811
$ kill -HUP 811
[1]+  Hangup                  sleep 100
$ kill -0 811 ; echo "still alive: $?"
still alive: 1
```

没有 `nohup` 保护,同一个 `SIGHUP` 直接把它干掉了(退出码 `1` 表示进程已经不在)。**`> 文件 2>&1` 这部分同样重要,不是可省略的装饰**:`nohup` 只管信号,不管输出——不重定向的话,`sleep`/`python train.py` 这类程序的标准输出/错误默认还是连着你这次的终端,连接一断,这部分输出就没地方可写了(`nohup` 自己会在你没有显式重定向的时候,默认把输出转存到当前目录下的 `nohup.out`,但显式指定一个你自己看得到、找得到的文件名,永远比依赖这个默认行为可靠)。

**第五步:`disown`——把任务从当前 shell 的 jobs 表里"除名"。**

```
$ sleep 100 &
[1] 819
$ jobs -l
[1]+   819 Running                 sleep 100 &
$ disown %1
$ jobs -l
$ kill -0 819 ; echo "still alive: $?"
still alive: 0
```

`disown` 之后,`jobs -l` 什么都不列了(空的)——但进程本身完全没受影响,还在正常跑(`kill -0` 确认存活)。它解决的是第三步提到的那条机制:一个任务如果已经不在 shell 自己的 jobs 表里,shell 收到 `SIGHUP` 快退出时,**没有什么可转发的对象了**,间接起到保护效果——但这个保护只针对"shell 自己转发"这条路径,不改变进程自身对 `SIGHUP` 的处理方式(这一点和 `nohup` 不一样,`nohup` 是从进程自己的信号处理设置入手,双重保险,更彻底)。

**第六步:两个组合起来,是"事后补救"最稳妥的搭配。** 如果任务已经在前台跑起来、忘了加 `nohup`,标准补救动作是:`Ctrl+Z` 挂起 → `bg` 转后台 → `disown` 把它从 jobs 表里摘出去。没法在事后补上 `nohup` 那种"从一开始就忽略信号"的保护,但 `disown` 至少堵住了"shell 主动转发 SIGHUP"这条已知的路径,配合任务本身对普通 `SIGHUP` 也不一定敏感(很多长时间运行的程序,尤其是 Python 训练脚本,默认对 `SIGHUP` 就是"收到就死"的默认行为,和 `sleep` 一样),这也是为什么"提前用 tmux 或者提前用 nohup",永远比"事后补救"更可靠——第三步已经验证过,补救类手段的效果**取决于一些你不一定完全清楚的环境细节**,不是 100% 保证。

**背后发生了什么:**

- **`SIGHUP` 的默认处理方式是终止进程**,这是 Unix 的历史传统("挂断电话线"字面意思的信号),大多数程序不会主动去改这个默认行为,除非像 `nohup` 这样显式设置成忽略。
- **`nohup` 的实现方式**:它本身是一个很薄的包装程序,先调用 `sigaction` 把 `SIGHUP` 设成 `SIG_IGN`(忽略),再用 `exec` 系列调用**把自己替换成**真正要跑的程序——`SIG_IGN` 这个设置在 `exec` 之后会被保留下来(这是 POSIX 的规定:忽略状态跨 `exec` 保留,自定义的信号处理函数则不会,会被重置回默认),这就是为什么 `ps` 看到的最终是 `sleep 100` 而不是 `nohup sleep 100` 两层进程,但信号忽略的效果还在。
- **`disown` 只操作 shell 自己内部的 jobs 表**,不碰对应进程的任何内核层面属性——这是它和 `nohup` 的本质区别:一个是"改变进程本身的行为",一个是"改变 shell 要不要管这个进程"。

**常见坑:**

| 现象 | 大概率原因 |
|---|---|
| `bg %1`/`fg %1` 报"没有这个任务" | 任务编号只在**当前 shell 会话**内有效,换一个新终端/新连接,`%1` 指代的历史全部作废,先 `jobs -l` 确认当前会话里真实存在的编号 |
| 已经 `nohup` 了,`SSH` 断开之后任务还是没了 | 检查是不是漏了输出重定向导致程序在写入卡住;也可能任务根本不是被 `SIGHUP` 杀的,是别的原因(比如内存不够被系统的 OOM killer 杀掉)——`nohup` 只挡得住 `SIGHUP`,挡不住这一类完全不同的死因,先确认死因是不是 `SIGHUP` 再往这个方向排查 |
| `nohup command &` 之后立刻 `kill -HUP` 那个 PID,进程居然还是死了 | 真实撞到过的边界情况:`nohup` 从"进程刚 fork 出来"到"真正完成 exec、信号忽略设置生效"之间有极短暂的窗口,如果信号来得比这个窗口还快,确实可能踩中——但现实中没有人会在启动命令的同一毫秒发送信号,断开连接的时间差以秒计,不会真的撞上这个极端窗口,不需要为这个特意加保护 |
| `jobs` 什么都不显示,但 `ps` 能看到进程还在跑 | 大概率是被 `disown` 过——这是预期行为,不是丢失了 |
| 分不清该用 tmux 还是该用 `nohup` | 需要中途重新连上去看实时输出、发交互指令,选 tmux;只是"启动了就不用管、跑完看结果文件就行"的一次性任务,`nohup ... &` 更轻量 |

**自测清单:**

- [ ] 能独立完成"起一个后台任务 → `Ctrl+Z`(或 `kill -TSTP`)挂起 → `bg` 恢复 → `fg` 拉回前台"整个循环
- [ ] 能说清楚 `nohup` 和 `disown` 各自保护的是哪一层(进程自身的信号处理 vs shell 的 jobs 表),为什么两者不是互相替代关系
- [ ] 能解释为什么"不加 nohup 也可能侥幸活下来"不能当成日常依赖的结论
- [ ] 知道 `nohup command > 文件 2>&1 &` 里,重定向那部分为什么不能省

---

## 5. 文本处理组合拳 —— 从训练 log 里揪出异常

**为什么需要这个 / 不会有什么后果:**

`grep` 本身的正则语法,[rhcsa-bash-deep-dive 01-essential-tools.md 第 4 点](../rhcsa-bash-deep-dive/01-essential-tools.md)已经讲过,这里不重复。这一节要解决的是更具体的日常问题:一次训练跑下来,log 文件几百上千行是常态,真正有价值的信号——loss 变成 `nan`、显存爆了、某个 worker 掉线——往往只占其中几行。人眼从头翻到尾找这几行,慢且容易漏看;打开编辑器用 `Ctrl+F` 一个个关键词试,效率也不高,而且没法做"两个条件组合"这种查询。这一节用一份**真实生成、真实跑过下面所有命令**的训练 log,演示怎么把 `grep`/`awk`/管道串成能真正解决问题的组合拳。

**环境要求:** 任意 bash + GNU `grep`/`awk`/`sed`(Rocky Linux 默认自带,Git Bash 也有)。

**一步步跟着做:**

先看一眼这份 log 长什么样(用脚本生成的、模拟了一次真实训练里典型的异常序列:梯度爆炸警告 → 显存溢出 → loss 变 nan → 从 checkpoint 恢复 → worker 掉线又重连,一共 404 行):

```
$ wc -l train_run.log
404 train_run.log
$ head -3 train_run.log
2026-07-20T10:00:04 [INFO] step=100 loss=2.4709 lr=0.0010 throughput=138.3tok/s
2026-07-20T10:00:08 [INFO] step=200 loss=2.4203 lr=0.0010 throughput=140.7tok/s
2026-07-20T10:00:12 [INFO] step=300 loss=2.3982 lr=0.0010 throughput=146.1tok/s
```

**第一步:先搞清楚"问题有多大"——总行数 vs 异常行数。**

```
$ grep -c -E "WARNING|ERROR" train_run.log
4
```

404 行里只有 4 行真正标了 `WARNING`/`ERROR`——这就是"靠人眼睛翻"效率低的根本原因:信号密度太低,人很容易在无聊的重复行里失去专注度,漏看关键的那几行。

**第二步:把这 4 行连同行号一次性揪出来。**

```
$ grep -nE "WARNING|ERROR" train_run.log
128:2026-07-20T10:08:32 [WARNING] gradient norm exceeded threshold: 15.23 > 10.0, clipping
154:2026-07-20T10:10:12 [ERROR] CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.65 GiB total capacity; 21.88 GiB already allocated)
223:2026-07-20T10:14:44 [ERROR] worker 3 disconnected: connection reset by peer
225:2026-07-20T10:14:48 [WARNING] worker 3 reconnected after 2 retries
```

有了行号,就知道接下来该往哪个方向查。

**第三步:只看这一行还不够——用 `-B`/`-A` 把"案发现场"前后几行一起调出来。**

```
$ grep -n -B2 -A3 "CUDA out of memory" train_run.log
152-2026-07-20T10:10:04 [INFO] step=15100 loss=0.2271 lr=0.0010 throughput=144.5tok/s
153-2026-07-20T10:10:08 [INFO] step=15200 loss=0.2438 lr=0.0010 throughput=145.0tok/s
154:2026-07-20T10:10:12 [ERROR] CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.65 GiB total capacity; 21.88 GiB already allocated)
155-2026-07-20T10:10:12 [INFO] step=15300 loss=nan lr=0.0010 throughput=0.0tok/s
156-2026-07-20T10:10:16 [INFO] step=15400 loss=2.0570 lr=0.0010 throughput=148.8tok/s
157-2026-07-20T10:10:20 [INFO] step=15500 loss=2.0439 lr=0.0010 throughput=148.3tok/s
```

这几行连起来看就是一个完整的故事:显存溢出前 loss 一直健康地在 `0.22`~`0.24` 附近(`-B2` 显示的两行);溢出那一刻紧接着下一步 loss 直接变成 `nan`(`-A3` 的第一行);再往后 loss 突然跳回 `2.05`(`-A3` 后两行)——这个"跳回高位"的模式,通常意味着训练脚本自动从上一个 checkpoint 恢复了。**只看 `ERROR` 那一行,你只知道"显存爆过一次";把前后文一起看,你能重建出完整的故障与恢复过程**,这正是 `-B`/`-A` 的价值。

**第四步:`loss=nan` 本身值得单独、显式地抓,不要依赖"级别标签"。**

```
$ grep -n "loss=nan" train_run.log
155:2026-07-20T10:10:12 [INFO] step=15300 loss=nan lr=0.0010 throughput=0.0tok/s
```

注意这一行的级别标签是 `[INFO]`,不是 `[WARNING]`/`[ERROR]`——很多训练框架的 `nan` 不会主动升级日志级别,它就安安静静地以一条"正常"日志的样子存在,如果你的排查思路只依赖"筛 WARNING/ERROR"(第一、二步那种),**这一行会被完全漏掉**。这是为什么"直接对着你最关心的具体异常值做字符串匹配"(这里是 `loss=nan`),比"依赖日志框架自己分类的级别标签"更可靠——级别标签是日志框架的主观判断,不一定和你的判断一致。

**第五步:`awk` 入门——按列处理,不是按整行。** `grep` 只能整行匹配,想要"抽取出 loss 这一个数值单独处理"(比如画图、比如找最大值),需要 `awk`。`awk` 的心智模型很简单:它把每一行按空白字符自动切成一个个"字段",`$1` 是第一个字段、`$2` 第二个,以此类推,`$0` 是整行;你可以用 `-F` 指定别的分隔符。先看最基础的用法——统计每种级别各有多少行:

```
$ awk '{print $2}' train_run.log | sort | uniq -c
      2 [ERROR]
    400 [INFO]
      2 [WARNING]
```

`{print $2}` 取每行第二个空白分隔的字段(这份 log 格式是 `时间戳 [级别] 内容...`,第二个字段正好是级别标签),`sort | uniq -c` 是标准的"统计计数"组合拳,和 `awk` 本身没关系,是任何输出行文本时都能配合用的通用套路。

**第六步:用 `-F` 自定义分隔符,精确抽出 loss 数值。**

```
$ awk -F'loss=' '/\[INFO\]/{split($2,a," "); if (a[1] != "nan") print a[1]}' train_run.log | sort -n | tail -3
2.3982
2.4203
2.4709
```

这一条稍微复杂,拆开看:`-F'loss='` 把每行从 `loss=` 这个位置切开,`$2` 就是"loss= 后面的所有内容"(比如 `2.4709 lr=0.0010 throughput=...`);`split($2,a," ")` 再按空格把 `$2` 进一步拆开,`a[1]` 就是纯粹的数值部分;判断不等于字符串 `"nan"` 才打印,过滤掉不是数字的那一行;最后接 `sort -n | tail -3` 找出数值最大的三个——这份 log 里最高的几个 loss 值出现在训练最开始(`2.47` 附近),符合训练一开始 loss 最高、逐渐下降的正常预期。

**第七步(如实记录一个真实踩到的怪异行为):数值比较遇到字符串 `"nan"` 混进来,结果不可靠——这也是为什么第四步要单独用字符串匹配抓 nan,不能只靠数值比较。**

```
$ grep '\[INFO\]' train_run.log | awk -F'loss=' '{split($2,a," "); print a[1]}' \
    | awk 'NR>1 && $1>prev+0.3 {print "spike at line " NR ": " prev " -> " $1} {prev=$1}'
spike at line 153: 0.2438 -> nan
spike at line 154: nan -> 2.0570
```

本意是想写一个"loss 突然跳变超过 0.3 就报警"的检测逻辑,真实跑出来发现:字符串 `"nan"` 混进本该是纯数字的比较时,`awk` 判定 `"nan" > 数字` 这类比较的行为**不总是符合直觉**(不同 `awk` 实现、不同场景下对非数字字符串参与数值比较的处理方式本身就不统一)。这正说明了第四步的做法更可靠:**想抓 `nan` 这种特定值,直接用字符串匹配(`grep "loss=nan"`)去精确地抓,不要指望一段通用的"数值突变检测"逻辑能顺带把它可靠地覆盖到。** 两种手段各有用途,不能互相替代。

**第八步:`tail -f`——训练还在跑的时候,实时看新增的 log 内容。** 前面几步都是"事后翻一份已经写完的 log";如果训练还在进行中,想要新的一行一出现就立刻看到,不用反复手动重新打开文件:

```
$ tail -f train.log
```

会一直占据这个终端,新写入的内容出现就立刻打印,`Ctrl+C` 退出跟踪(不影响被跟踪的文件或者写它的进程)。真实验证过这个行为:后台一个进程每隔 1 秒往文件里追加一行,`tail -f` 全程能实时看到每一行出现,不需要重新执行命令。日常最实用的写法是配合 `grep` 一起用,只看你关心的部分:

```
$ tail -f train.log | grep --line-buffered -E "WARNING|ERROR|loss=nan"
```

`--line-buffered` 这个参数值得单独提一句:`grep` 默认在输出不是直接连着终端(比如接了管道)的时候,会攒一批数据再一次性输出,用在 `tail -f` 这种"我要的是实时看到"的场景里会导致明明有新行但迟迟不显示——加上这个参数强制它每一行都立刻输出,不攒批。

**第九步:把某一类异常的具体内容去重,看清楚"到底出现过几种不同的错误",而不是"出现过几次"。**

```
$ grep '\[ERROR\]' train_run.log | awk -F'] ' '{print $2}'
CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.65 GiB total capacity; 21.88 GiB already allocated)
worker 3 disconnected: connection reset by peer
```

`-F'] '` 按 `"] "`(右方括号加空格)切分,这份 log 里唯一出现这个模式的地方就是 `[ERROR]` 标签后面,所以 `$2` 精确取到了去掉时间戳和级别标签之后、纯粹的错误内容本身。长时间训练如果同一类错误反复出现很多次,再接一个 `sort | uniq -c`,就能看出"这几十条 ERROR 其实是同一个根因反复触发"还是"真的是好几种不同的问题"。

**背后发生了什么:**

- **管道 `|` 的本质**:左边命令的标准输出,直接连接成右边命令的标准输入,两个是**同时运行的两个独立进程**,数据像流水线一样传递,不需要先把左边整个跑完、写出一个中间文件,右边再去读——这也是为什么 `tail -f | grep` 这种"持续产生数据"的场景能正常工作,而不是卡住等第一个命令"结束"(`tail -f` 本身就不会主动结束)。
- **`awk` 把每一行看成一条"记录"(record),默认按空白字符切成"字段"(field)**——这是它和 `grep`(只做整行匹配/不匹配的判断)、`sed`(主要做整行的替换/删除)最本质的区别:`awk` 天生是"表格"思维,你的 log 只要大致长得像"一行是一条记录、空格或者某个分隔符隔开不同信息",`awk` 就能很自然地处理。
- **为什么要养成组合小工具的习惯,而不是找一个"一站式"命令**:`grep` 负责"筛选出哪些行值得看",`awk` 负责"从这些行里精确取出你要的那个值",`sort`/`uniq -c` 负责"排序/计数"——每个工具只做一件事,但组合起来的表达能力,比任何单一工具试图"既筛选又提取又统计"要灵活得多,而且你今天学会的这几个工具,以后处理任何结构类似的文本(不只是训练 log)都能直接复用这套思路。

**常见坑:**

| 现象 | 大概率原因 |
|---|---|
| `awk -F'loss='` 抽出来的值里混进了不相关的内容 | log 格式本身不规整(比如某一行里 `loss=` 这个词组恰好在别的地方也出现过);先用 `grep 'loss='` 看看真实匹配到的都是些什么行,不要假设格式绝对统一 |
| `tail -f \| grep 关键词` 明明有匹配的新行,但半天不显示 | 忘了加 `--line-buffered`,`grep` 在管道里默认攒批输出 |
| 数值比较(比如找最大值/检测突变)结果诡异 | 字段里混进了非数字字符串(`nan`、空字符串、单位后缀比如 `138.3tok/s` 里的 `tok/s`),数值比较前一定要先确认字段是"纯净"的数字,必要时像第六步那样显式过滤掉不是数字的情况 |
| `grep -B`/`-A` 一次给出太多不相关的上下文,反而看花眼 | 上下文行数不是越多越好,先给小一点的数字(比如 2-3 行)看一眼故障发生前后的直接语境,需要更大范围再加大 |
| 好几个 `WARNING`/`ERROR` 实际是同一个根因反复触发的 | 用第九步的"去重看种类"思路——数量多不代表问题多,可能只是一个问题反复报了很多遍 |

**自测清单:**

- [ ] 拿到一份新的、没见过格式的 log 文件,能在几分钟内用 `grep -c`/`grep -n` 判断"问题严重不严重、大概在哪几行"
- [ ] 能说出 `-B`/`-A` 和单纯 `grep 关键词` 的区别,以及为什么"案发现场前后文"经常比"案发那一行"本身信息量更大
- [ ] 能写一条 `awk -F` 命令,从形如 `key=value` 的字段里精确抽出某个 value
- [ ] 知道为什么"依赖日志级别标签筛选"可能会漏掉真正重要的异常(比如本节的 `loss=nan` 例子),以及应对办法
- [ ] 能说出管道 `|` 连接的两个命令是不是同时运行的(是),这一点为什么让 `tail -f | grep` 能正常工作

---

## 6. 磁盘空间排查 —— `du`/`df`,以及"删了文件但空间没释放"

**为什么需要这个 / 不会有什么后果:**

服务器磁盘写满,是科研场景里相当高频的一类事故:checkpoint 越存越多没清理、数据集下载了好几份重复的、log 文件疯狂增长——后果通常不是"提前警告",而是训练脚本写 checkpoint 或者写 log 的那一刻**突然报错**(磁盘没空间了,系统调用直接失败),辛苦跑了很久的训练可能因为最后一步保存失败而白费。更麻烦的是"磁盘满了,但不知道是谁占的"——一台服务器可能几十上百 GB 的数据分散在很多目录里,人工一个个目录点开看大小极其低效。`du`(disk usage,查目录/文件占用了多少空间)和 `df`(disk free,查文件系统整体还剩多少空间)就是解决这两类问题的标准工具。

**环境要求:** 任意 Linux(`du`/`df` 是最基础的 POSIX 工具,到处都有)。

**一步步跟着做:**

**第一步:`df -h`——整体视角,这台机器/这个文件系统还剩多少空间。**

```
$ df -h /root
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G  1.9G  954G   1% /
```

`-h` 是"human-readable"(人类可读),自动把字节数换算成 `G`/`M` 这种好读的单位,不加的话默认是以 1K 块为单位的一长串数字,基本没人愿意手动换算。**`df` 看的是"这个文件所在的整个文件系统"**,不是某个具体目录——这是它和接下来要讲的 `du` 最根本的区别。

**第二步:`du -sh`——具体某个目录到底占了多少。** 先制造一个典型的科研项目目录结构(checkpoint、数据、log、代码各占一部分空间):

```
$ du -sh project
271M	project
```

`-s` 表示"只汇总一个总数"(summarize),不展开列出里面每个子文件/子目录各自的大小;不加 `-s` 的话,`du` 默认会**递归打印每一层子目录**各自的大小,对付一个层级很深的目录会刷屏刷到没法看。

**第三步:"到底是哪个子目录在占空间"——这才是真实排查时最常问的问题。**

```
$ cd project && du -sh */
241M	checkpoints/
31M	data/
40K	logs/
8.0K	src/
```

`*/` 这个通配符只匹配目录(末尾的 `/` 保证不会把普通文件也算进来),配合 `-s` 对每一个子目录分别给一个总数,一眼就能看出 `checkpoints/` 是大头。想要自动按大小排序,接 `sort -rh`(`-r` 倒序,`-h` 按人类可读单位的大小正确排序,不是简单按字符串排——不然 `2M` 会排在 `10M` 前面,因为字符串比较 `'2'>'1'`):

```
$ du -sh */ | sort -rh
241M	checkpoints/
31M	data/
40K	logs/
8.0K	src/
```

另一种常用写法,`--max-depth=1`,效果类似但连当前目录自己的总数也一起给出:

```
$ du -h --max-depth=1
8.0K	./src
31M	./data
241M	./checkpoints
40K	./logs
271M	.
```

**第四步(这是本节最容易被忽视、也最值得记住的一个真实现象):`rm` 删除了文件,`du` 立刻不认这个文件了,但 `df` 显示的已用空间可能完全没变。**

先看基线:

```
$ df -h /root | tail -1
/dev/sdd       1007G  2.1G  954G   1% /
```

制造一个 300MB 的"log 文件",并且让一个后台进程(模拟"还在写日志的训练进程")一直保持着这个文件的打开状态:

```
$ dd if=/dev/zero of=logs/huge_leftover.log bs=1M count=300 status=none
$ ( exec 9< logs/huge_leftover.log; sleep 12 ) &
[1] 824
$ df -h /root | tail -1
/dev/sdd       1007G  2.4G  954G   1% /
```

（多用了 300MB,`2.1G` 变 `2.4G`,符合预期。）现在,假设有个清理脚本(或者你自己手滑)删掉了这个"看起来已经用不到"的文件:

```
$ rm logs/huge_leftover.log
$ du -sh logs/
40K	logs/
$ ls -lh logs/ | grep huge
（没有任何输出——文件确实"不见了")
```

**关键的一步**——再看一次 `df`:

```
$ df -h /root | tail -1
/dev/sdd       1007G  2.4G  954G   1% /
```

**空间完全没有被释放!** 直到那个还占着这个文件的后台进程真正结束:

```
$ wait 824
$ df -h /root | tail -1
/dev/sdd       1007G  2.1G  954G   1% /
```

**这才真正掉回了 `2.1G`。**

**第五步:遇到这种"`du` 说没了、`df` 说还占着"的情况,怎么揪出真正占着这个文件的进程。**

```
$ ls -la /proc/824/fd
...
lr-x------ 1 root root 64 Jul 23 08:00 9 -> /root/dtd02-verify/project/logs/huge2.log (deleted)
```

`/proc/PID/fd/` 下面列出这个进程当前打开的所有文件描述符,如果某一项指向的路径后面**明确标注了 `(deleted)`**,说明这个进程手里还攥着一个"已经被删除、但空间还没真正归还"的文件——这是不需要装任何额外工具、Linux 系统自带就能查的方法。如果知道具体是哪个文件、只是不确定是哪个进程占着,`fuser` 更直接(Rocky Linux 默认装的 `psmisc` 包自带):

```
$ fuser -v logs/huge3.log
                     USER        PID ACCESS COMMAND
logs/huge3.log:
                     root        898 f.... sleep
```

直接告诉你 PID `898`(这里是 `sleep` 进程)正占着这个文件。确认之后,要么让这个进程正常结束(它写完该写的东西自己退出),要么(确认没有别的影响后)手动结束这个进程,空间才会真正被系统收回。

**背后发生了什么:**

- **`du` 统计的是"目录树遍历到的、当前还挂在目录结构里的文件"**——它本质上是在做"打开目录,列出里面的条目,递归下去,把每个文件实际占的磁盘块数加起来"这件事。一旦一个文件被 `rm`,它在目录结构里的那个"名字条目"立刻消失,`du` 从目录树的角度确实再也看不到它,自然不会把它算进去。
- **`df` 统计的是文件系统级别、真正被分配出去的磁盘块数量**——这是更底层的记账方式。Linux 文件系统里,一个文件的"名字"(目录项)和这个文件真正的数据内容(inode 及其指向的磁盘块)是两回事:`rm` 删除的**只是名字**,只有当"指向这个 inode 的名字数都变成 0,**并且**没有任何进程还打开着这个文件"两个条件同时满足,内核才会真正回收这些磁盘块、更新 `df` 看到的已用空间。这也是为什么"我明明删了文件,`du` 也确认没了,`df` 却说空间没变"不是 bug,而是这套"引用计数"机制的正常表现——它是内核为了在文件被删除的同时,还允许"当前正打开着它的进程"能继续正常读写这份数据而设计的,是一个**特性**,不是缺陷,只是很多人(包括不少工作了一段时间的工程师)第一次遇到时都会一头雾水。

**常见坑:**

| 现象 | 大概率原因 |
|---|---|
| `df` 和 `du` 报的空间用量对不上 | 大概率就是本节第四步的现象——有进程还打开着已经被删除的文件;用 `/proc/PID/fd` 或者 `fuser` 排查 |
| `du -sh 目录` 跑得很慢 | 目录里文件数量巨大(比如几十万个小文件的数据集)时,`du` 需要真的遍历每一个,没有捷径;如果只是想快速摸个大概,`du -sh --max-depth=1` 配合先看一眼哪个子目录最大,再往下钻,比一次性对最深层跑 `du` 划算 |
| 删了一堆文件,`df` 显示的可用空间涨得比预期慢 | 如果磁盘配置了定期快照/有其他进程持有引用,回收可能不是瞬时的;检查有没有正在运行的进程还占着相关文件 |
| `du` 和 `ls -la` 手动加总算出来的大小对不上 | `du` 默认统计的是文件实际占用的磁盘块(会按文件系统的块大小取整,通常比文件"逻辑大小"略大),`ls -l` 显示的是文件的逻辑字节数——两者统计口径本来就不完全一样,小文件多的目录这个差异会比较明显 |
| 想找"最近突然变大的目录",不知道从哪下手 | 没有一个绝对可靠的单命令能回答"最近变化",但 `du -sh */ \| sort -rh` 先定位到当前最大的几个子目录,通常就是问题所在——多数情况下"占用最大"和"最近增长"是同一个目录 |

**自测清单:**

- [ ] 能说出 `du` 和 `df` 分别是从哪个层面统计磁盘占用的(目录树遍历 vs 文件系统级块分配),这是理解两者为什么会"对不上"的关键
- [ ] 能用 `du -sh */ | sort -rh` 在一个多层目录结构里,快速定位到占用空间最大的子目录
- [ ] 遇到"删了文件但 `df` 显示空间没变",知道第一反应该查什么(有没有进程还打开着这个已删除的文件)
- [ ] 能用 `/proc/PID/fd` 或者 `fuser` 至少一种手段,找出是哪个进程占着一个文件

---

## 7. 收尾实操 —— 拿到一台新 Linux 机器,10 分钟内配好能干活的环境

**为什么需要这个 / 不会有什么后果:**

换实验室新服务器、申请到一个新的云 GPU 实例、甚至只是同一台机器上新开一个账号,这种"拿到一个几乎全新的 Linux 环境"的情况,读研期间会反复发生。如果每次都是"想起一件事、配一件事、缺了什么再现查",不仅慢,还容易配了一半就直接开始干活,过后才发现少了某个环境保护(比如忘记确认 tmux 装了没有,任务直接裸跑在前台,断线就白跑),欠下的债往往在最不合适的时候爆发。这一节把前 6 节内容串成一份**你可以直接照着跑一遍**的检查清单,目标是养成"拿到新机器先花 10 分钟做完这一遍,再开始正式干活"的习惯,而不是每次都临时现想。

下面全程用一个**专门新建、此前完全没配置过任何东西**的 Linux 用户(`freshdemo`)真实跑一遍,验证这份清单本身是可信的,不是空谈。

**环境要求:** 有一个刚拿到、还没配置过的 Linux 账号/机器(任何发行版都适用,具体安装命令可能因发行版而异,这里以 `dnf` 为例,Ubuntu/Debian 系换成 `apt`)。

**一步步跟着做(照着这个顺序做一遍):**

**第 1 步:摸清楚自己在什么环境上。**

```
$ echo $SHELL
/bin/bash
$ bash --version | head -1
GNU bash, version 5.2.26(1)-release (x86_64-redhat-linux-gnu)
$ cat /etc/os-release | head -2
NAME="Rocky Linux"
VERSION="10.2 (Red Quartz)"
$ whoami; pwd; hostname
freshdemo
/home/freshdemo
Eric-12900H-3080Ti
```

30 秒之内知道:什么 shell、什么发行版、当前是谁、在哪、这台机器叫什么——后面所有操作的基本前提。

**第 2 步:确认这篇文章依赖的几个工具都在,不在的现场装上。**

```
$ for cmd in tmux screen grep awk sed du df nohup; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "OK   $cmd -> $(command -v "$cmd")"
    else
      echo "MISSING  $cmd"
    fi
  done
OK   tmux -> /usr/bin/tmux
OK   screen -> /usr/bin/screen
OK   grep -> /usr/bin/grep
OK   awk -> /usr/bin/awk
OK   sed -> /usr/bin/sed
OK   du -> /usr/bin/du
OK   df -> /usr/bin/df
OK   nohup -> /usr/bin/nohup
```

`grep`/`awk`/`sed`/`du`/`df`/`nohup` 几乎所有 Linux 发行版都默认自带;`tmux`/`screen` 不一定,如果上面显示 `MISSING`,现场装:

```
$ dnf install -y tmux
```

**第 3 步:配置最基本的 `.bashrc` 自定义——每次登录都要用到的环境变量/别名,一次性写好。**

```
$ mkdir -p ~/bin
$ cat >> ~/.bashrc <<'EOF'

# --- 新机器初始化时加的 ---
export PATH="$HOME/bin:$PATH"
alias ll='ls -lah'
alias gs='git status'
EOF
$ source ~/.bashrc
```

`~/bin` 用来放你自己写的、以后经常要用的小工具脚本(参考第 2 节的做法);两条别名只是示例,按自己实际习惯增删。**养成把这类自定义写进 `.bashrc` 而不是每次手动 `export` 的习惯**——这正是第 2 节的核心结论:临时设置撑不过下一次登录。

**第 4 步:磁盘空间摸底——开始占用磁盘之前,先知道自己有多少余量。**

```
$ df -h $HOME | tail -1
/dev/sdd       1007G  2.1G  954G   1% /
```

心里有个数,以后第 6 节的 `du -sh */ | sort -rh` 才有一个"相对什么时候开始涨的"基准可以对比。

**第 5 步:开一个默认工作用的 tmux 会话,养成"干活先进 tmux"的习惯,而不是任务跑起来了才想起来。**

```
$ tmux new-session -d -s main
$ tmux list-sessions
main: 1 windows (created Thu Jul 23 08:02:36 2026)
```

以后每次登录这台机器,第一件事是 `tmux attach -t main`(如果断开过)或者继续在里面干活,而不是在裸终端上直接跑长时间任务。

**第 6 步:一次性自测,确认前 5 步都真的生效了。**

```
$ ok=1
$ command -v tmux >/dev/null || { echo "FAIL: tmux missing"; ok=0; }
$ grep -q "新机器初始化时加的" ~/.bashrc || { echo "FAIL: bashrc customization missing"; ok=0; }
$ tmux has-session -t main 2>/dev/null || { echo "FAIL: no main tmux session"; ok=0; }
$ [ "$ok" = "1" ] && echo "ALL CHECKS PASSED - ready to work"
ALL CHECKS PASSED - ready to work
```

**背后发生了什么:**

这一节没有引入任何新机制——纯粹是把第 1-6 节各自验证过的独立操作,按"新机器从零开始"这个真实场景重新组织了一遍执行顺序。之所以强调"顺序",是因为这几步之间有实际的依赖关系:先确认工具都在(第 2 步),`.bashrc` 里如果用到某个工具才有意义;`.bashrc` 配好(第 3 步)之后,`~/bin` 才会被自动加进 `PATH`,以后往这个目录扔脚本才能直接调用;养成先开 tmux 的习惯(第 5 步)放在最后,是因为一旦你在 tmux 会话里开始干活,前面几步如果没做完,再回头补做也不会有任何冲突,不存在"先后顺序搞反了会出问题"的情况——除了"一开始就该做的事,拖到后面做"这种效率损失。

**常见坑:**

| 现象 | 大概率原因 |
|---|---|
| 新机器上敲惯用的别名/脚本,发现"根本没配过" | 忘了 `.bashrc` 是每台机器/每个账号独立的,不会跟着你自动迁移——这就是本节存在的意义,养成"新机器先跑一遍清单"的习惯 |
| 装好 tmux,但已经手滑在裸终端跑起来一个长任务了 | 回第 4 节的补救流程:`Ctrl+Z` 挂起 → `bg` → `disown`,下次提前进 tmux |
| 10 分钟根本不够,清单里有些步骤要等很久(比如 `dnf install` 网络慢) | 正常,"10 分钟"是不含网络安装等待时间的操作部分,网速慢的环境本身没有办法,基本操作的部分应该能压缩到很短 |

**自测清单:**

- [ ] 能不看这篇文章,独立在一台新账号/新机器上,把第 1-6 步跑完一遍
- [ ] 能说出这份清单里每一步分别对应本文前面哪一节的内容
- [ ] 能解释为什么"先确认工具在不在""再配 `.bashrc`""最后开 tmux"这个顺序是有道理的,不是随便排的
- [ ] 拿到一台真实的新服务器时,会不会想起来跑这份清单,而不是直接开始裸敲命令

---

*创建:2026-07-23*
