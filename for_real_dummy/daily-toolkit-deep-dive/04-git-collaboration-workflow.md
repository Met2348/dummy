# Git/GitHub 日常协作实操 —— 分支、commit、冲突、review 手把手教

> 前提:你已经在这个仓库里用过最基础的 `git add`/`git commit`/`git push`(哪怕是学长带着操作的),知道"仓库里能 commit"这件事本身是怎么回事。这篇文章不重新讲这些最基础的单人操作,只补一层——**协作**:多个人(你和学长、你和导师)共用一个仓库时,怎么不把彼此的工作搞乱。
> 目标:看完 + 跟着动手做完文中的真实演示,你应该能做到:敢开分支、commit 写得让 3 个月后的自己看懂、遇到 merge conflict 不慌、误删的 commit 能找回来、知道 PR 和 review 到底在干什么、以及哪些文件天生就不该进 git。

---

## 0. 这篇文章的实操演示是怎么做的(先说清楚)

这篇文章要教的几件事——开分支、制造并解决 merge conflict、用 `git reflog` 找回"丢失"的 commit——都属于"第一次不知道会发生什么"的操作。如果直接拿你现在正在用的这个真实仓库(也就是你打开这篇文章所在的这个仓库)练手,一旦手滑,可能会打乱你和学长共享的真实历史。

所以本文里**所有真实跑过的命令和真实贴出来的输出**,都是在两个和这个仓库完全无关、用完即扔的临时目录里做的:

```
C:\Users\ericp\AppData\Local\Temp\tmp.6oLTWBMxFb\demo-a-no-branches\   ← 对应第 1 节:"不用分支会怎样"
C:\Users\ericp\AppData\Local\Temp\tmp.6oLTWBMxFb\demo-b-workflow\      ← 对应第 2、3、4、5、6 节
```

这两个目录里发生的一切(建仓库、开分支、制造冲突、误删 commit)都和这个仓库(`e:\Workspace\dummy`)**完全隔离**——不是同一个 `.git`,不共享任何历史,删掉这两个临时目录不会对这个仓库产生任何影响。写这篇文章的过程中,每做一步有真实风险的操作之前,都先用 `pwd` 确认自己确实站在临时目录里,不是仓库目录。

**这本身就是值得你现在就养成的习惯:任何你没把握、想先"试一下会发生什么"的 git 操作,找一个临时目录试,不要拿正在用的真实仓库练手。** 临时目录怎么建、要装什么,看下面第 1 节的"环境要求"——之后每一节的真实演示,你都可以在自己电脑上照着重新跑一遍,不会影响这个仓库半个字节。

**读代码块的约定:** 下面的代码块里,以 `$` 开头的行是实际敲的命令,不带 `$` 的是 git 真实打印出来的输出。你自己跟着敲一模一样的命令,应该会看到几乎一样的输出——**除了 commit 的哈希值(比如 `5ca7571` 这种 7 位十六进制)一定会不一样**,这是正常的:哈希是根据内容、作者、时间戳算出来的指纹,你的作者名字和操作时间跟这篇文章不同,算出来的指纹自然不同,不代表你做错了。

---

## 1. 为什么需要分支工作流

### 1.1 为什么需要这个 / 不会有什么后果

先说清楚"分支"解决的是什么问题。

你现在这个仓库,`git log` 里看到的那条主线(通常叫 `main` 分支)代表"大家都认可的、当前能用的版本"——学长会从这条主线开始看代码,导师会从这条主线判断项目进展,你自己下次打开电脑也会默认这条主线是"对的"。**"分支"这个机制存在的意义,就是让你能在不破坏这条主线的前提下,单独开一条线去做还没验证完的改动。**

不用分支会有什么后果?来看一个具体场景:

> 你和 Alice(假设是另一个共同维护这个仓库的人,比如学长)都 clone 了同一个仓库,都直接在 `main` 分支上改代码,不开分支。某天,你们俩**碰巧同时**在改同一个训练配置文件的同一行(比如学习率),各自改完在自己电脑上 commit。Alice 手快,先 `git push`,成功了。你紧接着也 `git push`——会发生什么?

直觉上可能觉得"后 push 的会不会把 Alice 的覆�位掉"，或者反过来"我的改动是不是白做了"。实际发生的是第三种情况,也是 git 在这里最重要的一条保护机制:**你的 push 会被拒绝**,git 不会让你在不知情的情况下覆盖别人已经推上去的历史。下面第 1.3 节会真实演示这个过程,包括 git 拒绝你时打印出来的原文。

但"被拒绝"不等于"没事了"——真正的代价是:

1. **冲突被推迟到最不合适的时间和地点。** 你被拒绝之后必须马上处理:拉取 Alice 的改动、和自己的改动比对、解决冲突——这些都发生在 `main` 分支上,而 `main` 是大家随时可能拉取的分支。你在那手忙脚乱地解决冲突的这段时间,`main` 本身处于一个不上不下的状态。
2. **`main` 不再是"可信"的。** 一旦大家习惯直接在 `main` 上改,`main` 上就会混进各种"改到一半""还没测完"的中间状态,失去了"这是大家都认可的版本"这个前提。
3. **最危险的后果:被拒绝之后,图省事的人会想用 `git push --force` 强推。** `--force` 会无条件用你本地的版本覆盖远程仓库,不会有任何冲突提示——也就是说,Alice 已经推上去的那次 commit,可能会被你的强推**直接、安静地抹掉**,她不会收到任何警告。这才是真正的"数据丢失",而且丢的是别人的工作。这也是为什么这个仓库的协作纪律里反复强调"避免破坏性 git 操作"——`push --force` 就是典型代表。第 6 节会讲清楚,就算真出了这种事故,补救手段是什么、能救回多少。
4. **没有任何人在改动生效之前看过一眼。** 直接推 `main`,意味着没有第二双眼睛在这段代码影响到所有人之前检查过它——这是第 5 节"PR 和 review"要解决的问题。

分支工作流的解法:每个人的"还没验证完"的工作,都放在自己的分支上,不直接触碰 `main`;真正要合并回 `main` 的时候,冲突(如果有的话)在一个明确、可控的时间点被处理,而不是在 push 被拒绝的慌乱中处理。

### 1.2 环境要求

- Git Bash(或任何能跑 git 命令的终端),确认版本:

```
$ git --version
git version 2.38.1.windows.1
```

- 一个和你任何真实项目都无关的临时目录,用来做本节的实验。在 Git Bash 里,`mktemp -d` 会自动建一个绝对不会跟别的东西撞名的临时目录并把路径打印出来:

```
$ mktemp -d
/tmp/tmp.xxxxxxxxxx
```

（本文用的就是这个命令建出来的目录,只是随机后缀不一样。你自己跑一遍会得到一个不同的随机路径,这是正常的。）

### 1.3 一步步跟着做:真实演示"不用分支会发生什么"

**第 0 步:搭一个"迷你 GitHub"。** 真实的协作要有个大家都能访问的远程仓库(GitHub 上的仓库),本地没有真实网络也能模拟出这个角色——用 `git init --bare` 建一个"裸仓库"(bare repository):裸仓库只存 git 的历史数据,没有工作区(没有你能直接编辑的文件),专门用来当"大家共同推送/拉取的中心"。这正是 GitHub 服务器上每个仓库的本质。

```
$ mkdir demo-a-no-branches && cd demo-a-no-branches
$ git init --bare -b main remote.git
Initialized empty Git repository in .../demo-a-no-branches/remote.git/
```

再 clone 出两份"个人工作区",分别代表你和 Alice：

```
$ git clone remote.git alice
Cloning into 'alice'...
warning: You appear to have cloned an empty repository.
done.
```

（这条 warning 正常——因为这时候 `remote.git` 里还什么都没有,是空仓库。）

**第 1 步:Alice 先建好项目、推上去。**

```
$ cd alice
$ git config user.name "Alice"
$ git config user.email "alice@example.com"
$ printf 'learning_rate: 0.001\nbatch_size: 32\n' > train_config.yaml
$ git add train_config.yaml
$ git commit -m "Add initial training config"
[main (root-commit) 5ca7571] Add initial training config
 1 file changed, 2 insertions(+)
$ git push -u origin main
branch 'main' set up to track 'origin/main'.
To .../remote.git
 * [new branch]      main -> main
```

**第 2 步:你(Bob)加入项目,clone 下来。**（这一步用了另一个人名 Bob 代表"你",纯粹是为了在下面的命令输出里跟 Alice 的操作区分开。）

```
$ cd ..
$ git clone remote.git bob
Cloning into 'bob'...
done.
$ cd bob
$ git config user.name "Bob"
$ git config user.email "bob@example.com"
$ cat train_config.yaml
learning_rate: 0.001
batch_size: 32
```

**第 3 步:你们俩碰巧同时改了同一行,谁都不知道对方在改。** 你先改完,commit,但还没来得及 push：

```
$ sed -i 's/learning_rate: 0.001/learning_rate: 0.0005/' train_config.yaml
$ git add train_config.yaml
$ git commit -m "Lower learning rate to fix loss divergence"
[main 00ac511] Lower learning rate to fix loss divergence
 1 file changed, 1 insertion(+), 1 deletion(-)
```

与此同时,Alice 也改了同一行(改成另一个值),而且她手快,commit 完立刻 push：

```
$ cd ../alice
$ sed -i 's/learning_rate: 0.001/learning_rate: 0.01/' train_config.yaml
$ git add train_config.yaml
$ git commit -m "Try higher learning rate for faster convergence"
[main c81ab3b] Try higher learning rate for faster convergence
 1 file changed, 1 insertion(+), 1 deletion(-)
$ git push origin main
To .../remote.git
   5ca7571..c81ab3b  main -> main
```

Alice 推成功了。**关键的一步来了**——你现在也去 push：

```
$ cd ../bob
$ git push origin main
To .../remote.git
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to '.../remote.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

**这就是"不用分支,两个人同时改同一个仓库"真实会发生的事情**——不是静默覆盖,而是被拒绝。git 老老实实告诉你:远程有你本地没有的改动(`the remote contains work that you do not have locally`),并建议你先 `git pull`。

按提示做,拉取一下：

```
$ git pull origin main
From .../remote
   5ca7571..c81ab3b  main       -> origin/main
Auto-merging train_config.yaml
CONFLICT (content): Merge conflict in train_config.yaml
Automatic merge failed; fix conflicts and then commit the result.
```

看到了:**没有分支,并不会让你逃过冲突**——冲突照样发生,只是现在发生在你自己已经被拒绝、心态更慌张的时刻,而且是直接发生在 `main` 上,`git status` 会告诉你此刻 `main` 处于一个两边都改过、尚未合并完成的状态：

```
$ git status
On branch main
Your branch and 'origin/main' have diverged,
and have 1 and 1 different commits each, respectively.
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)
Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   train_config.yaml
```

这个冲突具体要怎么读、怎么解决,和用分支时遇到的冲突,处理方式完全一样——这个先卖个关子,第 3 节会用一个更干净的例子把这一步慢慢拆开讲。这里想让你看到的重点是:**"不用分支"并没有让协作变得更简单,只是让本该被隔离处理的冲突,被迫在最公共、最没有退路的分支上处理。**

### 1.4 背后发生了什么

- **push 为什么会被拒绝:** 默认情况下,git 允许你 push 的前提是"你本地的提交历史是远程当前状态的直接延伸"(这叫 **fast-forward**,快进——想象成远程的指针只是顺着一条直线往前挪了一格,没有绕路)。Alice push 之后,远程的 `main` 已经不再是你 clone 下来时的那个点了;你本地的 `main` 是从"旧的远程状态"往下延伸的,不是"新的远程状态"的直接延伸,所以 git 拒绝直接接受,怕的就是你在不知情的情况下把 Alice 的工作从远程历史里挤掉。这是一种保护,不是 bug。
- **`git pull` 做了两件事:** 先 `git fetch`(把远程的最新状态下载下来,但不改动你当前的文件),再自动执行一次 `git merge`(把远程的改动和你本地的改动合并)。合并这一步,如果两边改了同一处,就是冲突的来源——这和第 3 节要讲的"合并两个分支"在机制上是同一件事,唯一的区别是这里合并的对象是"你的本地 main"和"远程的 main",而不是两个显式建出来的分支。
- **分支本身是什么(提前预告,第 3 节会再展开):** 一个分支,本质上只是一个"指向某个 commit 的、可以移动的名字"——不是文件的复制品。`git branch xxx` 几乎是瞬间完成的,因为它不需要复制任何文件,只是新建一个指针。正因为分支这么"廉价",Alice 和 Bob 才应该各自开自己的分支去改,谁的指针指向哪里互不影响,直到显式地合并那一刻才需要面对冲突。

### 1.5 常见坑

| 现象 | 原因 | 怎么办 |
|---|---|---|
| push 被拒绝,提示 `rejected` / `fetch first` / `non-fast-forward` | 远程有你本地没有的提交(常见于没用分支、大家都往 `main` 直接提交) | 先 `git pull`(或 `git fetch` + `git merge`),处理完可能出现的冲突再 push,不要 `--force` |
| 忍不住想用 `git push --force` 让 push "过去" | 图省事,没弄懂拒绝提示的含义 | 别用。它会无条件覆盖远程,可能安静地丢掉别人的提交,没有任何警告。自己独占的分支(比如只有你在用的实验分支)真要强推,`--force-with-lease` 相对安全一些(推送前会先确认远程没有被别人动过),但这已经不是日常协作该养成的习惯 |
| 不确定该不该新建分支,干脆一直在 `main` 上改 | 没意识到"能跑"和"该被所有人依赖"是两件事 | 只要是"还没验证完/还想改"的状态,就该在分支上,不该在 `main` 上 |
| 只有自己一个人用这个仓库,觉得分支没必要 | 分支的价值被简化成了"多人协作专用" | 单人也一样受益:一个分支专门放"能跑的版本",另一个分支放"正在打的补丁",出问题的时候至少还有一个能退回去的干净状态 |

### 1.6 自测清单

- [ ] 能说清楚:Alice 和 Bob 都直接在 `main` 上改同一行,Bob 后 push 会发生什么(是"被拒绝",不是"被静默覆盖")
- [ ] 能解释为什么"被拒绝"其实是 git 在保护你,以及为什么 `git push --force` 会把这层保护绕过去
- [ ] 知道"分支"本质上是什么(一个会移动的、指向某个 commit 的名字),而不是"文件的复制"
- [ ] 能在自己电脑上,用一个 bare 仓库 + 两个 clone,重复一遍本节的演示,亲眼看到同样的 `rejected`

---

## 2. commit 粒度与写好的 commit message

### 2.1 为什么需要这个 / 不会有什么后果

先明确一个概念:一次 **commit** 是仓库历史里的一个"存档点"——它记录了这次改动涉及哪些文件的哪些内容变化、是谁在什么时候做的、以及一句你自己写的说明(commit message)。`git log` 看到的那条历史,就是这些存档点按时间顺序串成的链条。

**commit 的"粒度"(大小)选得不对,会有两种相反的后果:**

- **太大:** 把一整天(甚至几天)攒的、好几个不相关的改动打包成一个 commit。后果是:这个 commit 没法单独撤销(`git revert`)——因为撤销它会把好几件不相关的事一起撤销掉;`code review` 没法针对一件事去看,审稿人要在一堆无关改动里自己拆分哪部分对应哪个意图;将来用 `git log`/`git blame` 追查"这一行是什么时候、为什么变成这样"的时候,答案会混着好几件不相关的事,等于没有答案。
- **太小:** 改一个字符就 commit 一次。后果是:历史被"wip"“fix”“再改一点”这种噪音淹没,真正有意义的节点被埋没,review 一个包含几十个这种碎 commit 的分支,比 review 一个干净的大 commit 还累。

**合适的粒度是:一次 commit = 一个逻辑上完整、可以独立被理解的改动。** 一个简单的自检方法:如果我现在把这个 commit `git revert` 掉,仓库会不会回到一个说得通的状态?如果撤销它会连带撤销 3 件你还想保留的事,这个 commit 就是太大了;如果这个改动本身还没构成一件"完整的事"(比如函数写了一半),那还不到 commit 的时候。

**commit message 同样重要,而且检验标准很具体:** 3 个月后,如果训练 loss 突然变成 NaN,你打开 `git log --oneline` 想找"最近哪次改动碰过训练相关的代码"——一条写得好的 message,能让你不用一个个打开 diff 就大致定位到嫌疑 commit;一条写成 `"update"`/`"fix"`/`"1"` 的 message,等于这条历史记录没有提供任何信息,你只能挨个打开 diff 硬看。**"未来排查 bug 时,`git log` 能不能看懂",就是判断一条 commit message 好不好的标准**,不是"是否符合某种格式"。

下面用真实操作对比这两种情况。

### 2.2 环境要求

同 1.2:Git Bash + git 2.38.1,一个临时目录。这里继续沿用一个新的临时仓库(与第 1 节的 `demo-a-no-branches` 无关),下文第 3-6 节也会一直复用这一个仓库,让整篇文章像一个连续的小项目。

### 2.3 一步步跟着做

先搭一个最小的"训练脚本"项目(内容是示意用的极简伪代码,不追求真的能跑,只是为了让下面的 git 操作有实际的文件可改):

```
$ mkdir demo-b-workflow && cd demo-b-workflow
$ git init -b main
$ git config user.name "You"
$ git config user.email "you@example.com"
```

建三个文件:`config.yaml`(超参数)、`train.py`(训练脚本骨架)、`README.md`(项目说明,故意留一个拼写错误 `minimall`,后面会用到)：

```
$ git add config.yaml train.py README.md
$ git commit -m "Initial commit: skeleton training script"
[main (root-commit) 611c941] Initial commit: skeleton training script
 3 files changed, 9 insertions(+)
```

**先演示"太大"的反面教材:** 假设你同时想做三件不相关的事——给训练加梯度裁剪(防止 loss 炸成 NaN)、把 batch size 从 16 调到 8(显存不够)、顺手把 README 里的拼写错误改掉。图省事,一次性全改了,一个 commit 搞定,message 随手写：

```
$ git diff --stat
 README.md   | 2 +-
 config.yaml | 2 +-
 train.py    | 1 +
 3 files changed, 3 insertions(+), 2 deletions(-)
$ git add -A
$ git commit -m "update"
[main 0314e9e] update
 3 files changed, 3 insertions(+), 2 deletions(-)
```

现在假装自己是"3 个月后的你",想知道这个 commit 干了什么：

```
$ git show --stat HEAD
commit 0314e9edcefcda1353ab828711bf2e111327303a
Author: You <you@example.com>
Date:   Thu Jul 23 03:17:53 2026 +0800

    update

 README.md   | 2 +-
 config.yaml | 2 +-
 train.py    | 1 +
 3 files changed, 3 insertions(+), 2 deletions(-)
```

**看到问题了:** 你只知道"三个文件被改了",message 是 `update`,没有提供任何"为什么"的信息。你得自己打开完整 diff,逐行猜每处改动是为了什么、彼此有没有关系。

**现在演示正确做法:撤销这个 commit,拆成三个。** 用 `git reset HEAD~1`(不带 `--hard`)——它会把 `main` 指针退回上一个 commit,但**保留工作区里的文件改动**,只是变成"未暂存"状态：

```
$ git reset HEAD~1
Unstaged changes after reset:
M	README.md
M	config.yaml
M	train.py
$ git status
On branch main
Changes not staged for commit:
	modified:   README.md
	modified:   config.yaml
	modified:   train.py
```

三处改动都还在,只是从"已经封存成一个 commit"退回成"还没提交"。现在分别提交,每次只 `git add` 一个文件,并且写清楚"为什么":

```
$ git add train.py
$ git commit -m "Add gradient clipping to prevent loss explosion

Loss occasionally spikes to NaN after ~200 steps on the current
config. Clipping the gradient norm to 1.0 keeps updates bounded
without changing the optimizer."
[main 54c4ecf] Add gradient clipping to prevent loss explosion

$ git add config.yaml
$ git commit -m "Reduce batch size to 8 to fit in GPU memory

16 was OOMing on the lab workstation's 8GB card after the model
grew in the previous change."
[main 57a3110] Reduce batch size to 8 to fit in GPU memory

$ git add README.md
$ git commit -m "Fix typo in README (minimall -> minimal)"
[main bcb0d1d] Fix typo in README (minimall -> minimal)
```

现在再看历史,对比一下"3 个月后的你"这次能看懂多少：

```
$ git log --oneline
bcb0d1d Fix typo in README (minimall -> minimal)
57a3110 Reduce batch size to 8 to fit in GPU memory
54c4ecf Add gradient clipping to prevent loss explosion
611c941 Initial commit: skeleton training script
```

单看标题就已经说清楚"改了什么";带上 body 的那两条,连"为什么"都写明白了——`git log`(不加 `--oneline`)能看到完整内容：

```
commit 57a3110...
    Reduce batch size to 8 to fit in GPU memory

    16 was OOMing on the lab workstation's 8GB card after the model
    grew in the previous change.

commit 54c4ecf...
    Add gradient clipping to prevent loss explosion

    Loss occasionally spikes to NaN after ~200 steps on the current
    config. Clipping the gradient norm to 1.0 keeps updates bounded
    without changing the optimizer.
```

同样三处改动,拆开之后,每一条都可以单独 `git revert`,单独 review,单独在 `git log` 里读懂——这就是"合适粒度"实际带来的差别。

> 另一个极端("太小")这里不专门演示了——想象一下把上面三处改动拆成十几个 commit,每个 commit 只改一两个字符,`git log --oneline` 会被 `"fix"`"再改一点""typo"这类占满,真正有意义的三条会被淹没在里面,效果不会比拆开来的一个 `"update"` 好多少。太大和太小是同一个问题的两个方向:**commit 的边界应该由"这是不是一件独立、完整的事"决定,不是由时间(一天一个)或者字符数(改一点就提交)决定。**

### 2.4 背后发生了什么

- **`git reset HEAD~1`(不带 `--hard`)做了什么:** git 的 reset 有三种模式,这里用的是默认的 `--mixed`。它只挪动分支指针(让 `main` 重新指向上一个 commit),同时把原来那个 commit 里的改动内容,从"暂存区"(index/staging area,也就是 `git add` 之后、`git commit` 之前的那个中间区域)退回到"工作区"(working directory,也就是你能直接看到、编辑的文件)。文件本身的内容完全没变,只是"打包成一个 commit"这件事被撤销了。这和 `--hard`(工作区内容也会被强制改写,真的会丢东西)是完全不同的两种操作——第 6 节会详细讲 `--hard` 的风险。
- **为什么要拆开分别 `git add`:** `git add <file>` 只把指定文件当前的内容放进暂存区,不影响其他文件。分别 `add` 再分别 `commit`,就是在人为地把一次性的"一堆改动"重新切成"多个独立的存档点"。
- **好的 message 格式为什么长这样(一行标题 + 空行 + 正文):** 这是 git 生态里通用的约定,不是这篇文章发明的——`git log --oneline`、GitHub 的 PR 列表、大多数 git 图形工具,都只显示第一行作为"标题";正文只有在你主动展开时才会看到。所以第一行要能独立说清楚"做了什么"(祈使句,比如 `"Add X"` 而不是 `"Added X"`,想象成在下命令"执行这个改动"),正文留给"为什么"——diff 本身已经能看出"做了什么",message 正文的价值在于回答 diff 回答不了的"为什么"。

### 2.5 常见坑

| 现象 | 原因 | 怎么办 |
|---|---|---|
| commit message 写成 `"update"`/`"fix"`/`"WIP"`/`"1"` | 图快,没多想 | 提交前问自己一句:"如果我 revert 这个 commit,别人看 log 能不能猜到是为了什么" |
| 一个 commit 改了看起来不相关的好几个文件 | 把一段时间的工作攒到一起才提交 | 提交前先 `git status`/`git diff` 看一眼要提交的范围,不相关的改动分开 `git add` |
| 想把同一个文件里"部分改动"分批提交,但 `git add <file>` 是整个文件一起暂存 | 不知道还有更细粒度的暂存方式 | `git add -p` 可以按代码块(hunk)交互式选择要暂存的部分,这里不展开,先知道它存在,需要的时候再查 |
| commit 数量特别多,一个小改动反复 commit 好几次 | 把"存个盘"和"commit"混为一谈 | commit 前先想清楚"这是不是一件完整的小事";还没写完可以先不提交,或者用 `git commit --amend`(仅限还没 push 出去的最后一个本地 commit)去补充,而不是新开一堆碎 commit |

### 2.6 自测清单

- [ ] 能说出评价"一个 commit 大小是否合适"的标准(不是行数或时间,是"是不是一件独立、完整的改动")
- [ ] 能写出一条"一行祈使句标题 + 空行 + 说明为什么"的 commit message
- [ ] 知道 `git reset HEAD~1`(不加 `--hard`)和 `git reset --hard HEAD~1` 的核心区别是什么
- [ ] 拿自己仓库里最近 5 条 `git log --oneline`,能诚实评价:3 个月后回来看,这几条 message 够不够用

---

## 3. 真实 merge conflict 怎么读懂并解决

这是这篇文章最重要的一节。下面的冲突不是编出来的,是在写这篇文章的过程中真实触发、真实解决的——包括冲突标记的原始内容,一个字符都没有改动过。

### 3.1 为什么需要这个 / 不会有什么后果

**冲突(conflict)发生的条件很具体:两个分支,从同一个共同祖先出发,各自改了同一个文件的同一处内容,但改成了不一样的东西。** git 能自动处理"两边改的是文件不同地方"的情况(直接把两边的改动都合并进来),但没法替你决定"同一处地方,到底该听谁的"——所以它停下来,把决定权交给你。

**这不是你操作失误的信号,是协作里正常会发生的事**,单人开发只要同时维护两个分支也一样会遇到。真正的风险不是冲突本身,而是不知道怎么读冲突标记时的两种错误反应:一是慌张之下随便删一通、甚至删掉整个仓库重新 clone,可能把还没推送的工作弄丢;二是不理解标记的含义就随手保留一边、删掉另一边,可能悄悄丢掉队友本来是对的那部分改动。还有一种更隐蔽的失误:解决完之后忘了把 `<<<<<<<`/`=======`/`>>>>>>>` 这三种标记本身删干净,导致这些标记原样被提交进了历史,文件从此损坏。

学会读懂冲突标记、按部就班地解决,这几种风险基本就不存在了。

### 3.2 环境要求

同前面几节:Git Bash + git 2.38.1 + 临时目录。这里继续用第 2 节的 `demo-b-workflow` 仓库。

（如果你在 VSCode 里打开一个有冲突的文件,VSCode 会给冲突区域上色,并且在旁边提供"Accept Current/Incoming/Both"这样的图形化按钮——原理和这里手动读命令行输出完全一样,只是操作更方便。这篇文章为了让你先看懂原始格式,全部用命令行演示;VSCode 里具体长什么样,以后 `01-vscode-editor-workflow.md` 写到的时候会补上截图说明。）

### 3.3 一步步跟着做

**第 1 步:从同一个起点,开两个分支,各自改同一行。** 这时 `config.yaml` 里的 `learning_rate: 0.001` 这一行,从第 2 节到现在还没被动过,是干净的:

```
$ cat config.yaml
learning_rate: 0.001
batch_size: 8
```

开第一个分支,把学习率调高：

```
$ git checkout -b feature/higher-lr
Switched to a new branch 'feature/higher-lr'
$ sed -i 's/learning_rate: 0.001/learning_rate: 0.01/' config.yaml
$ git add config.yaml
$ git commit -m "Increase learning rate to 0.01 to speed up convergence"
[feature/higher-lr 804160f] Increase learning rate to 0.01 to speed up convergence
```

回到 `main`,再开第二个分支(注意:也是从 `main` 这个起点开出去的,不是从 `feature/higher-lr` 继续改),把学习率调低：

```
$ git checkout main
$ git checkout -b feature/lower-lr
Switched to a new branch 'feature/lower-lr'
$ sed -i 's/learning_rate: 0.001/learning_rate: 0.0001/' config.yaml
$ git add config.yaml
$ git commit -m "Lower learning rate to 0.0001 to stabilize training near convergence"
[feature/lower-lr da0197a] Lower learning rate to 0.0001 to stabilize training near convergence
```

此刻的历史长这样——两个分支从同一点分叉,各自往前走了一步:

```
$ git log --oneline --all --graph
* 804160f Increase learning rate to 0.01 to speed up convergence
| * da0197a Lower learning rate to 0.0001 to stabilize training near convergence
|/
* bcb0d1d Fix typo in README (minimall -> minimal)
...
```

**第 2 步:先合并第一个分支——不会冲突。** 因为 `main` 从分叉之后什么都没变,`feature/higher-lr` 相对 `main` 是直接往前延伸,git 只需要挪动指针(fast-forward):

```
$ git checkout main
$ git merge feature/higher-lr
Updating bcb0d1d..804160f
Fast-forward
 config.yaml | 2 +-
```

**第 3 步:再合并第二个分支——真实冲突发生。** 现在 `main` 已经不是 `feature/lower-lr` 分叉时的那个起点了(`main` 已经带上了 `0.01` 这个改动),而 `feature/lower-lr` 也改了同一行:

```
$ git merge feature/lower-lr
Auto-merging config.yaml
CONFLICT (content): Merge conflict in config.yaml
Automatic merge failed; fix conflicts and then commit the result.
```

`git status` 会告诉你哪些文件卡在冲突状态：

```
$ git status
On branch main
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)
Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   config.yaml
```

打开这个文件,看真实的冲突标记(下面这段是 `cat config.yaml` 的原始输出,一个字符没改):

```
<<<<<<< HEAD
learning_rate: 0.01
=======
learning_rate: 0.0001
>>>>>>> feature/lower-lr
batch_size: 8
```

**怎么读这段标记:**

- `<<<<<<< HEAD` 到 `=======` 之间:**你当前所在分支**(`HEAD`,这里是 `main`,而且 `main` 已经先合并过 `feature/higher-lr` 了)的版本——`learning_rate: 0.01`
- `=======` 这一行本身:纯分隔线,不是任何一边的数据
- `=======` 到 `>>>>>>> feature/lower-lr` 之间:**你正在合并进来的那个分支**的版本——`learning_rate: 0.0001`
- 没有被 `<<<<<<<` 包住的行(比如这里的 `batch_size: 8`):说明这一行两边没有冲突,git 已经自动处理好了,不用你管

**第 4 步:决定要哪个版本,手动改成最终结果。** 假设经过判断,决定用较低的学习率(比如查过组里的约定,或者这只是个示例决定),把整个文件改写成你想要的最终内容,**三种标记全部删掉**：

```
$ cat > config.yaml <<'EOF'
learning_rate: 0.0001
batch_size: 8
EOF
```

（这里选的是"二选一",但完全不是必须二选一——你也可以在这个区域写第三个你自己综合出来的版本,git 不限制。）

**第 5 步:标记为已解决,提交。**

```
$ git add config.yaml
$ git status
On branch main
All conflicts fixed but you are still merging.
  (use "git commit" to conclude merge)
Changes to be committed:
	modified:   config.yaml
$ git commit --no-edit
[main dc5f96a] Merge branch 'feature/lower-lr'
```

（`--no-edit` 只是让这次演示不弹出编辑器、直接用 git 自动生成的默认合并说明;正常操作里直接 `git commit` 也可以,会弹出一个预填好默认信息的编辑器让你确认或修改。**`git commit` 和 `git merge --continue` 在这里是等价的**——`git merge --continue` 就是"解决完冲突后继续未完成的合并"的专用写法,效果和直接 `git commit` 一样,只是它不接受额外参数,比如试着运行 `git merge --continue --no-edit` 会直接报错 `fatal: --continue expects no arguments`,老老实实只打 `git merge --continue` 就行。)

合并完成后的历史,能清楚看到两条分支重新汇合的那一点:

```
$ git log --oneline --graph --all
*   dc5f96a Merge branch 'feature/lower-lr'
|\
| * da0197a Lower learning rate to 0.0001 to stabilize training near convergence
* | 804160f Increase learning rate to 0.01 to speed up convergence
|/
* bcb0d1d Fix typo in README (minimall -> minimal)
...
```

### 3.4 背后发生了什么

- **git 怎么判断"哪里冲突":** 合并时 git 实际在比较三个版本:两个分支各自的最新提交,以及它们的**共同祖先**(这里是 `bcb0d1d`,那时 `learning_rate` 还是 `0.001`)。对文件的每一处,如果只有一边相对共同祖先改过,git 直接采用那一边的结果(这就是为什么 `batch_size: 8` 没有出现在冲突标记里——两边都没碰这一行,或者只有一边碰过,能自动处理);只有当**两边相对共同祖先都改了,且改成了不一样的东西**,才会产生冲突标记。这种"对比三个点"的方式叫三方合并(three-way merge)。
- **`HEAD` 在冲突标记里指的是谁:** 你站在哪个分支上执行 `git merge`,`HEAD` 就代表那个分支当前的版本;标记里 `>>>>>>>` 后面写的名字,是你传给 `git merge` 的那个分支名(这里是 `feature/lower-lr`)。谁在前谁在后不是随意的,是"我自己 vs 我要合并进来的对方"。
- **`git add` 在冲突时的额外含义:** 平时 `git add` 只是"把这个文件的当前内容放进暂存区,准备提交"。但在冲突状态下,`git add <file>` 还多了一层意思——告诉 git "这个文件我已经处理完了,不再是 unmerged 状态"。`git status` 里 "Unmerged paths" 的列表会随着你逐个 `git add` 而清空。
- **为什么这次合并产生的 commit(`dc5f96a`)比较特殊:** 普通 commit 只有一个"父提交"(parent),但合并两条分支产生的这个 commit 有**两个父提交**(`804160f` 和 `da0197a`)——这是 git 记录"这里是两条历史线重新汇合的地方"的方式,`git log --graph` 里那个从两条线合成一条线的图形,画的就是这个结构。

### 3.5 常见坑

| 现象 | 原因 | 怎么办 |
|---|---|---|
| commit 完之后发现文件里还留着 `<<<<<<<`/`=======`/`>>>>>>>` | 解决冲突时手滑,只改了内容忘了删标记 | 提交前搜一下文件里还有没有这几种标记(比如 `grep -n "<<<<<<<" 文件名`),养成习惯;已经 commit 了就再改一次、重新提交修复 |
| 冲突文件里只有一小段被标记,其他地方看起来是自动处理好的 | 正常现象——不是整份文件都冲突,只有两边都改了同一处才会冲突 | 不用紧张,没被 `<<<<<<<` 包住的内容就是已经处理好的,只需要处理被标记的那几段 |
| 改到一半发现自己判断不了该留哪边,或者觉得这次合并本身就不该现在做 | 想先退出来,冷静一下再决定 | `git merge --abort`——回到执行 `git merge` 之前的状态,不会留下任何痕迹,可以重新想清楚再来 |
| `git status` 一直提示 unmerged,`git commit` 好像没反应 | 还有文件没有 `git add` | `git status` 会列出所有 `both modified` 的文件,一个个确认都改完、都 `add` 了 |
| 一次冲突涉及好几个文件,不知道从哪下手 | 想一次看完全局反而更乱 | 先用 `git status` 看清楚一共几个文件冲突,一个个来,不要跳着改 |
| 觉得冲突标记的两边都不对,真正想要的是第三种写法 | 以为只能"二选一" | 完全可以不选任何一边,自己在冲突区域写一个新版本,git 不限制你必须原样选 HEAD 或对方 |

### 3.6 自测清单

- [ ] 拿到一段冲突标记,能马上说出哪部分是"我这边(`HEAD`)"、哪部分是"对方"
- [ ] 知道解决完之后的完整动作顺序:改内容 → 删掉三种标记 → `git add` → `git commit`(或者等价的 `git merge --continue`)
- [ ] 知道卡住了可以用 `git merge --abort` 安全撤退,而不是硬着头皮乱改
- [ ] 能解释为什么这次合并产生的 commit 有两个父提交,而平时的 commit 只有一个
- [ ] 能在自己电脑的临时仓库里,从"建两个分支"到"合并出冲突"到"解决并提交",不看这篇文章完整重复一遍

---

## 4. `.gitignore` 与大文件的正确处理

### 4.1 为什么需要这个 / 不会有什么后果

git 会追踪每一个被你 `git add` 过的文件的每一次变化,而且**历史是只增不减的**——就算你后来删掉某个文件,它的旧版本依然完整地留在历史记录里,仓库体积不会自动瘦身。这意味着,一旦某类文件第一次被错误地 `add` 进去,后面每次改动都会被继续记录,越滚越大。

哪些文件不该进 git,以及各自的理由:

- **虚拟环境(`.venv/`、`venv/`、conda 环境目录):** 体积通常几百 MB 到几 GB,而且是"能从 `requirements.txt`/`environment.yml` 之类的清单在任何机器上重新生成"的东西——进 git 纯粹是浪费空间,还经常绑定着你这台机器的操作系统和绝对路径,换一台机器很可能直接用不了。
- **`__pycache__/`、`*.pyc`:** Python 解释器自动生成的字节码缓存,每次运行代码都可能重新生成,不是源代码,没有任何值得追踪的信息,而且不同 Python 版本生成的还互不兼容。
- **数据集、模型 checkpoint(`*.pt`、`*.ckpt`、`*.safetensors` 等):** 体积可能从几百 MB 到几十 GB。git 的设计前提是"文本文件、体积小、经常需要看出精确改了哪几行"——塞进大体积二进制文件,会让 `clone`/`pull` 越来越慢,而且 diff 对二进制文件基本没有意义(git 没法告诉你"这个模型权重文件改了哪一行",只能整份重新存一次)。

> 这类"大文件/模型权重/数据集到底该怎么系统性地管理版本"是个更大的话题(比如 Git LFS、专门的对象存储、Hugging Face Hub 这类工具),这篇文章不展开——`pretraining-infra-deep-dive` 系列有更系统的处理。**这里只讲 git 层面最基本的规矩:不要让这类文件第一次就被 `git add` 进去。**

后果如果没处理好:仓库越 `clone` 越慢(尤其是"新人第一次拉取这个仓库"这种场景,可能要多下载几个 GB 的历史垃圾);`.venv` 里可能混进这台机器上的绝对路径甚至密钥之类不该公开的信息;队友的 checkpoint 文件互相覆盖,产生毫无意义的"冲突"(这种文件天生没法像文本一样合并)。

### 4.2 环境要求

同前面几节。不需要额外安装任何东西,能在 Git Bash 里建文件夹和文件即可。继续用 `demo-b-workflow` 这个仓库。

### 4.3 一步步跟着做

**第 1 步:先真实制造一次"手滑"——把不该进 git 的东西提交了。**

```
$ mkdir -p .venv/lib && echo "fake venv binary content" > .venv/lib/site.py
$ mkdir -p __pycache__ && echo "fake bytecode" > __pycache__/train.cpython-313.pyc
$ echo "fake checkpoint tensor data" > model_checkpoint.pt
$ git status --short
?? .venv/
?? __pycache__/
?? model_checkpoint.pt
$ git add .
$ git commit -m "Add training artifacts"
[main 236879c] Add training artifacts
 3 files changed, 3 insertions(+)
```

现在这三类东西已经被 git 追踪了:

```
$ git ls-files
.venv/lib/site.py
README.md
__pycache__/train.cpython-313.pyc
config.yaml
model_checkpoint.pt
train.py
```

**第 2 步:意识到问题,建 `.gitignore`。**

```
$ cat > .gitignore <<'EOF'
# Python virtual environment
.venv/

# Python bytecode cache
__pycache__/
*.pyc

# model checkpoints / weights
*.pt
EOF
```

`.gitignore` 语法很简单:一行一条规则,`#` 开头是注释;`路径/` 表示忽略这个名字的整个目录(不管它出现在仓库哪个位置);`*.扩展名` 是通配符,匹配所有这个后缀的文件。

**第 3 步(容易踩的坑):加了 `.gitignore` 之后,看看这三个文件的状态有没有变化。**

```
$ git status --short
?? .gitignore
```

**注意:** 那三个已经被追踪的路径(`.venv/`、`__pycache__/`、`model_checkpoint.pt`)完全没有出现在这次输出里——**`.gitignore` 对已经被 git 追踪的文件不起任何作用**,它只影响"还没被 `add` 过"的文件。这是最容易搞错的一点。

**第 4 步:真正把它们从追踪里移除——`git rm --cached`。**

```
$ git rm -r --cached .venv __pycache__ model_checkpoint.pt
rm '.venv/lib/site.py'
rm '__pycache__/train.cpython-313.pyc'
rm 'model_checkpoint.pt'
$ git status --short
D  .venv/lib/site.py
D  __pycache__/train.cpython-313.pyc
D  model_checkpoint.pt
?? .gitignore
```

`-r` 表示递归(因为 `.venv`、`__pycache__` 是目录,不是单个文件,类似 shell 里 `rm -r` 要处理目录也得加 `-r`);**关键是 `--cached`——它只把文件从 git 的追踪索引里移除,不会删除硬盘上的真实文件。**（如果漏了 `--cached`,就变成普通 `git rm`,会真的把文件从硬盘删掉,这个区别下面"常见坑"会再强调一次。）

**第 5 步:提交这次修复。**

```
$ git add .gitignore
$ git commit -m "Stop tracking .venv, __pycache__, and checkpoint files

These were committed by accident. They still exist on disk (see
.gitignore) but git no longer tracks them, so future changes to
them won't show up in git status."
[main 2347451] Stop tracking .venv, __pycache__, and checkpoint files
 4 files changed, 9 insertions(+), 3 deletions(-)
```

**第 6 步:验证——文件还在硬盘上,但 git 不再关心它们了。**

```
$ ls .venv/lib __pycache__
.venv/lib:
site.py
__pycache__:
train.cpython-313.pyc

$ git ls-files
.gitignore
README.md
config.yaml
train.py
```

文件确实还在(`ls` 能看到),但 `git ls-files`(列出所有被追踪的文件)里已经没有它们了。再验证一下"以后新产生的同类文件,会不会被自动忽略"：

```
$ echo "more fake bytecode" > __pycache__/other.pyc
$ git status --short
（没有任何输出——新文件被自动忽略了,git status 干干净净）
```

也可以用 `git check-ignore -v` 直接问 git "这个路径到底有没有被忽略、是被哪条规则忽略的"：

```
$ git check-ignore -v __pycache__/other.pyc
.gitignore:5:__pycache__/	__pycache__/other.pyc
$ git check-ignore -v .venv/lib/site.py
.gitignore:2:.venv/	.venv/lib/site.py
$ git check-ignore -v train.py
（没有输出,退出码是 1——说明 train.py 没有被忽略,是正常追踪的文件）
```

输出里 `.gitignore:5:` 的意思是"这条判断来自 `.gitignore` 文件第 5 行的规则",正好对应上面写的 `__pycache__/` 那一行。

### 4.4 背后发生了什么

- **`.gitignore` 只影响"未追踪"状态:** git 判断一个文件要不要出现在 `git status` 里、要不要被 `git add .` 这种通配操作扫进去,靠的是"这个文件当前有没有被追踪"。`.gitignore` 的作用范围严格限定在"还没被追踪的文件":对于已经在追踪列表里的文件,git 认为你之前已经明确决定要追踪它了,不会因为后来加了一条忽略规则就自动反悔。这也是为什么第 3 步那次 `git status --short` 里,那三个已追踪的路径完全没有反应。
- **`--cached` 操作的是"索引",不是"工作区":** git 内部有两份和文件相关的记录——工作区(你能直接看到、编辑的文件)和暂存区/索引(index,记录"下次 commit 会包含哪些内容")。`git rm --cached` 只从索引里删除这条记录(相当于告诉 git "下次提交起,不要再管这个文件了"),完全不碰工作区里的真实文件。这和第 2 节提到的"工作区/暂存区"是同一套机制。
- **忽略规则是怎么匹配到目录里新文件的:** `.gitignore` 里写的 `__pycache__/` 这条规则,匹配的是"这个名字的目录本身",目录一旦被忽略,里面不管以后新增多少文件,都会被一并忽略,不需要针对每个新文件单独写规则——这就是为什么后来新建的 `other.pyc` 不用额外操作就自动被忽略了。

### 4.5 常见坑

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 写了 `.gitignore`,但 `git status` 里那些文件还在 | 文件在写 `.gitignore` 之前就已经被 `add`/`commit` 过 | `git rm -r --cached <路径>`,再提交 |
| 用了 `git rm --cached` 之后,文件从硬盘上真的消失了 | 忘了加 `--cached`,变成了普通 `git rm`(会真删文件) | 记得加 `--cached`;真删了的话,如果之前 commit 过,可以用 `git checkout <某个还包含它的 commit> -- <路径>` 找回,或者参考第 6 节 `git reflog` 的思路 |
| checkpoint/数据集已经反复 commit 了很多次,仓库体积巨大 | 历史记录里留着所有旧版本的大文件 | `git rm --cached` 只解决"以后不再追踪",不会让已有历史变小;真要清理历史体积是更复杂、风险更高的操作(比如 `git filter-repo`),不是这一节要解决的问题——先做到"不再往坑里跳"就是这一节的目标 |
| `.gitignore` 里的规则好像没生效 | 路径写法用了 `\` 而不是 `/`,或者规则写的位置/相对路径不对 | `.gitignore` 里统一用 `/`;拿不准的时候用 `git check-ignore -v <路径>` 直接问 git,它会告诉你到底有没有被忽略、被哪一条规则命中 |

### 4.6 自测清单

- [ ] 能解释为什么 `.gitignore` 对"已经 `add` 过"的文件不起作用
- [ ] 知道 `git rm --cached` 和普通 `git rm` 的区别(会不会删掉硬盘上的真实文件)
- [ ] 能说出至少三类不该进 git 的文件,并各自说出一个理由
- [ ] 能在自己的 `.gitignore` 里写出"忽略某个文件夹"和"忽略某个后缀"两种规则
- [ ] 遇到"这个文件到底有没有被忽略"拿不准的情况,知道用什么命令去问 git(`git check-ignore -v`)

---

## 5. PR 怎么开、code review 怎么参与

### 5.1 为什么需要这个 / 不会有什么后果

先厘清一个容易搞混的地方:**PR(Pull Request,GitHub 的说法;GitLab 里叫 Merge Request,是同一个概念)根本不是 git 本身的功能**,而是 GitHub/GitLab 这类托管平台,在 git 的分支机制之上搭的一层"审查 + 合并"界面。纯用命令行的 git(`git log`、`git branch`)完全不知道"PR"是什么——这个状态活在 GitHub 的服务器数据库里,不在你本地的 `.git` 目录里。

一句话说清楚 PR 本质上是什么:**"我这边有个分支,比 `main` 多出了几个 commit,现在请你看看这些改动,确认没问题之后,把它并进 main。"**

为什么不直接把改动 push 到 `main`,而要绕这一圈?呼应第 1 节的结论——`main` 是所有人默认信任、随时会拉取的分支,PR 存在的意义就是在"改动影响到所有人"之前,插入一个明确的检查点:

1. **提供一个天然的"审查单位":** 不是让审稿人在你陆续发生的每次 commit 里连续跟踪,而是一次性看"这个分支相对 main 完整改了什么"。
2. **给自动化检查一个运行的机会:** 如果仓库配置了 CI(自动跑测试的机制),PR 是它介入的标准时机——测试没过,合并按钮通常会被限制。
3. **留下一份"为什么"的记录:** PR 的描述、讨论串,通常比单条 commit message 承载的上下文更多,而且这份记录会一直挂在那,方便以后回溯。
4. **一个明确的"闸门":** 保证没有任何改动是"意外"出现在 `main` 上的——一定有人明确点了"合并"这个动作。

跳过这层直接推 `main` 的后果,和第 1 节讲的完全一样:`main` 逐渐失去"可信"这个前提,没有人在改动影响所有人之前把过关。

### 5.2 环境要求

- 一个你有推送权限的远程仓库(GitHub 上的仓库,或者这里继续用本地 bare 仓库模拟)
- 一个基于最新 `main` 开出来的分支,上面有至少一个 commit
- **说明:** 下面第 5.3 节分两部分——"推分支到远程"这部分是纯 git 操作,本机真实跑过;"在 GitHub 网页上点开 PR、请求 review"这部分是图形界面操作,这篇文章按 GitHub 当前(2026 年)的界面描述完整流程,但不是本机自动化跑出来的,如实标注。

### 5.3 一步步跟着做

**Part A:推分支到远程(真实操作)。**

先给 `demo-b-workflow` 配一个远程仓库(继续用 bare 仓库模拟 GitHub):

```
$ git init --bare -b main ../demo-b-remote.git
$ git remote add origin ../demo-b-remote.git
$ git push -u origin main
branch 'main' set up to track 'origin/main'.
To ../demo-b-remote.git
 * [new branch]      main -> main
```

开一个功能分支,做一个独立的小改动(给训练脚本加一个 `accuracy` 辅助函数),提交,**只推这个分支,不碰 `main`**：

```
$ git checkout -b feature/add-eval-metric
$ cat >> train.py <<'EOF'

def accuracy(preds, labels):
    return (preds.argmax(dim=-1) == labels).float().mean().item()
EOF
$ git add train.py
$ git commit -m "Add accuracy metric for evaluation

train_step only returned loss; there was no way to see how the
model is actually doing on eval batches. Adds a plain accuracy
helper to be called from the (future) eval loop."
[feature/add-eval-metric 20e3396] Add accuracy metric for evaluation

$ git push -u origin feature/add-eval-metric
branch 'feature/add-eval-metric' set up to track 'origin/feature/add-eval-metric'.
To ../demo-b-remote.git
 * [new branch]      feature/add-eval-metric -> feature/add-eval-metric
```

如果这个远程真的是 GitHub,push 完之后打开仓库页面,GitHub 通常会自动检测到你刚推上去的分支,弹出一条 "Compare & pull request" 的提示条。

**Part B:开 PR、请求 review(图形界面操作,按当前 GitHub 界面描述,未做自动化验证)。**

1. 点 "Compare & pull request"(或者手动去仓库的 "Pull requests" 标签页,点 "New pull request"),选好 base(通常是 `main`)和 compare(你刚推的分支,这里是 `feature/add-eval-metric`)。
2. 填标题和描述——好的 PR 描述应该说清楚"这个改动做了什么、为什么、大概怎么验证过"，和写好一条 commit message 是同一个道理,只是范围从"一次改动"扩大到"一整个分支"。
3. 在右侧 "Reviewers" 里指定审稿人(比如学长、导师),GitHub 会给对方发通知。
4. 等待 review。**审稿人大概会关注这几类问题:** 逻辑对不对、有没有更简洁的写法、有没有漏掉的边界情况、commit 历史是否清晰(呼应第 2 节)、有没有意外带上不该提交的文件(呼应第 4 节)、如果配置了 CI,自动化测试有没有跑过。
5. review 一般有三种结果:**Approve**(同意,可以合并)、**Request changes**(需要改动,会附上具体意见)、**Comment**(单纯提问或建议,不算硬性阻塞)。
6. 如果收到了修改意见:直接在原来那个分支上继续改、`git commit`、`git push`——PR 会自动同步最新的 commit,**不需要关掉重开**。
7. 都通过之后,点 "Merge pull request"(具体是普通合并、"Squash and merge" 还是 "Rebase and merge",看仓库自己的约定,下面第 5.4 节会解释这三种的区别)。
8. 合并完成后,这个功能分支通常就可以删掉了(远程、本地都可以删)——改动已经并入 `main`,分支本身不需要再留着。

### 5.4 背后发生了什么

- **PR 页面上的每个标签,其实都对应一条你已经会用的 git 命令。** PR 的 "Commits" 标签,对应的就是"这个分支比 `main` 多出来的提交":

  ```
  $ git log --oneline main..feature/add-eval-metric
  20e3396 Add accuracy metric for evaluation
  ```

  PR 的 "Files changed" 标签,对应的是这个分支相对 `main` 的完整 diff:

  ```
  $ git diff main...feature/add-eval-metric
  diff --git a/train.py b/train.py
  index 586352a..f22c7c8 100644
  --- a/train.py
  +++ b/train.py
  @@ -3,3 +3,6 @@ def train_step(model, batch, lr):
       loss = model.compute_loss(batch)
       loss.backward()
       return loss
  +
  +def accuracy(preds, labels):
  +    return (preds.argmax(dim=-1) == labels).float().mean().item()
  ```

  这里有个值得注意的细节:上面用的是**三个点**`main...feature/add-eval-metric`,不是两个点。两个点(`git diff main feature/add-eval-metric`)是直接比较两个分支当前的文件内容,如果 `main` 在你开分支之后又有了新的、和你无关的改动,两个点的 diff 会把那些无关改动也混进来;**三个点是拿"两个分支的共同祖先"和"你的分支"比较,只显示你的分支自己贡献了什么**——这才是 PR 页面实际展示给审稿人看的内容,不会被 `main` 上后续无关的进展干扰。这个区别在实测中被验证过:同样两个分支,两点 diff 和三点 diff 给出的结果确实不一样,三点 diff 更干净。

- **三种合并方式改变的是"最终历史长什么样",不是改动内容本身:** 普通 merge(保留完整分支历史 + 一个双父提交的 merge commit,就是本文第 3 节演示的那种)、squash merge(把整个分支的所有 commit 压缩成 `main` 上的一个 commit)、rebase merge(把分支上的 commit 逐个"移植"到 `main` 最新位置,历史变成一条直线,不产生 merge commit)。这篇文章不展开对比哪种更好——这通常是仓库/团队已经定好的约定,不确定就跟着现有习惯走或者直接问。

### 5.5 常见坑

| 现象 | 原因 | 怎么办 |
|---|---|---|
| push 完分支,GitHub 网页上找不到开 PR 的入口 | 推到了错误的远程/没有权限,或者页面缓存没刷新 | `git remote -v` 确认推的是不是自己以为的那个仓库;刷新页面;直接去 "Pull requests" 标签手动点 "New pull request" |
| PR 里出现了一堆不相关的文件改动 | 分支不是从最新的 `main` 开出来的,混进了别的分支的内容 | 开新分支之前先 `git checkout main && git pull`,确保从最新的 `main` 出发 |
| 收到 review 意见,不知道该怎么回应 | 以为要关掉重开一个 PR | 直接在原分支上改、`commit`、`push`,原 PR 会自动更新,不用关掉重开 |
| 只有自己一个人维护仓库,没人 review,想直接跳过 PR 推 `main` | 图省事 | 即使没人 review,PR 依然有用:强迫自己在合并前完整看一遍 diff、留一份"这次改了什么、为什么"的记录,相当于给"以后的自己"当审稿人 |
| 合并方式(merge/squash/rebase)选错,历史变得和预期不一样 | 没注意合并按钮旁边的下拉选项 | 合并前确认一下具体选的是哪种;拿不准就用仓库/团队已经约定好的默认选项 |

### 5.6 自测清单

- [ ] 能用一句话说清楚 PR 本质上是什么(不是 git 概念,是托管平台在两个分支之间搭的审查/合并界面)
- [ ] 能说出为什么 `git log main..你的分支` 对应 PR 页面的 "Commits" 标签,`git diff main...你的分支` 对应 "Files changed" 标签
- [ ] 知道 review 时大概会被关注哪几类问题(逻辑、简洁性、边界情况、无关文件、测试)
- [ ] 知道收到 "Request changes" 之后该做什么(继续在原分支 commit + push,不是关掉重开)
- [ ] 能说出至少两种合并方式的名字,以及它们对最终历史的影响有什么不同

---

## 6. 常见救命命令:`git reflog`

### 6.1 为什么需要这个 / 不会有什么后果

早晚有一次,你会跑出一条比预期更"重"的 git 命令——最常见的是 `git reset --hard` 到了错误的位置,或者删错了分支。那一刻的恐慌感通常是:`git log` 里看不到自己明明记得做过的 commit 了——工作真的丢了吗?

**多数这类"事故",工作并没有真的丢。** 只要那份改动**曾经被 commit 过至少一次**,git 就在本地留了一份记录,记着 `HEAD`(以及每个分支)曾经指向过哪些地方——即使你现在的分支指针已经不指向那里了。`git reflog` 就是查看这份记录的命令,也是从这类事故里把工作找回来的标准手段。

**但要说清楚这不是什么:**

- **它救不了从来没有 commit 过的改动。** 比如 `git checkout -- <文件>` 丢弃的工作区改动,或者忘了 `git stash pop` 又被清空的 stash——这些从来没有形成一个 commit,reflog 里通常找不到,恢复难度大得多,有些情况下根本无法恢复。
- **它是有有效期的,不是永久保险箱。** git 默认对"没有任何分支指向、也没有被 reflog 记录"的旧数据,大约 90 天后会通过 `git gc` 真正清理掉。
- **它只在你自己的电脑上有效。** reflog 不会随 `git push`/`git clone` 传播给别人——如果队友 `push --force` 覆盖了远程历史,你自己电脑上的 reflog 帮不了远程那份丢失的记录(除非那正好也是你自己机器上做过的操作)。

**正是因为这条安全网有这么多限制,`git reflog` 应该被理解成"事故已经发生之后的补救手段",而不是"反正能找回来,`--force`/`--hard` 随便用"的理由。** 平时该有的谨慎(执行这类命令前想清楚、不确定就先别加 `--hard`/`--force`)一点都不能少,reflog 只是在谨慎失效的那次事故里,给你一次挽回的机会。这和这个仓库一直强调的"避免破坏性 git 操作"是同一条纪律。

### 6.2 环境要求

同前面几节。继续用 `demo-b-workflow`,而且这次的演示会直接受益于第 2 节养成的习惯——好的 commit message 会让接下来在 reflog 里"认出哪条是自己要找的记录"容易得多。

### 6.3 一步步跟着做

**第 1 步:正常做两次 commit。**

```
$ git checkout main
$ echo "more logging" >> train.py
$ git add train.py
$ git commit -m "Add debug logging to train_step"
[main 02dcaf2] Add debug logging to train_step

$ echo "early stopping check" >> train.py
$ git add train.py
$ git commit -m "Add early stopping check after eval"
[main cbb8875] Add early stopping check after eval

$ git log --oneline -3
cbb8875 Add early stopping check after eval
02dcaf2 Add debug logging to train_step
2347451 Stop tracking .venv, __pycache__, and checkpoint files
```

**第 2 步:制造一次真实的事故——本来想撤销 1 个 commit,手滑写成了 2 个。**

```
$ git reset --hard HEAD~2
HEAD is now at 2347451 Stop tracking .venv, __pycache__, and checkpoint files
```

看一下损失有多大——`git log` 里那两条不见了,文件内容也真的回退了(注意这次用的是 `--hard`,工作区文件也会被强制改写,和第 2 节 `git reset HEAD~1`(不带 `--hard`)保留工作区内容的效果完全不同):

```
$ git log --oneline -5
2347451 Stop tracking .venv, __pycache__, and checkpoint files
236879c Add training artifacts
dc5f96a Merge branch 'feature/lower-lr'
804160f Increase learning rate to 0.01 to speed up convergence
da0197a Lower learning rate to 0.0001 to stabilize training near convergence

$ tail -3 train.py
def train_step(model, batch, lr):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    loss = model.compute_loss(batch)
```

两次 commit 加的内容确实从 `git log` 和文件里都消失了。

**第 3 步:`git reflog` 找回来。**

```
$ git reflog
2347451 HEAD@{0}: reset: moving to HEAD~2
cbb8875 HEAD@{1}: commit: Add early stopping check after eval
02dcaf2 HEAD@{2}: commit: Add debug logging to train_step
2347451 HEAD@{3}: checkout: moving from feature/add-eval-metric to main
20e3396 HEAD@{4}: commit: Add accuracy metric for evaluation
2347451 HEAD@{5}: checkout: moving from main to feature/add-eval-metric
2347451 HEAD@{6}: commit: Stop tracking .venv, __pycache__, and checkpoint files
236879c HEAD@{7}: commit: Add training artifacts
dc5f96a HEAD@{8}: commit (merge): Merge branch 'feature/lower-lr'
804160f HEAD@{9}: merge feature/higher-lr: Fast-forward
...(后面还有更早的记录,这里省略)
```

这是本地这个仓库里 `HEAD` 每一次移动的完整记录,不只是这次事故相关的——每一行都带着"当时发生了什么"(`commit`/`checkout`/`merge`/`reset`……)和当时的 commit message。**`HEAD@{1}` 这一行清楚写着 `commit: Add early stopping check after eval`——这正是丢失的那两个 commit 里比较新的那个**,而且能一眼认出来,靠的就是第 2 节强调的"写清楚这个 commit 干了什么"的习惯:如果这里全是 `"update"`/`"fix"` 这种 message,你面对几十行 reflog 会完全无从下手。

**第 4 步:恢复。**

```
$ git reset --hard cbb8875
HEAD is now at cbb8875 Add early stopping check after eval

$ git log --oneline -3
cbb8875 Add early stopping check after eval
02dcaf2 Add debug logging to train_step
2347451 Stop tracking .venv, __pycache__, and checkpoint files

$ tail -3 train.py
    return loss
more logging
early stopping check
```

两个 commit 和文件内容都真实地回来了。

### 6.4 背后发生了什么

- **`git reflog` 记录的是什么:** 每个本地仓库自己维护一份日志(存在 `.git/logs/HEAD`,以及每个分支各自也有一份),记录 `HEAD`/分支指针每一次移动前后分别指向哪个 commit——不管这次移动是靠 `commit`、`checkout`、`merge` 还是 `reset` 触发的。它是纯本地的,不会随 `push`/`clone` 传播,这也是为什么第 1 节结尾提到"被 `--force` 覆盖的远程历史",reflog 能救回的只是"这份记录曾经在你自己机器上出现过"的那部分,救不回一个从没在你机器上出现、纯粹活在别人机器上又被覆盖掉的提交。
- **为什么 `reset --hard` 之后 commit 数据本身通常还在:** commit 一旦被创建,只要没有被真正垃圾回收(`git gc`,而且不可达的对象有几十天的默认宽限期),数据本身依然完整地存在仓库的对象库里。`git log` 只是"从当前分支指针出发,顺着父提交链条往下数"能看到的东西——`reset --hard` 做的事情只是把分支指针挪到别处,原来那几个 commit 对象根本没删,只是暂时没有任何指针指向它们,变得不容易直接找到。`git reflog` 相当于一份"指针曾经指过哪里"的备忘录,帮你把丢失的坐标重新找回来。

### 6.5 常见坑

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `git reflog` 输出几十行,不知道该找哪一条 | reflog 记录了这个仓库里所有分支移动过的历史,不只是这次事故 | 先想清楚事故发生前"最后一个还对的状态"大概是什么操作(通常是最近一次 commit),从上往下找对应的 commit message——这也是为什么第 2 节强调的好 message 习惯在这里直接有用 |
| 用 `git reset --hard <reflog 里的 hash>` 之后发现还是不对 | 找错了 reflog 条目 | reflog 记录还在,可以再试一次;拿不准可以先用 `git show <hash>` 或 `git log <hash>` 确认内容对不对,再决定要不要 `reset --hard` |
| 已经过了很久(几周甚至更久),reflog 里翻不到了 | reflog 默认有过期时间,过期后不可达的 commit 会被 `git gc` 真正清理 | 说明"事后补救"的窗口已经关闭——这正是为什么不能把 reflog 当成随便用 `--hard`/`--force` 的免死金牌,它是有过期时间的安全网,不是无限时光机 |
| 不敢直接 `git reset --hard`,担心再出一次事故 | 合理的谨慎 | 更保守的做法:先 `git branch rescue-branch <hash>` 新建一个分支指向那个 commit(这一步不改动当前分支,零风险),确认没问题之后再决定怎么处理它(合并回来,或者直接把 `main` 指过去) |
| 队友说"你的 push 被我 force 覆盖了,提交丢了" | 这属于远程历史被覆盖,不是本地 reflog 能直接解决的场景 | 先看看丢失提交的那个人自己电脑上是不是还留着(他自己的 reflog/分支大概率还在),从他那里重新 push 一次;这也是第 1 节强调"push 被拒绝时不要 `--force`"的原因——一旦 `--force` 推送覆盖了远程,受影响的是别人,不是自己 |

### 6.6 自测清单

- [ ] 能说出 `git reflog` 记录的是什么(本地 `HEAD`/分支指针的移动历史),以及明确不是什么(不会同步给别人,不是永久保留)
- [ ] 遇到"commit 突然从 log 里消失了"的情况,第一反应是 `git reflog`,而不是慌张地重新写一遍
- [ ] 能解释为什么 `reset --hard` 之后 commit 数据本身通常还在,只是指针挪开了
- [ ] 知道更保守的恢复方式(先开个新分支指过去看看),而不是一上来就再 `--hard` 一次
- [ ] 能对自己复述一遍:reflog 是出事故后的补救手段,不是可以随便用 `--force`/`--hard` 的理由——这条和"避免破坏性 git 操作"的纪律是同一件事

---

*本文所有真实命令输出均在 2026-07-23 于隔离的临时目录(`demo-a-no-branches/`、`demo-b-workflow/`)中实测,未对任何真实项目仓库执行过写操作。更新:2026-07-23*
