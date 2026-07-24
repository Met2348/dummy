# 03 · 学术社区互动 —— 宣传帖怎么写、GitHub issue 怎么回、长线关系怎么维护

> 总览见 [00-roadmap.md](00-roadmap.md)
> 前提:已经看过 [01-open-source-code-release.md](01-open-source-code-release.md)(代码仓库该长什么样)和 [02-huggingface-release.md](02-huggingface-release.md)(模型/数据集怎么发布)。这篇讲论文和代码都已经放出去之后的事:怎么让人知道你做了这件事、别人来找你时(不管是夸你还是说"在我这跑不通")怎么应对。这是全系列里"可运行代码"占比最低的一篇——判断力和文字表达本身没法被 assert,这篇如实按这个特点处理,不硬凑代码。

---

## 0. 这篇文章是怎么验证的(先说清楚)

- **社区规范部分,全部来自真实检索,不是凭印象总结**:宣传帖的"Tweetorial"六步结构、GitHub issue 回复的"假定善意+要具体信息"社区共识、NeurIPS reproducibility program 里"作者主动联系原作者澄清疑问"的真实实践记录——每处都在正文标了来源链接。
- **两段可运行代码,是真实、离线、可重复验证的"文字机械检查器"**(不理解语义,只检查能被程序判断的机械信号,比如字符数、是否命中固定词表)——这类工具在 [docs/superpowers/specs/2026-07-25-paper-publication-series-design.md](../../docs/superpowers/specs/2026-07-25-paper-publication-series-design.md) 里"系列 1 科研写作"那一节提出过同样的方法论("能量化的部分写真实代码验证,比如统计一段文字的被动语态占比"),这里是同一套纪律在"社区互动文字"这个场景下的应用:判断力本身不能 assert,但支撑判断的机械信号可以。
- **文中出现的示例宣传帖、示例 GitHub issue/回复,是为教学目的构造的示例,不是真实发过的帖子/真实收到的 issue**——但内容不是凭空编的:示例宣传帖借用的科研问题设定,直接取自 `research/world-model-imagination-controller/00-brainstorm-10-ideas.md` 已经写好的真实"背景一句话"(现有 world model 的想象预算是训练前定死的全局超参数这个真实的研究动机);示例 issue 借用的场景,直接对应 `research/world-model-imagination-controller/eval-protocol/` 目录下真实存在的 `run_pilot_study.py` 脚本和 `PROTOCOL.md` 里真实记录的"5 个随机种子报告均值±标准差"这套方法论。这样做是为了让示例落地、不悬空,但不代表项目真的发过这条帖子、真的收到过这条 issue——每处都会用括注标明"示例,非真实发生"。

---

## 1. 论文放出后,第一步去哪几个地方"挂号"

**为什么需要这个 / 不会有什么后果:**

论文挂上 arXiv 只是"存在"了,不代表"被看见"了。学术圈每天新增的 arXiv 论文数以千计,不主动做点什么,你的论文大概率会淹没在时间线里,不会有什么后果——不代表工作不好,只是"好工作"和"被看到的好工作"之间隔着这一步。

**一步步跟着做(清单式,判断力大于操作步骤):**

1. **arXiv**:论文本体,其余几处大多要依赖这个链接。
2. **HuggingFace Papers 页面**:[02 号文件第 7 节](02-huggingface-release.md)已经讲过——只要你发布的 model card / dataset card 里出现 arXiv 链接,这一步会自动发生,不需要额外提交。
3. **Papers with Code**:[01 号文件](01-open-source-code-release.md)引用过的同一个组织,论文发布代码后,他们的官方建议是"把代码链接加到 Papers with Code",让读者能同时看到论文和排行榜式的对比。
4. **实验室/个人主页**:更新一条"新论文"记录,这是最容易被忽略但最持久的曝光渠道之一——搜索引擎会收录,几年后依然有效,不像社交媒体帖子会被时间线快速冲走。
5. **X (Twitter) / Bluesky 等社交媒体**:第 2 节详细展开。
6. **Google Scholar / Semantic Scholar 个人主页**:确认新论文被正确关联到你的作者档案(有时候需要手动认领,尤其是刚发布的预印本)。

**背后发生了什么:** 这几处渠道服务的读者群体并不完全重叠——arXiv/Google Scholar 是"已经在做相关研究、主动搜索"的读者;X/Bluesky 是"刷时间线偶然刷到"的读者;实验室主页是"已经认识你、来查你最新进展"的读者。挂号这件事本质是"覆盖不同类型的读者获取信息的路径",不是重复劳动。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 论文中了会议才想起来去挂号 | 把"发布"和"投稿"两件事的时间线搞混 | arXiv 预印本通常在投稿时或投稿后不久就可以挂(需先确认目标会议的双盲政策是否允许,见 [paper-submission 系列](../paper-submission-deep-dive/00-roadmap.md) 关于匿名化的内容——如果读到这里发现链接目标还不存在,说明该系列还在并行撰写中,不代表判断错误),不需要等到最终录用 |
| 只发了一次社交媒体帖子就再没动静 | 把"发布"当成一次性动作 | 真实的科研传播建议是"不要发一次就算了",会议录用、代码开源、被别的工作引用,都是可以再发一次的自然节点 |

**自测清单:**

- [ ] 能说出至少 4 个论文发布后应该去"挂号"的地方
- [ ] 能说清楚 HF Papers 页面关联是自动的,不需要额外手动提交(前提是 model/dataset card 里有 arXiv 链接)
- [ ] 知道这几个渠道对应的读者群体不完全一样,不是重复劳动

---

## 2. 宣传帖怎么写 —— 不夸大不模糊

**为什么需要这个 / 不会有什么后果:**

真实调研到的一个常见误区:很多人把论文宣传帖写成"我在 XX 期刊发表了新论文,感谢实验室和基金支持,链接在此"这种格式化模板——这类帖子信息密度极低,读者花几秒钟扫过去,得不到任何关于"这篇论文讲了什么、为什么值得点进去看"的线索,传播效果很差。

真实调研到的更有效结构,业界称为"Tweetorial"(推文教程),一个常见的六步框架:①用一句话钩子说清楚核心发现 ②给背景/上下文 ③极简方法论(不要展开细节) ④关键结果/数据 ⑤这个工作的独特贡献是什么 ⑥附上论文链接。同时要注意平台的机械限制——X 免费账号单条帖子上限 280 字符,超过要么被截断要么要拆成多条。不会有什么后果——不这么组织依然可以发帖,只是大概率达不到"让人愿意点进去看"的效果,论文本身的质量不会因为宣传帖写得好坏而改变,但"有多少人知道这篇论文存在"会有实质差异。

**环境要求:** `.venv` 即可跑下面的检查脚本,不需要真实 X/Twitter 账号。

**一步步跟着做:**

**第一步:一个真实、可运行的"机械检查器"**——只检查两类能用代码判断的信号(字符数是否超限、是否踩到几个典型的夸大用词),**不判断科学内容本身写得对不对**(这需要人去读):

```python
def check_post_draft(text: str, max_chars: int = 280) -> dict:
    """A minimal, real, checkable lint for a paper-announcement post draft.
    Only catches mechanical issues (length, a fixed list of overclaiming words) --
    it cannot judge whether the science itself is accurately described.
    """
    overclaim_words = ["revolutionary", "groundbreaking", "solves", "the best", "perfect"]
    hits = [w for w in overclaim_words if w.lower() in text.lower()]
    return {
        "char_count": len(text),
        "fits_free_tier_limit": len(text) <= max_chars,
        "overclaim_words_found": hits,
    }


# 示例草稿(教学示例,不是真实发过的帖子)——问题设定取自
# research/world-model-imagination-controller/00-brainstorm-10-ideas.md
# 里真实写好的研究动机("现有 world model 的想象预算普遍是训练前定死的全局超参数")
draft = (
    "New preprint: current world models spend a fixed imagination budget on every "
    "decision, whether or not imagining actually helps. We measure how often extra "
    "rollouts change the decision at all, on a gridworld with a known ground truth. "
    "Code + writeup in the thread."
)
report = check_post_draft(draft)
print(report)
assert report["fits_free_tier_limit"] is True
assert report["overclaim_words_found"] == []
print(f"draft is {report['char_count']}/280 characters, no overclaim words matched")

# 对照:一条故意写得夸大的草稿,应该被同一个检查器抓出来
overclaiming_draft = "Our revolutionary method solves imagination budget allocation once and for all!"
bad_report = check_post_draft(overclaiming_draft)
print(bad_report)
assert bad_report["overclaim_words_found"] == ["revolutionary", "solves"]
print("OK: the same checker flags a deliberately overclaiming draft")
```

真实运行结果(`_verify_md.py` 独立验证通过):第一条草稿 263/280 字符,没有命中夸大用词;第二条故意写夸张的草稿只有 79 字符(远没超限),但命中了 `revolutionary`、`solves` 两个词——**这正是这个检查器的局限所在**:字符数超限是硬约束,能完全自动判断;"夸不夸大"只能靠一个人工维护的词表做非常粗略的初筛,真正的判断("这句话有没有夸大这项工作的贡献")还是要靠人读。

**这条草稿好在哪(人工判断,不是脚本判断的部分)**:钩子(第一句)直接给出"现有方法有个具体问题"而不是"我发论文了";第二句给出方法论一句话("测量额外的想象多大程度上真的改变了决策");没有下结论说"我们的方法最好",只是诚实描述"测了什么、怎么测的"——这也是为什么第 0 节强调这条草稿的问题设定是真实项目已经写好的动机,不是为了"听起来厉害"而编的。

**背后发生了什么:** 280 字符限制不是这篇文章编的规则,是 X 平台对免费账号的真实机械限制,直接决定了"钩子"这一步必须极度精炼——六步结构里"钩子"和"方法论"要塞进一条尽量短的帖子,后续几步可以拆成同一个帖子串(thread)里的后续帖子,每条依然各自受 280 字符限制。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 通篇在感谢合作者/基金/期刊,没有说清楚论文讲了什么 | 把"发布公告"和"介绍工作内容"搞反了主次 | 感谢部分放在帖子串靠后的位置,开头必须是内容本身 |
| 一条帖子塞了摘要全文 | 想面面俱到 | 严格执行"六步里每一步只放一条帖子该放的信息",细节留给论文本身 |
| 只贴了论文标题截图 | 图省事 | 调研发现的建议是用自定义配图/图示而不是摘要截图——截图在小屏幕上很难读,自定义图示更抓眼球 |
| 用了检查器却把"没有命中夸大词表"当成"这条帖子没问题" | 把机械检查当成了充分条件 | 参照上面的诚实说明:词表检查只能查出显而易见的夸大用词,查不出"夸大但没用到列表里那几个词"的情况,仍需要人读一遍 |

**自测清单:**

- [ ] 能说出 Tweetorial 六步结构分别是什么
- [ ] 能说出为什么"我发论文了,感谢XXX"这种格式化模板效果通常不好
- [ ] 知道 280 字符是真实的平台机械限制,不是这篇文章编的规则
- [ ] 能说清楚这个"夸大用词检查器"的局限在哪里(只能查固定词表,不能判断语义)

---

## 3. GitHub issue 怎么回 —— "在我这不 work"这类反馈怎么接

**为什么需要这个 / 不会有什么后果:**

代码发布之后,收到"在我这跑不通"类的 issue 是正常现象,不是代码写得不好的信号——[01 号文件第 4 节](01-open-source-code-release.md)已经讲过环境/版本差异真实会导致结果不同。真实调研到的学术圈研究(专门研究"复现失败后作者怎么应对"的论文)记录了一个值得记住的原则:研究者主动联系原作者寻求澄清时,**普遍避免暗示或怀疑论文本身的真实性**,更多把复现失败归因于notation不清楚、某个步骤没写全、某个"看似不重要"的超参数细节缺失,而不是怀疑作者造假或者实现有严重缺陷。这个原则反过来对回复方同样适用:**收到"复现不出来"的反馈,默认对方是遇到了真实的技术/沟通问题,不是来找茬的**,这是回复的第一前提。不会有什么后果——态度上如果先入为主觉得对方在挑刺,容易让回复显得防御性强、不耐烦,即使技术判断是对的,也会伤害社区对这份工作的整体印象,以及你个人的口碑。

**环境要求:** `.venv` 即可跑下面的检查脚本,不需要真实 GitHub 仓库。

**一步步跟着做:**

**第一步:同样是一个只做机械检查的小工具**,检查一条回复有没有踩到几个典型的"敷衍/防御性"表达、有没有主动索要能帮助排查的具体信息:

```python
def check_issue_reply(text: str) -> dict:
    """Lint for a GitHub issue reply about a failed reproduction -- checks for a few
    mechanical signals of good practice, not a judge of tone or correctness."""
    dismissive_phrases = ["works on my machine", "not our problem", "user error"]
    good_faith_hits = [p for p in dismissive_phrases if p.lower() in text.lower()]
    asks_for_specifics = any(
        kw in text.lower() for kw in ["version", "python", "cuda", "seed", "log", "traceback"]
    )
    return {
        "dismissive_phrases_found": good_faith_hits,
        "asks_for_reproduction_specifics": asks_for_specifics,
    }


good_reply = (
    "Thanks for the detailed report. To narrow this down, could you share: "
    "your Python and torch versions, the exact command + seed you ran, and the "
    "full traceback (not just the final line)? We'll try to reproduce with those exact versions."
)
report = check_issue_reply(good_reply)
print(report)
assert report["dismissive_phrases_found"] == []
assert report["asks_for_reproduction_specifics"] is True

bad_reply = "Works on my machine, not sure what to tell you."
bad_report = check_issue_reply(bad_reply)
print(bad_report)
assert bad_report["dismissive_phrases_found"] == ["works on my machine"]
assert bad_report["asks_for_reproduction_specifics"] is False
print("OK: checker distinguishes a specifics-seeking reply from a dismissive one")
```

真实运行结果(`_verify_md.py` 独立验证通过):第一条回复没有命中敷衍词表,且命中了"索要具体信息"的关键词;第二条("在我这能跑,不知道该说啥")命中了敷衍词表,也没有索要任何排查用的具体信息。

**一个示例(教学示例,不是真实收到过的 issue)**,场景对应 `research/world-model-imagination-controller/eval-protocol/` 里真实存在的 `run_pilot_study.py` 和 `PROTOCOL.md` 记录的方法论:

> **Issue 标题**:`run_pilot_study.py` 跑出来的命中率数字和 README 里报告的不一样
>
> 我在自己机器上跑了 `python run_pilot_study.py`,固定用了 README 里写的种子,但拿到的命中率比论文报告的数字低了大概 5 个百分点。是我哪里操作错了吗?
>
> **一个符合上面"假定善意+要具体信息"原则的回复**:
>
> 谢谢反馈。我们这份 pilot 的方法论是 5 个随机种子(0-4)各自独立跑一遍数据采样+训练+评测,报告的是均值±标准差(`PROTOCOL.md` 里写明了)——如果你只跑了单个种子,单种子和 5 种子均值之间有几个百分点的差异是预期内的正常波动,不代表哪里错了。能否麻烦提供:①你具体跑的是单个种子还是全部 5 个种子的均值 ②`numpy`/`python` 版本(`pip show numpy` 和 `python --version` 的输出)③完整的命令行输出?我们这边会用同样的版本尝试复现一遍你的具体设置。

**背后发生了什么:** 这条示例回复做对的几件事,和上面检查器检测的信号是对应的:没有说"在我这能跑"就把责任推给对方;主动给出一个"这可能是正常波动而不是 bug"的具体解释(呼应第 1 号文件第 4 节的方差报告方法论,而不是空口说"应该是正常的");明确列出了几项能真正帮助排查的具体信息。这不是运气好写对的,是把"假定善意"+"要具体信息"两条原则,逐句对照着落实到了回复里。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 回复"在我这能跑" | 下意识把"我这边没问题"当成回复的终点 | 补一句"能否提供你的具体环境/版本/命令",把这当成排查的起点而不是终点 |
| 很久没有回复 issue | 时间/精力有限,或者不知道从哪下手 | 哪怕暂时没时间深入排查,先回复"收到了,这几天会看",让对方知道没有被无视——比彻底沉默的观感好得多 |
| 发现真的是自己代码的 bug,不好意思承认 | 怕显得不专业 | 承认+感谢+修复,是最直接建立可信度的方式,参照第 5 节的进一步讨论 |
| 回复很长很详细,但语气生硬 | 只关注信息完整,没注意语气 | 具体和友善不冲突,开头一句"谢谢反馈/谢谢指出这个问题"成本很低,效果不小 |

**自测清单:**

- [ ] 能说出"假定善意"这条原则的具体含义,以及为什么它对回复方也适用
- [ ] 能写一条"索要具体排查信息"的回复模板,不是"在我这能跑"
- [ ] 遇到真实的 bug,知道该怎么回应(承认+感谢+修复,不是回避)

---

## 4. Follow-up 邮件怎么回

**为什么需要这个 / 不会有什么后果:**

不是所有反馈都会走 GitHub issue——有人会直接发邮件问问题、提合作意向、指出论文里的一个错误。邮件比 issue 更私人、更容易被搁置("等有空再回")然后彻底忘记。不会有什么后果——不回复邮件本身不会有任何系统提醒你,但对发邮件的人来说,长时间不回复通常会被解读成"不重视"或者"傲慢",即使真实原因只是太忙,这种印象一旦形成很难挽回。

**一步步跟着做(判断力清单):**

1. **哪怕不能马上深入回复,先确认收到**:一句"谢谢来信,这周内会仔细看"成本很低,能立刻消除"石沉大海"的印象。
2. **具体问题给具体回答,不要用论文摘要打发**:对方通常已经读过摘要才会写信,重复摘要内容等于没回答。
3. **涉及合作意向的邮件,哪怕最终拒绝,也给一个明确的回复**:不回复比婉拒更伤感情,对方会一直不确定是不是该等你。
4. **如果问题指向论文/代码里一个真实的错误**,处理方式和第 3 节 issue 场景一致:承认、感谢、说明会怎么处理(修 errata、发 GitHub issue 跟踪、下个版本修复)。

**背后发生了什么:** 邮件和 GitHub issue 的核心判断力其实是同一套(假定善意、给具体回应、承认真实错误),差别只在于邮件更私人化、没有公开的时间线压力("总有人会看到这条 issue 挂着没人回"这种隐性督促在邮件里不存在),所以更容易被无限期搁置——知道这一点,能帮你更主动地对抗这个惰性,而不是一套新的方法论要学。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 邮件一直"稍后回复"最后忘了 | 缺少像 GitHub issue 那样的公开可见性提醒 | 可以自己建一个简单的待办清单跟踪未回复的邮件,不要依赖"应该会记得" |
| 收到不感兴趣的合作邀约,选择不回复 | 不知道怎么礼貌拒绝 | 简短明确地说"目前精力有限,这次可能没法参与,祝顺利"比沉默更专业 |

**自测清单:**

- [ ] 知道"先确认收到"这个动作本身的价值,即使还没准备好深入回复
- [ ] 能说出邮件场景和 GitHub issue 场景背后是同一套判断力(假定善意+给具体回应+承认真实错误)
- [ ] 知道为什么邮件比 issue 更容易被无限期搁置,以及怎么对抗这个倾向

---

## 5. 什么时候该说"我不知道" / "这确实是个 bug",什么时候该坚持自己是对的

**为什么需要这个 / 不会有什么后果:**

这一节和"投稿系列"里 rebuttal 的写作技巧,处理的是同一类张力(有人指出你的工作有问题,你该怎么回应),但场景完全不同:rebuttal 是有严格字数限制、几天内必须回完、面向评审这个特定读者的**一次性、高强度对抗**;这里是发布之后、时间线拉长到几个月甚至几年、面向任何一个可能来找你的社区成员的**长线关系维护**。判断标准也因此不同:rebuttal 追求"在有限篇幅内最大化说服评审接收论文",这里追求"维护你和整个社区的长期信任",两者有时候要求的具体做法一致(比如"证据不够就该承认"),但出发点不一样。

不会有什么后果——如果这条判断力没建立好,两种方向都有真实代价:**遇到真 bug 死不承认**,会被认为不诚实,一旦被别人发现你嘴硬,后续所有工作的可信度都会被连带质疑;**遇到对方理解有误就轻易让步**,会在没有必要的地方浪费自己的时间去"修复"一个本来不存在的问题,甚至可能把原本正确的设计改坏。

**一步步跟着做(判断框架,不是操作步骤):**

区分三种真实不同的情况,分别对应不同的应对:

1. **对方指出的确实是一个 bug**:承认、感谢指出、说明修复计划(参照第 3 节的示例回复模式)。这是最容易判断,也最不该犹豫的一种——工程上的对错通常有客观标准(能不能复现、逻辑是否自洽),死撑没有任何好处。
2. **对方的理解有误(不是 bug,是沟通/文档没写清楚)**:耐心解释,并且**把这次解释变成改进文档的机会**——如果一个人会理解错,大概率不是唯一一个,这正是第 1 号文件第 2 节强调的"README 该写清楚复现步骤"在事后被验证的时刻,值得回头把这部分文档补得更清楚,而不只是私下回复了事。
3. **对方指出的其实是一个已知的、诚实披露过的局限,不是新发现的问题**:这种情况下"坚持"是合理的,但坚持的方式是**引用你已经写在论文/README里的局限性说明**,而不是和对方争辩——如果你的局限性小节写得足够诚实具体,这种"重复发现已披露局限"的情况本身出现频率会更低,这也是为什么"局限性诚实自曝"(这类具体写法属于科研写作系列的内容,这篇不重复展开)和这里的社区互动能够互相印证:前期文档写得越诚实,后期需要"坚持自己是对的"的场合就越少、越站得住脚。

**背后发生了什么:** 三种情况的共同判断标准其实只有一个:**回应是不是基于具体、可核查的证据**——是 bug,证据是"确实复现不出来,代码逻辑确实有问题";是理解有误,证据是"文档/论文里其实已经写清楚了,可以直接指给对方看";是已知局限,证据是"论文/README 里那一段文字"。三种情况的应对方式表面不同,但都在做同一件事:用具体证据代替"我觉得"或者"相信我"。

**常见坑:**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 明明是 bug,回复里绕来绕去不肯直说 | 觉得承认错误会显得不专业 | 相反,干脆利落地承认+快速修复,通常会提升而不是降低对方对你的信任 |
| 对方理解有误,回复语气里带着"你没读懂"的意味 | 没意识到"理解错了"往往是文档没写清楚的信号,不完全是读者的问题 | 把语气调整成"看起来这块文档确实容易让人误解,我补充说明一下",而不是单纯指出对方错了 |
| 每次被问到局限性相关的问题都要重新打一遍字解释 | 论文/README 里没有把这条局限写清楚,或者写了但没人注意到 | 补充/加强 README 里"已知局限"这一节(参照 [01 号文件第 2 节](01-open-source-code-release.md)的骨架),下次可以直接引用链接,不用每次重打 |

**自测清单:**

- [ ] 能说出这一节的判断力和"投稿系列"rebuttal 技巧的核心区别(一次性对抗 vs 长线关系维护)
- [ ] 能区分"真 bug"“理解有误"“已知局限被重复问到"三种情况,并各自说出对应的应对方式
- [ ] 能说出三种应对方式背后共同的判断标准(用具体证据代替"我觉得")

---

*参考来源:Tweetorial 六步结构综合自多篇科研社交媒体传播指南([Pitch Science](https://www.pitchscience.com.au/blog/tweetorials-share-academic-paper-twitter-x-bluesky)、[IHPI Twitter 指南](https://ihpi.umich.edu/building-twitter-thread-more-impact)、[KamounLab 论文帖子写作技巧](https://kamounlab.medium.com/tips-for-writing-academic-paper-threads-e578bc50316a));GitHub issue"假定善意"的社区规范参照独立复现性研究文献([A Step Toward Quantifying Independently Reproducible Machine Learning Research](https://arxiv.org/pdf/1909.06674))里作者联系原作者澄清疑问时的真实实践记录。示例宣传帖与示例 issue 场景取材于 `research/world-model-imagination-controller/`(真实研究项目)已有的真实研究动机与真实 pilot 代码,均为教学目的构造的示例,非真实发生过的帖子/issue。*
*创建:2026-07-25*
