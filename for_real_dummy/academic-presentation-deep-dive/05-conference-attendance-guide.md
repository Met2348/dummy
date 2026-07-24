# 05 · 会议参会实用指南

> 前四篇讲的都是"你自己上台/摆摊时怎么表现",这一篇换个身份——大部分会议时间里,你是**观众**:
> 听别人的 talk、逛别人的 poster、在茶歇和走廊里和陌生人搭话。这篇讲怎么把几天的会议行程,过成
> 真正对研究和职业发展有帮助的几天,而不是"报了名、去晃了几天、拍了张会场照片发朋友圈"。

**格式模板**:延续本系列六步演讲判断力模板。

---

## 1. 会前准备:日程规划与电梯演讲

### 常见误区/反面例子

到了会场才第一次翻开日程表,发现同一时间段自己想看的 poster session 和想听的 invited talk 撞车,
或者发现自己想认识的人的报告已经错过了——会议日程通常几天前就会公开,不提前规划是最容易浪费掉的
一类机会。

### 逐处修改对照

**改前**:完全不做行前准备,走一步看一步,遇到感兴趣的 session 就进去听。

**改后(真实调研到的会前准备清单)**:
1. 会前翻一遍完整日程,标出**一定要看的人的报告**(不只是感兴趣的主题,也包括想认识的具体的人)。
2. 检查标出来的场次之间有没有时间冲突,提前决定取舍(下方给出一个可运行的冲突检测工具)。
3. 准备好至少一版 30-60 秒的电梯演讲(呼应
   [03-poster-design-and-pitching.md](03-poster-design-and-pitching.md) 第 3 节的 ABT 结构),不只是
   给自己的 poster 用,茶歇被问"你是做什么方向的"时同样用得上。
4. 如果有认识但不熟的人也会参会,提前打个招呼约个时间("听说你也去 XX 会议,要不要找时间喝杯咖啡")。

**为什么改**:会议日程通常是几天内高密度并行的多个 session,不提前规划几乎必然会撞车错过——尤其是
"poster session 里有一个目标合作者的 poster"这种情况,一旦和另一场感兴趣的 invited talk 撞车,
往往只能二选一,提前发现冲突至少能提前决定优先级,而不是到了现场才纠结。

### 可操作检查清单

- [ ] 有没有在会前完整翻过一遍日程,标出"一定要看的人/主题"?
- [ ] 标出来的场次之间有没有检查过时间冲突?
- [ ] 有没有准备好一版可以随时脱口而出的 30-60 秒自我介绍/研究方向介绍?
- [ ] 如果有目标合作者/想认识的人,有没有提前想好怎么开口(哪怕只是准备好一句破冰的话)?

### 量化验证(真实代码)

日程冲突检测是一个可以完全自动化的真实工具——给定一份"必看清单"(带时间段),自动找出所有互相冲突
的场次,包括容易被忽略的"首尾相接但其实没有真正冲突"这类边界情况。

```python
def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end

def find_conflicts(sessions):
    conflicts = []
    items = list(sessions.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            name_i, (s_i, e_i) = items[i]
            name_j, (s_j, e_j) = items[j]
            if overlaps(s_i, e_i, s_j, e_j):
                conflicts.append((name_i, name_j))
    return conflicts

# 时间用"当天从0点起的分钟数"表示,方便直接比较,不引入datetime的时区/日期复杂度
must_see = {
    "oral: your talk": (9 * 60, 9 * 60 + 15),
    "poster session A (target collaborator's poster)": (10 * 60, 12 * 60),
    "invited talk: test-time scaling": (10 * 60 + 30, 11 * 60 + 15),
    "lunch with labmates": (12 * 60, 13 * 60),
    "workshop: world models": (13 * 60, 15 * 60),
}

conflicts = find_conflicts(must_see)
print("detected schedule conflicts:", conflicts)

names = set()
for a, b in conflicts:
    names.add(a); names.add(b)
assert "poster session A (target collaborator's poster)" in names and "invited talk: test-time scaling" in names, "should detect the conflict between poster A and the invited talk"
assert len(conflicts) == 1, f"this schedule should have exactly 1 conflict, got {len(conflicts)}"
# 边界情况:lunch(12:00-13:00)和workshop(13:00-15:00)首尾相接,不应该被误判为冲突
assert ("lunch with labmates", "workshop: world models") not in conflicts
assert ("workshop: world models", "lunch with labmates") not in conflicts
print("ALL SCHEDULE-CONFLICT ASSERTIONS PASSED")
```

这个例子里,`poster session A` 和 `invited talk` 真实冲突了——这正是会前规划要提前暴露的问题:发现
冲突之后,可以提前决定"invited talk 结束前先快速去打个照面、留个联系方式,正式深聊约在其他时间",
而不是到了现场才发现只能二选一。

### 听众/评委会怎么问

会前规划本身不会被人当面追问,但没做规划的后果会在事后被自己追问——"当时那场 talk 我怎么错过了"。
这一节没有对应的自然"听众提问"场景,如实说明,不硬凑。

### 常见坑

- **只标了"感兴趣的主题",没标"想认识的人"**:很多人规划日程只按论文主题筛选,却忘了同样重要的是
  "这个人会不会在场"——会议价值很大一部分来自见到人本身,不只是听到内容(内容通常之后也能在线看到
  录播或读到论文,人面对面的机会窗口更窄)。
- **把日程排得密不透风**:每个时间段都排满 session,没有给走廊偶遇、临时被拉去喝咖啡这类非计划性
  社交留出空间——下一节会讲到,很多真正有价值的交流发生在非正式场合,不是排满的正式 session 里。

---

## 2. Networking 怎么做:横向 + 纵向、破冰话术、真实对话

### 常见误区/反面例子

常见的两种极端:一种是完全不主动搭话,全程只跟自己实验室的人待在一起;另一种是逮住机会就冲向会场
里最有名的大牛,只为了合影或者递名片,交换完联系方式就再也没有下文——这两种都不构成真正意义上的
networking。

### 逐处修改对照

**改前**:只在感兴趣的大牛报告结束后冲上前说"老师我们加个联系方式吧",对方礼貌回应后就结束了,
没有任何实质性的交流内容。

**改后(真实调研到的分层策略)**:
1. **横向 + 纵向都要做**——不是只盯着领域大牛(通常被围得水泄不通),同样重要的是认识其他博士生
   /博士后(纵向指职级不同,横向指同职级的同行)——同龄人之间反而更容易展开真正的对话,也是未来
   长期合作/互通信息的重要来源。
2. **用具体问题破冰,不是泛泛的寒暄**——真实调研到的好用开场白包括"目前为止你觉得最好的一篇/一场
   talk 是哪个"、"你这次会议感觉怎么样"这类容易接话的问题,比空洞地报姓名单位更容易展开对话。
3. **真正的对话,不是"打卡式"社交**——只是说声"你好"、交换名片不算真正建立了联系,茶歇/午餐这类
   相对轻松的场合更适合展开有实质内容的交流。
4. **独自参会时找同样独自一人的人**——如果发现自己形单影只,找另一个看起来也是一个人的参会者搭话,
   往往比硬挤进一群已经在聊天的人更自然。

**为什么改**:networking 的目标不是"加到多少个联系方式",是"建立起未来真的会有后续互动的关系"——
真实调研到的建议反复强调这一点:真正有价值的连接来自具体的、有实质内容的对话,不是形式化的社交动作。

### 可操作检查清单

- [ ] 有没有主动认识过和自己职级相近的同行,而不是只想着接触"大牛"?
- [ ] 破冰有没有用具体、容易接话的问题,而不是泛泛的"你好,你是做什么的"?
- [ ] 有没有至少一次真正深入的对话(不只是寒暄几句就结束)?
- [ ] 独自一人的时候,有没有主动找另一个独自一人的参会者搭话?

### 量化验证(真实代码)

这一节的核心是社交判断力,大部分内容不可 assert,如实说明。可以做的量化辅助是:检查一份"会议联系人
跟进清单"里,有没有真正区分"只是加了联系方式"和"有实质对话内容记录"这两类,帮助会后跟进时优先级
排序更清楚(这条直接服务第 4 节的会后跟进)。

```python
def summarize_contacts(contacts):
    """contacts: list of dicts,每条至少有 name 和 conversation_notes(字符串,没有实质交流则为空字符串)
    返回真正值得优先跟进的联系人列表——用"备注是否非空"作为"是否有实质对话"的量化代理指标。"""
    substantive = [c["name"] for c in contacts if c["conversation_notes"].strip()]
    card_only = [c["name"] for c in contacts if not c["conversation_notes"].strip()]
    return substantive, card_only

contacts = [
    {"name": "Alice (PhD, same subfield)", "conversation_notes": "discussed her meta-RL work, mentioned a possible shared-benchmark opportunity"},
    {"name": "Bob (booth swag table)", "conversation_notes": ""},
    {"name": "Prof. Carol (invited talk speaker)", "conversation_notes": "asked about follow-up work to the VOC theory paper she cited, she recommended a recent paper"},
    {"name": "Dave (elevator small talk)", "conversation_notes": ""},
]

substantive, card_only = summarize_contacts(contacts)
print("worth following up (substantive conversation):", substantive)
print("contact info only (card-swap networking):", card_only)

assert len(substantive) == 2
assert len(card_only) == 2
assert "Alice (PhD, same subfield)" in substantive
print("ALL CONTACT-SUMMARY ASSERTIONS PASSED")
```

### 听众/评委会怎么问

Networking 场景里"被问"的常见情形反而是自己该主动问的问题被对方反问回来,比如你问"这次会议你觉得
哪场 talk 最好",对方可能反问你同样的问题——**这本身就是对话正常展开的信号**,提前准备好自己对
"这场会议印象最深的内容"有一个真实、具体的答案(不是"都挺好的"这种敷衍回应),是破冰话术能不能
真正延续成对话的关键。

### 常见坑

- **只在自己 poster/talk 时段社交,其余时间当"隐形人"**:networking 不是只发生在自己上场的时候,
  茶歇、午餐、晚上的 social event 往往是更松弛、更容易展开真实对话的场合,如果只在自己有展示任务
  时才主动,等于错过了大部分机会。
- **一次不成功的搭话就自我否定**:不是每一次主动搭话都会有理想的回应,这是正常现象,不代表"我不
  擅长社交"——真实调研到的建议明确提到这一点:networking 成功率不可能是 100%,遇到冷淡反应很正常,
  找到真正支持性的同伴/导师继续鼓励自己尝试,好过因为一两次不顺利就彻底放弃主动搭话。

---

## 3. 作为 Poster Session"参观者"的礼仪

### 常见误区/反面例子

逛别人 poster 时的常见问题有两类:一类是完全不提问,扫一眼标题就走(浪费了和作者直接交流的机会);
另一类走向反面——不是真的在提问,是站在别人的 poster 前发表一通自己的长篇见解,占用大量时间却没有
给出一个真正的问题,也不给其他等候的参观者机会。

### 逐处修改对照

**改前**:在别人 poster 前说"我觉得你们这个方法有个问题,就是……"然后开始长篇大论讲自己的看法,
中间夹杂"这不是问题,只是想法"这类免责声明,占用五分钟却没有落到一个具体问题上。

**改后(真实调研到的提问礼仪)**:先让作者简单介绍,或者主动说"我自己先看一下,有问题再问你"——尊重
作者引导交流节奏的权利。提问时聚焦到一个具体、简短的问题,比如"你们这个消融实验里,去掉这个模块后
效果具体掉了多少",而不是不加节制的长篇评论。如果确实有更深的看法想交流,可以在提完具体问题、得到
回应之后,再补一句"如果你有兴趣,会后我很想多聊聊我的一个想法"。

**为什么改**:poster session 是一对一/一对少数人的高密度交流场合,真实调研到的礼仪建议反复强调
"提问要具体、简短",把"评论"和"问题"混在一起讲成一大段独白,既不礼貌(占用了本该属于作者和其他
参观者的时间),对提问人自己也没有真正获得想要的信息——一个聚焦的具体问题,通常比一段模糊的长篇
评论更容易得到有价值的回应。

### 可操作检查清单

- [ ] 走近一张 poster 时,有没有先给作者一个简短的信号(比如眼神/点头示意),而不是径直插进正在
      进行的对话?
- [ ] 提问是不是聚焦到一个具体、简短的问题,而不是一段长篇评论?
- [ ] 如果自己有更深的看法想交流,有没有先提完具体问题、听完回应,再另外约时间深聊,而不是当场
      占用大量时间?
- [ ] 有没有意识到自己在这张 poster 前停留的时间,可能挤占了排在后面的其他参观者的机会?

### 量化验证(真实代码)

"是不是真正的问题"这类判断力问题不可 assert,但可以做一个客观的启发式检测:一段发言里有没有出现
真正的疑问句标记,以及长度是否明显超出"简短问题"的合理区间——用来自查自己准备要说的话,不是用来
评判别人。

```python
def is_probably_a_real_question(text, max_words=40):
    has_question_mark = "?" in text
    word_count = len(text.split())
    is_concise = word_count <= max_words
    return has_question_mark and is_concise, word_count

comment_disguised_as_question = (
    "I think there's an issue with how you framed the baseline comparison here, because in "
    "my own work we found that this kind of setup tends to underestimate the always-on "
    "policy's true cost, and I've written a few papers on this exact topic, so I think you "
    "should probably reconsider the whole framing, not that it's a question really."
)
real_question = "How much does accuracy drop specifically when you remove the gating head?"

is_q1, n1 = is_probably_a_real_question(comment_disguised_as_question)
is_q2, n2 = is_probably_a_real_question(real_question)
print(f"'comment disguised as question': looks_real={is_q1}, word_count={n1}")
print(f"real specific question         : looks_real={is_q2}, word_count={n2}")

assert not is_q1, "a long monologue with no question mark should not be judged as 'looks like a real question'"
assert is_q2, "a short question with a question mark should be judged as 'looks like a real question'"
print("ALL REAL-QUESTION-CHECK ASSERTIONS PASSED")
```

### 听众/评委会怎么问

这一节的角色是"参观者"而不是"被提问的人",对应的现场场景是反过来的:如果作者礼貌地打断说"不好
意思,能具体讲下你的问题是什么吗",这是一个明确信号,说明自己刚才的发言变成了评论而不是问题,应该
从容收住、直接问出那个具体问题,而不是尴尬或者不高兴。

### 常见坑

- **在提问时"考"作者,而不是真心求解**:有些提问者会用问题来展示自己懂得多,而不是真的想了解作者
  的工作——如果目的是交流而不是较量,提问的语气和内容会很不一样,后者更容易换来作者真诚、详细的
  回应。
- **忽略排队等候的其他人**:如果作者面前已经有人在等,压缩自己的提问/交流时间,或者主动说"我看你
  这边还有人在等,我们简单聊两句,后面再细聊"——这是前面 04 号文件"控制回答时长"的对称版本,这里
  是参观者主动控制自己占用的时长。

---

## 4. 会后跟进

### 常见误区/反面例子

会议期间加了十几个联系方式,回去之后因为忙碌全部搁置,几个月后再想起来,对方大概率已经完全不记得
这个人是谁,联系方式变成了通讯录里一个再也不会被联系的条目。

### 逐处修改对照

**改前**:会议结束就结束了,没有任何后续动作。

**改后(真实调研到的跟进节奏)**:
1. **会后几天内**,给有实质对话的联系人发一封简短的感谢/后续邮件,提一句具体聊过的内容(而不是
   泛泛的"很高兴认识你"),让对方能想起你是谁。
2. **中长期(比如下次会议前)**,可以主动问一句"下次 XX 会议你会去吗,到时候要不要约个时间聊聊",
   把一次性的相遇变成有节奏的长期联系。
3. **持续关注对方的动态**——比如对方发了新论文、拿了奖、换了工作,发一句简短的祝贺,是低成本但
   有效的"保持在对方雷达上"的方式。

**为什么改**:第 2 节已经强调过 networking 的目标是"建立会有后续互动的关系",如果加了联系方式却
不跟进,等于把第 2、3 节做的所有努力浪费在了最后一步——真实调研到的建议明确指出,及时的、具体的
跟进邮件,是在对方心里留下正面印象的低成本高回报动作。

### 可操作检查清单

- [ ] 会后几天内,有没有给"值得优先跟进"的联系人(参考第 2 节的量化工具)发一封简短、具体的后续
      邮件?
- [ ] 邮件里有没有提一句具体聊过的内容,而不是泛泛的客套话?
- [ ] 有没有为可能的长期联系埋一个"钩子"(比如提到下次可能相关的会议/合作机会)?
- [ ] 有没有把这次认识的人,加进一个自己会定期回顾的联系人列表,而不是散落在聊天记录里再也不会
      翻到?

### 常见坑

- **跟进邮件写得像模板**:内容空洞、看得出是群发模板的邮件("很高兴认识你,期待未来合作"),效果
  通常远不如提一句具体细节("我们聊到的关于 XX benchmark 的想法,我后来又想了一下……")——具体
  细节证明你真的记得这次交流,而不是走个形式。
- **只在需要对方帮忙时才联系**:如果联系记录只有"求推荐信""求内推"这类单向索取,缺少平时的正常
  互动(比如祝贺对方的新进展),关系的长期质量会打折扣——第 2 节强调的"真实对话"和这里的"持续
  关注",本质上是同一件事在不同时间尺度上的体现。

---

## 参考来源

- 会前日程规划、"标出想认识的人而不只是想听的主题"的建议,综合自多篇 PhD 会议参会指南,检索关键词
  "how to prepare for an academic conference schedule"。
- 横向+纵向 networking、"Old Faithful"式破冰话术("这次会议感觉怎么样""目前听到最好的 talk 是
  哪场")、独自参会时的搭话策略,综合自多篇学术 networking 指南,检索关键词 "how to network at
  academic conferences advice PhD students"。
- Poster session 参观者礼仪(具体简短提问、不要把评论伪装成问题、照顾排队等候的其他人),综合自
  多篇会议礼仪指南,检索关键词 "poster session etiquette academic conference how to ask questions"。
- 会后跟进节奏(几天内发具体的感谢邮件、长期保持联系),综合自上述 PhD 会议参会指南与 networking
  指南。

---

*上一篇:[04-live-qa-skills.md](04-live-qa-skills.md) ·
下一篇:[06-conference-day-capstone.md](06-conference-day-capstone.md)*
