"""社招关卡三：隐藏Bug调试深水（21个DeepPoint）。

覆盖2026年国内大厂社招面试loop里增长最快的一类关卡——debugging round：面试官给一段
表面正常、只在特定输入/时序/并发条件下才会触发的隐蔽bug代码，候选人必须先读懂已有代码
逻辑（而不是自己从零写），再预测给定触发输入下的真实运行时行为，然后精确定位出错的
代码行并解释根因机制，最后给出修复方案并诚实说明这类bug日常开发里容易在什么场景被
忽略、有没有更系统性的预防手段（linter规则/code review checklist/单元测试用例设计）。
覆盖Python语言特性坑（可变默认参数、闭包晚绑定、生成器耗尽、bool是int子类、format注入）、
并发与时序类（TOCTOU竞态、cache-aside失效竞态、asyncio共享状态竞态、双重检查锁定单例、
锁顺序死锁）、数值与边界类（整数溢出/numpy定长回绕、浮点精度、分页差一、空集合边界、
时区DST边界）、以及资源与工程实践类（文件句柄泄漏、字典迭代修改、递归栈溢出、异常静默
吞没）四大类隐蔽bug，每个场景都给出可以实际验证行为的具体代码片段和触发条件。

边界：这一关和gate2(手撕代码/被追问设计权衡)不同——这里不要求候选人从零实现算法，
考的是阅读、复现、定位、修复别人代码里已经存在的bug这条独立能力线；不涉及大规模分布式
系统的架构设计判断(那是ML infra/agentic system design这类系统设计关卡的范围)，也不
展开某个具体bug背后语言/框架的完整原理教程，只聚焦'这段代码为什么在这个输入下是错的'
这一具体链条。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, grade_chain  # noqa: E402

CAT = "社招关卡三:隐藏Bug调试深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-sh-dbg-01",
        cat=CAT,
        trigger="""面试官说：'看下这段订单系统里的函数，先说说它整体在干什么？'

```python
def add_item(item, basket=[]):
    basket.append(item)
    return basket
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段函数把传入的item追加到basket列表末尾再返回basket；函数签名写成def add_item(item, basket=[])，"
             "如果调用方不传basket，直觉上会以为每次调用都会拿到一个全新的空列表来装东西。",
             ("def add_item(item, basket=[])", "追加到basket列表末尾", "全新的空列表")),
            ("如果连续两次不传basket参数调用add_item(1)然后add_item(2)，实际会发生什么？",
             "第一次调用add_item(1)返回[1]，看起来一切正常；但第二次调用add_item(2)时实际返回的是[1, 2]，"
             "而不是很多人以为的[2]，因为两次调用背后共用的是同一个basket列表对象。",
             ("add_item(1)返回[1]", "实际返回的是[1, 2]", "共用的是同一个basket列表对象")),
            ("具体是哪一行导致的，根因机制是什么？",
             "根因在函数定义那一行def add_item(item, basket=[])：Python的默认参数值只在函数定义(def语句执行)"
             "时被求值一次并绑定到函数对象上，此后每次省略该参数调用都会复用这同一个默认对象，而不是每次调用都"
             "重新创建一个新的空列表；basket又是可变对象，一旦某次调用append修改了它，这个修改会残留到下一次"
             "调用里。",
             ("函数定义(def语句执行)时被求值一次", "复用这同一个默认对象", "残留到下一次调用里")),
            ("怎么修复，这类bug在日常开发里容易在什么场景被忽略，有没有系统性预防手段？",
             "正确写法是把默认值改成None，在函数体内部判断if basket is None: basket = []，保证每次调用没传参"
             "时都新建一个列表；这类bug最容易在工具函数写完之后长期没人碰、也没人在单元测试里连续调用两次的"
             "场景里被忽略，系统性预防手段包括开pylint的W0102 dangerous-default-value规则或flake8-bugbear的"
             "B006，以及在code review checklist里加一条默认参数是否为list/dict/set等可变对象，单元测试要专门"
             "设计连续调用两次不传该参数、断言两次结果互不污染这种用例。",
             ("if basket is None: basket = []", "pylint的W0102 dangerous-default-value", "连续调用两次不传该参数、断言两次结果互不污染")),
        ),
        pitfall="很多人知道'Python默认参数有坑'这句话，但说不清楚是def语句执行时求值一次这个具体机制，第2层被追问连续调用两次的具体输出时容易愣住答不出[1, 2]这个具体结果。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-02",
        cat=CAT,
        trigger="""面试官说：'这段代码想批量注册几个回调，你看看有什么问题？'

```python
handlers = []
for i in range(3):
    handlers.append(lambda: print(i))
for h in handlers:
    h()
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段代码先用一个for i in range(3)循环，把三个lambda: print(i)依次追加进handlers列表，每个lambda"
             "表面上看起来记住了当时循环到的i值；然后再用第二个for h in handlers循环依次调用这三个handler，"
             "直觉上应该分别打印0、1、2。",
             ("for i in range(3)循环", "lambda: print(i)依次追加进handlers", "分别打印0、1、2")),
            ("实际运行这段代码，三次调用分别打印出什么？",
             "实际打印出来的是三个2，也就是2、2、2，而不是直觉上的0、1、2。",
             ("三个2", "2、2、2", "0、1、2")),
            ("具体是什么机制导致的？",
             "根因是Python闭包的晚绑定(late binding)：lambda: print(i)里的i不是在lambda创建那一刻把当时的值"
             "拷贝进去，而是保存了对外层作用域里变量i的引用，真正查找i的值发生在lambda被调用的那一刻；等第一个"
             "for循环结束时，i已经被循环体一路递增到最后一个值2，所以三个lambda调用时查到的都是同一个已经变成"
             "2的i。",
             ("闭包的晚绑定(late binding)", "保存了对外层作用域里变量i的引用", "已经被循环体一路递增到最后一个值2")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复常见做法是给lambda加一个默认参数把当前i的值在创建时就拷贝绑定进去，写成lambda i=i: print(i)，"
             "或者用functools.partial(print, i)显式绑定当前值；这类bug最容易在批量创建回调/任务列表后延迟执行"
             "这种场景被忽略，比如GUI按钮回调、批量注册定时任务、批量生成异步任务闭包，写的时候功能测试往往只测"
             "单个callback顺手就通过了；系统性预防手段是code review checklist里加一条循环体内创建闭包时是否"
             "直接引用了循环变量，以及ruff的B023 function-uses-loop-variable规则，单元测试要设计创建多个闭包后"
             "逐个调用、断言每个返回值与创建时的循环变量一一对应这种用例。",
             ("lambda i=i: print(i)", "循环体内创建闭包时是否直接引用了循环变量", "ruff的B023 function-uses-loop-variable规则")),
        ),
        pitfall="很多人知道'闭包晚绑定'这个名词，但只会含糊地说循环变量有坑，具体被问到三次调用实际打印什么时答不出是三个2，也说不清楚i是引用而不是拷贝这个关键区别。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-03",
        cat=CAT,
        trigger="""面试官说：'这是账户取款的核心逻辑，先讲讲它在做什么？'

```python
def withdraw(account, amount):
    if account.balance >= amount:
        time.sleep(0.001)  # 模拟IO/网络延迟
        account.balance -= amount
        return True
    return False
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段withdraw函数先检查account.balance是否大于等于要取的amount，如果余额够，就把account.balance"
             "减去amount并返回True表示扣款成功；如果余额不够，直接返回False拒绝这次取款，看起来是一个标准的"
             "先检查余额、够了再扣款的逻辑。",
             ("检查account.balance是否大于等于要取的amount", "把account.balance减去amount", "先检查余额、够了再扣款")),
            ("如果account.balance初始是100，两个线程/协程几乎同时分别调用withdraw(account, 80)和"
             "withdraw(account, 60)，实际会发生什么？",
             "理想情况下两笔取款总额140大于余额100，应该只有一笔能成功；但实际运行时两个调用的"
             "if account.balance >= amount检查都可能在对方还没执行减法之前就先后通过，于是两次都判定余额够、"
             "都执行了减法，最终account.balance变成100-80-60=-40，出现了不该被允许的透支，而且两次调用都"
             "返回True。",
             ("总额140大于余额100", "都判定余额够、都执行了减法", "account.balance变成100-80-60=-40")),
            ("具体是哪里出的问题，根因机制是什么？",
             "根因是if account.balance >= amount这行检查和account.balance -= amount这行扣减之间存在一个时间"
             "窗口，检查(check)和真正执行扣减动作(act)不是一个原子操作，这就是经典的TOCTOU(time-of-check to "
             "time-of-use)竞态条件：两个并发调用都在这个窗口内读到了扣减前的旧余额，各自都认为自己的检查通过"
             "了，谁也没看到对方即将做的修改。",
             ("if account.balance >= amount这行检查和account.balance -= amount这行扣减之间存在一个时间窗口",
              "TOCTOU(time-of-check to time-of-use)竞态条件", "读到了扣减前的旧余额")),
            ("怎么修复，这类问题什么场景容易被日常开发忽略，有没有系统性预防手段？",
             "修复要把检查和扣减合并成一个原子操作，比如用一把锁把检查+扣减整体包起来，或者在数据库层用"
             "UPDATE accounts SET balance = balance - %s WHERE balance >= %s这种带WHERE条件的原子更新语句，"
             "靠数据库自身的行锁保证原子性，再用返回的受影响行数判断是否扣款成功；这类bug最容易在本地单元测试"
             "都是单线程跑、从来没模拟过并发的场景被日常开发忽略，只有上生产遇到高并发流量才会偶发触发；"
             "系统性预防手段包括code review checklist里加一条凡是check-then-act的读改写逻辑是否有并发保护，"
             "以及专门写并发压测用例，用多线程/多协程同时对同一账户发起竞争性请求、断言最终余额不会出现透支。",
             ("UPDATE accounts SET balance = balance - %s WHERE balance >= %s", "本地单元测试都是单线程跑、从来没模拟过并发",
              "多线程/多协程同时对同一账户发起竞争性请求")),
        ),
        pitfall="很多人知道'并发要加锁'这句正确的废话，但说不出这是TOCTOU竞态条件、也说不出具体两个并发调用下余额会变成-40这个可验证的结果，容易被面试官追问具体数字时卡住。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-04",
        cat=CAT,
        trigger="""面试官说：'这是个分片路由用的哈希函数，先说说它在做什么？'

```python
import numpy as np

def hash_bucket(key: int, n_buckets: int) -> int:
    h = np.int32(key * 2654435761)
    return int(h) % n_buckets
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段hash_bucket函数用Knuth乘法哈希的思路，把key乘以常数2654435761后强制转换成np.int32(32位有"
             "符号整数)，再对n_buckets取模，目的是把任意key打散映射到[0, n_buckets)这个桶范围里，用于类似分片"
             "路由这样的场景。",
             ("Knuth乘法哈希的思路", "强制转换成np.int32(32位有符号整数)", "映射到[0, n_buckets)这个桶范围里")),
            ("如果key=3000000000，n_buckets=16，这行代码实际执行会发生什么？",
             "key=3000000000乘以2654435761后得到的结果远远超过32位有符号整数能表示的最大值(2147483647)，"
             "np.int32(...)这一步不会报错，而是按照C风格的补码回绕(wraparound)规则把结果截断成一个32位有符号"
             "整数，很可能变成一个看起来莫名其妙的负数，最终产生一个虽然落在[0,16)范围内、但完全不等于预期"
             "数学乘积mod 16应有结果的桶编号。",
             ("远远超过32位有符号整数能表示的最大值(2147483647)", "按照C风格的补码回绕(wraparound)规则把结果截断",
              "完全不等于预期数学乘积mod 16应有结果的桶编号")),
            ("具体是哪一行、什么机制导致的？",
             "问题出在np.int32(key * 2654435761)这一行：key * 2654435761在Python原生int运算里是没有位宽限制"
             "的大整数乘法，不会溢出；但只要外面套一层np.int32(...)做类型转换，numpy就会按照C语言定长整数的"
             "语义把这个大整数截断成32位补码表示，超出范围的高位直接被丢弃，这和Python原生整数自动扩位、永不"
             "溢出的直觉完全相反，是从其他语言(C/Java)移植哈希算法到Python+numpy环境时最容易踩的坑——原作者"
             "以为这里的int32只是类型标注，没意识到它真的会做定长截断。",
             ("np.int32(key * 2654435761)这一行", "numpy就会按照C语言定长整数的语义把这个大整数截断成32位补码表示",
              "以为这里的int32只是类型标注")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复可以是在转换前先对key乘积结果按照0xFFFFFFFF做位与操作显式模拟32位无符号回绕再解释符号位，"
             "或者干脆放弃用np.int32，只在最后返回前对Python原生大整数结果做%n_buckets；这类bug最容易在把别的"
             "语言的哈希/校验和算法照抄过来、顺手套一层看起来像类型标注的np.int32这种场景被忽略，日常开发很少"
             "会专门测试超大key输入；系统性预防手段是code review checklist里加一条凡是引入numpy/ctypes定长"
             "整数类型的地方要明确写注释说明为什么需要定长语义，以及单元测试里专门覆盖远超2^31的key输入这类"
             "边界用例。",
             ("对key乘积结果按照0xFFFFFFFF做位与操作", "把别的语言的哈希/校验和算法照抄过来", "远超2^31的key输入")),
        ),
        pitfall="很多人知道'整数溢出是C/Java的问题，Python没有溢出'，恰恰因为这个先入为主的印象，看到np.int32时容易直接跳过不当回事，答不出具体截断+补码回绕的机制，也想不到要专门测超大key这个边界。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-05",
        cat=CAT,
        trigger="""面试官说：'这是订单对账逻辑，先说说这段代码在做什么？'

```python
def total_price(prices: list[float]) -> float:
    total = 0.0
    for p in prices:
        total += p
    return total

if total_price([19.9, 19.9, 19.9]) == 59.7:
    print('价格核对通过')
else:
    print('价格核对失败，人工复核')
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段代码用一个for循环把prices列表里的每个价格累加到total变量里得到总价，然后拿累加结果和硬编码"
             "的期望值59.7做==比较，如果相等就打印价格核对通过，不相等就打印价格核对失败，人工复核，看起来是"
             "一个简单的对账校验逻辑。",
             ("用一个for循环把prices列表里的每个价格累加到total变量里", "拿累加结果和硬编码的期望值59.7做==比较", "价格核对通过")),
            ("如果prices就是[19.9, 19.9, 19.9]，这段代码实际会走到哪个分支、打印什么？",
             "实际上total_price([19.9, 19.9, 19.9])算出来的浮点数结果并不精确等于59.7，而是类似"
             "59.699999999999996这种带有微小误差的值，所以total == 59.7这个比较结果是False，代码实际会走到"
             "else分支，打印价格核对失败，人工复核。",
             ("类似59.699999999999996这种带有微小误差的值", "total == 59.7这个比较结果是False", "价格核对失败，人工复核")),
            ("具体是什么机制导致的？",
             "根因是19.9这个十进制小数在IEEE 754双精度浮点数里并不能被精确表示，实际存的是一个最接近19.9的"
             "二进制近似值，三次累加把这个近似误差逐步放大累积，最终得到的total和数学上精确的59.7之间存在一个"
             "极小的浮点误差；而代码里用的是对浮点数做严格的==相等比较，这种比较对任何浮点误差都零容忍，哪怕"
             "误差只有1e-13这个量级也会判定为不相等。",
             ("19.9这个十进制小数在IEEE 754双精度浮点数里并不能被精确表示", "对浮点数做严格的==相等比较",
              "1e-13这个量级也会判定为不相等")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复方案一是用math.isclose(total, 59.7, rel_tol=1e-9)这类带容差的比较代替==；方案二是涉及金额"
             "这类需要精确十进制运算的场景，从一开始就用decimal.Decimal或者用分为单位的整数运算，避免引入"
             "二进制浮点误差；这类bug最容易在开发时随手写==比较两个看起来算出来该相等的浮点数这种场景被忽略，"
             "尤其是金额、计费这类字段；系统性预防手段包括在code review checklist里加一条凡是涉及金额的字段"
             "是否用了Decimal而不是float，单元测试要专门设计多笔小数金额反复累加后与预期值比较这类用例来暴露"
             "浮点误差。",
             ("math.isclose(total, 59.7, rel_tol=1e-9)", "用分为单位的整数运算", "凡是涉及金额的字段是否用了Decimal而不是float")),
        ),
        pitfall="很多人知道'浮点数不能直接用==比较'这句常识，但具体问到[19.9,19.9,19.9]累加后到底走if还是else分支时容易含糊带过，答不出59.699999999999996这个具体值。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-06",
        cat=CAT,
        trigger="""面试官说：'这段代码给数据库字段做长度截断，先说说它在做什么？'

```python
def truncate_bytes(s: str, max_bytes: int) -> str:
    b = s.encode('utf-8')
    return b[:max_bytes].decode('utf-8')
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段truncate_bytes函数先把字符串s编码成UTF-8字节串b，然后取b的前max_bytes个字节，再把这部分"
             "字节解码回字符串返回，目的是把一个字符串限制在指定的字节长度以内，比如给一个只允许存储N字节标题"
             "的数据库字段做截断。",
             ("先把字符串s编码成UTF-8字节串b", "取b的前max_bytes个字节", "限制在指定的字节长度以内")),
            ("如果s='今天天气'，max_bytes=7，这行代码实际执行会发生什么？",
             "今天天气这四个汉字每个在UTF-8下占3个字节，s.encode('utf-8')总共是12个字节；取前7个字节意味着"
             "正好切在第3个字符天的中间，这不是一个合法的UTF-8字符边界，所以b[:7].decode('utf-8')这一步会"
             "直接抛出UnicodeDecodeError，而不是优雅地返回一个截断到今天或者今天天的字符串。",
             ("每个在UTF-8下占3个字节", "正好切在第3个字符天的中间", "直接抛出UnicodeDecodeError")),
            ("具体是哪一行、什么机制导致的？",
             "问题出在b[:max_bytes]这个纯字节切片操作：它完全不知道UTF-8里一个字符可能由1到4个字节组成，只"
             "会机械地在第max_bytes个字节处一刀切下去，如果这一刀恰好落在某个多字节字符编码序列的中间，切出来"
             "的字节串末尾就是一个不完整、不合法的UTF-8序列，随后的.decode('utf-8')无法还原出对应字符，只能"
             "抛异常；根因是把按字节截断和按字符截断当成了同一件事。",
             ("b[:max_bytes]这个纯字节切片操作", "只会机械地在第max_bytes个字节处一刀切下去",
              "把按字节截断和按字符截断当成了同一件事")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复思路是在截断后做一次安全解码，比如用b[:max_bytes].decode('utf-8', errors='ignore')丢弃末尾"
             "不完整的字节，或者更严谨地从max_bytes往前找最近的合法UTF-8字符边界再截断；这类bug最容易在开发和"
             "测试时只用英文/纯ASCII样例数据、从没测过中文、emoji这类多字节字符的场景被忽略；系统性预防手段"
             "包括code review checklist里加一条凡是涉及字符串按长度截断的地方是否用了字符级而不是字节级操作，"
             "单元测试要专门覆盖包含中文/emoji的字符串在各种截断长度下不会抛异常这类用例。",
             ("b[:max_bytes].decode('utf-8', errors='ignore')", "开发和测试时只用英文/纯ASCII样例数据",
              "包含中文/emoji的字符串在各种截断长度下不会抛异常")),
        ),
        pitfall="很多人知道'中文一个字占多个字节'这个常识，但没意识到按字节切片会切到字符中间导致解码报错，具体问到s='今天天气'、max_bytes=7会发生什么时说不出会抛UnicodeDecodeError这个具体结果。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-07",
        cat=CAT,
        trigger="""面试官说：'这是缓存读写逻辑，先说说这两个函数各自在做什么？'

```python
def get_user(uid):
    val = cache.get(uid)
    if val is not None:
        return val
    val = db.query(uid)          # 耗时的DB查询
    cache.set(uid, val)          # 写回缓存
    return val

def update_user(uid, new_val):
    db.update(uid, new_val)
    cache.delete(uid)
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "get_user走的是标准的cache-aside读路径：先查cache，命中直接返回；不命中就去db.query查询，把查到"
             "的值写回cache.set后再返回；update_user走的是写路径，先执行db.update更新数据库，再执行cache.delete"
             "把对应的缓存条目删掉，让下次读的时候重新从数据库加载最新值。",
             ("先查cache，命中直接返回", "先执行db.update更新数据库，再执行cache.delete", "重新从数据库加载最新值")),
            ("如果线程A调用get_user(uid)时缓存恰好未命中、正在执行db.query()这个耗时查询期间，线程B并发调用"
             "update_user(uid, new_val)完整执行完毕，之后线程A才从db.query()拿到结果继续往下执行cache.set，"
             "最终缓存里会是什么值？",
             "最终cache里存的会是线程A读到的旧值(update之前的值)，而不是线程B刚写进数据库的new_val：因为线程A"
             "的db.query()是在线程B更新之前就已经发出去的查询，拿回来的天然是旧数据；线程B的cache.delete在"
             "线程A的cache.set之前执行完，之后线程A用旧值重新调用cache.set把这个旧值写回了缓存，缓存里从此"
             "固定停留在这个错误的旧值上。",
             ("最终cache里存的会是线程A读到的旧值(update之前的值)", "线程B的cache.delete在线程A的cache.set之前执行完",
              "缓存里从此固定停留在这个错误的旧值上")),
            ("具体是什么机制导致的？",
             "根因是get_user里查库和写缓存这两步之间存在一个可被打断的时间窗口，一旦在这个窗口期间发生了一次"
             "针对同一个key的写入+删缓存操作，读路径后续的cache.set就会用一个已经过期的值覆盖掉本该是空的缓存"
             "状态；这是cache-aside模式里一个被广泛讨论的经典竞态，本质上是读写两条路径对同一个key的操作顺序"
             "发生了错误的交错(interleaving)。",
             ("查库和写缓存这两步之间存在一个可被打断的时间窗口", "cache-aside模式里一个被广泛讨论的经典竞态",
              "读写两条路径对同一个key的操作顺序发生了错误的交错(interleaving)")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "常见修复手段包括给写路径加一个延迟双删(更新完db后先删一次缓存，sleep几百毫秒后再删一次)，或者"
             "在写回缓存时带上版本号/时间戳做compare-and-set，只有当前写回的数据版本不早于缓存里已有版本时才"
             "真正写入；这类bug最容易在压测/功能测试都是单线程串行跑、从没让读和写路径真的并发打过同一个key的"
             "场景被忽略；系统性预防手段是code review checklist里加一条cache-aside模式下读路径的查询+回写缓存"
             "是否可能被并发的写删除操作打断，以及专门写并发测试用例，模拟读进行中，写并发完成这种交错时序。",
             ("延迟双删", "带上版本号/时间戳做compare-and-set", "读进行中，写并发完成")),
        ),
        pitfall="很多人知道cache-aside要先更新DB再删缓存这条经验法则，但说不清楚这条法则本身在读写并发交错时依然会留下一个读路径把旧值写回缓存的漏洞，具体被问到交错时序下最终缓存值时容易语塞。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-08",
        cat=CAT,
        trigger="""面试官说：'这是一个监控指标的滑动平均类，先说说各个方法在做什么？'

```python
from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)

    def add(self, value):
        self.window.append(value)

    def average(self):
        return sum(self.window) / len(self.window)

    def reset_if_stale(self, is_stale):
        if is_stale:
            self.window.clear()
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "这段MovingAverage类用一个固定长度maxlen=window_size的deque做滑动窗口：add往窗口里追加新的value；"
             "average用sum(self.window)/len(self.window)算窗口内当前所有值的平均数；reset_if_stale在判断数据"
             "已经过期(is_stale为True)时调用self.window.clear()把窗口清空，为下一轮重新开始积累做准备。",
             ("固定长度maxlen=window_size的deque做滑动窗口", "sum(self.window)/len(self.window)算窗口内当前所有值的平均数",
              "self.window.clear()把窗口清空")),
            ("如果监控系统在调用了reset_if_stale(True)清空窗口之后、还没来得及调用add重新塞入新数据之前，恰好"
             "有一次定时上报任务调用了average()，会发生什么？",
             "self.window在被clear()清空之后是一个长度为0的空deque，此时调用average()执行的是"
             "sum(self.window) / len(self.window)，sum对空deque得到0，但len(self.window)也是0，0/0会直接"
             "抛出ZeroDivisionError异常，而不是像很多人以为的那样返回0或者None这种安全的默认值。",
             ("self.window在被clear()清空之后是一个长度为0的空deque", "0/0会直接抛出ZeroDivisionError异常",
              "返回0或者None这种安全的默认值")),
            ("具体是哪一行、什么机制导致的？",
             "问题就出在average方法里的sum(self.window) / len(self.window)这一行：这行代码隐含了一个窗口里"
             "至少有一个元素的假设，正常业务流程下add总是先于average被调用；但reset_if_stale这个过期即清空的"
             "操作打破了这个隐含假设，一旦clear()和下一次add之间恰好夹了一次average调用，窗口长度就是0，根因是"
             "代码把窗口非空当成了理所当然的前提，却没有对这个前提做任何显式校验。",
             ("隐含了一个窗口里至少有一个元素的假设", "reset_if_stale这个过期即清空的操作打破了这个隐含假设",
              "把窗口非空当成了理所当然的前提，却没有对这个前提做任何显式校验")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复很直接，在average方法里加一个显式判空：if not self.window: return 0.0，把窗口为空该怎么办这"
             "个决策显式地做出来；这类bug最容易在先写了正常的读写逻辑、后面才补充了一个清空/重置功能这种迭代式"
             "开发场景里被忽略，因为清空功能往往是后加的，加的人没有回头检查所有读取窗口的方法在空窗口下的行为；"
             "系统性预防手段是code review checklist里加一条凡是新增了清空/重置某个集合状态的操作是否检查过所有"
             "读取该集合的下游方法，单元测试专门覆盖clear之后立即调用average这个组合用例。",
             ("if not self.window: return 0.0", "清空功能往往是后加的", "clear之后立即调用average")),
        ),
        pitfall="很多人只会说'除0要判断'这句正确的废话，但具体到这段代码时想不到是reset_if_stale这个后加的清空功能打破了窗口非空的隐含假设，容易把注意力放错在add/average本身。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-09",
        cat=CAT,
        trigger="""面试官说：'这是链表反转的实现，先说说它在做什么？'

```python
class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

def reverse_list(node, prev=None):
    if node is None:
        return prev
    next_node = node.next
    node.next = prev
    return reverse_list(next_node, node)
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "reverse_list用递归的方式反转单链表：每次调用先把当前node的next指针保存到next_node里，再把"
             "node.next指向prev(反转当前这一节的指向)，然后递归调用reverse_list(next_node, node)去处理剩下的"
             "链表，直到node为None时把prev(也就是反转后的新头节点)返回出来，逐层往回传，最终得到整条反转后的"
             "链表头。",
             ("把当前node的next指针保存到next_node里", "node.next指向prev(反转当前这一节的指向)",
              "反转后的新头节点)返回出来")),
            ("如果这个链表是通过循环insert构造出来的10万个节点的长链表，调用reverse_list(head)会发生什么？",
             "对10万个节点的链表调用reverse_list(head)，实际运行会在处理到大约1000层左右递归调用时抛出"
             "RecursionError: maximum recursion depth exceeded，程序直接崩溃退出，根本走不到把整条链表反转"
             "完成的那一步。",
             ("大约1000层左右递归调用时抛出RecursionError: maximum recursion depth exceeded", "程序直接崩溃退出",
              "走不到把整条链表反转完成的那一步")),
            ("具体是什么机制导致的？",
             "根因是reverse_list每处理一个节点就发起一次新的函数调用而不是复用当前调用帧，Python解释器默认的"
             "递归深度上限(sys.getrecursionlimit()默认通常是1000左右)会在链表长度超过这个上限时被触发；虽然"
             "这是尾递归的写法，但CPython解释器并不做尾调用优化(tail-call optimization)，每一层递归调用都会"
             "实实在在地在调用栈上压入一个新的栈帧，10万层调用意味着10万个栈帧。",
             ("Python解释器默认的递归深度上限(sys.getrecursionlimit()默认通常是1000左右)",
              "CPython解释器并不做尾调用优化(tail-call optimization)", "10万层调用意味着10万个栈帧")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复应该把递归改写成迭代版本，用一个while循环显式维护prev/next_node这两个指针逐步反转，彻底"
             "避免函数调用栈随链表长度线性增长；这类bug最容易在开发时只用几个节点的小样例做单元测试、从没构造"
             "过大规模真实数据量的链表/树/嵌套结构这种场景被忽略；系统性预防手段包括code review checklist里加"
             "一条凡是对链表/树等递归数据结构做递归遍历是否评估过真实数据规模下的递归深度上限，能否改写成迭代，"
             "以及单元测试里专门覆盖超过sys.getrecursionlimit()量级的大规模输入这类边界用例。",
             ("把递归改写成迭代版本", "只用几个节点的小样例做单元测试", "超过sys.getrecursionlimit()量级的大规模输入")),
        ),
        pitfall="很多人知道'递归可能栈溢出'这个常识，但会误以为这段代码是尾递归所以Python会自动优化掉栈增长，说不清楚CPython根本不做尾调用优化这个关键事实。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-10",
        cat=CAT,
        trigger="""面试官说：'这是读取配置项的工具函数，先说说它在做什么？'

```python
def read_config_value(path, key):
    f = open(path, 'r')
    data = json.load(f)
    if key not in data:
        raise KeyError(f'missing key: {key}')
    f.close()
    return data[key]
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "read_config_value先用open(path, 'r')打开配置文件得到文件对象f，用json.load(f)把内容解析成字典"
             "data，检查key是否在data里，如果不在就抛出KeyError；如果在，就调用f.close()关闭文件句柄，再把"
             "data[key]对应的值返回，看起来是一个打开文件-读取-关闭-返回值的标准四步流程。",
             ("用open(path, 'r')打开配置文件得到文件对象f", "检查key是否在data里，如果不在就抛出KeyError",
              "调用f.close()关闭文件句柄")),
            ("如果这个函数在一个长期运行的服务里被反复调用，每次调用传入的key在配置文件里恰好都不存在，运行"
             "足够多次之后会发生什么？",
             "每次调用只要key不在data里，代码就会在f.close()这行之前提前经由raise KeyError跳出函数，"
             "f.close()根本不会被执行，这个文件句柄就一直保持打开状态没有被释放；如果这个函数被反复调用几千次，"
             "进程持有的打开文件句柄数会不断累积，最终会撞上操作系统对单进程可打开文件描述符数量的上限，抛出"
             "OSError: [Errno 24] Too many open files。",
             ("在f.close()这行之前提前经由raise KeyError跳出函数，f.close()根本不会被执行",
              "进程持有的打开文件句柄数会不断累积", "OSError: [Errno 24] Too many open files")),
            ("具体是什么机制导致的？",
             "根因是f.close()被写在了检查key是否存在这个可能提前raise的代码路径之后，close操作和open操作之间"
             "没有用try/finally或者上下文管理器绑定在一起，只要中间任何一步抛出异常，close就会被跳过；这是一"
             "种典型的资源获取和资源释放没有被异常安全地配对的问题，只有走到异常路径才会暴露出资源没被释放这个"
             "隐患。",
             ("close操作和open操作之间没有用try/finally或者上下文管理器绑定在一起", "资源获取和资源释放没有被异常安全地配对",
              "只有走到异常路径才会暴露出资源没被释放这个隐患")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复应该用with open(path, 'r') as f:上下文管理器包住文件操作，让Python无论函数是正常return还是"
             "中途抛异常都能保证文件被自动关闭；这类bug最容易在资源释放代码写在函数末尾、而函数中间又存在提前"
             "return或raise的分支这种场景被忽略，开发时只跑通了正常路径的测试用例，没有专门为异常路径写资源"
             "泄漏检查；系统性预防手段包括code review checklist里加一条凡是手动调用close()释放资源的地方是否"
             "都用了with/try-finally，以及linter规则pylint的consider-using-with。",
             ("用with open(path, 'r') as f:上下文管理器包住文件操作", "只跑通了正常路径的测试用例，没有专门为异常路径写资源泄漏检查",
              "pylint的consider-using-with")),
        ),
        pitfall="很多人知道'文件要记得关'，但看这段代码时容易被确实写了f.close()这个表面现象骗过去，没意识到异常分支会跳过close这一行，说不出具体的Too many open files这个报错。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-11",
        cat=CAT,
        trigger="""面试官说：'这是一段异步计数代码，先说说它在做什么？'

```python
counter = {'value': 0}

async def increment():
    current = counter['value']
    await asyncio.sleep(0)   # 模拟一次真实的IO让出控制权
    counter['value'] = current + 1

async def main():
    await asyncio.gather(*[increment() for _ in range(100)])
    print(counter['value'])
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "increment协程先把counter['value']读到局部变量current里，然后await asyncio.sleep(0)模拟一次真实"
             "的IO操作，再把counter['value']写回current + 1；main函数用asyncio.gather同时并发发起100个"
             "increment协程，全部完成后打印counter['value']，直觉上100次自增之后这个值应该是100。",
             ("把counter['value']读到局部变量current里", "asyncio.gather同时并发发起100个increment协程",
              "100次自增之后这个值应该是100")),
            ("实际运行asyncio.run(main())，最终打印出来的counter['value']是100吗？",
             "实际打印出来的值几乎总是小于100，因为100个increment协程在await asyncio.sleep(0)这一行都会让出"
             "控制权，事件循环会去调度其他还没执行到写回那一步的协程，等好几个协程都已经把旧的current值读到了"
             "各自的局部变量里之后，它们才依次恢复执行、各自用自己手里那个已经过时的current + 1写回"
             "counter['value']，导致多次自增互相覆盖，最终值比100少。",
             ("实际打印出来的值几乎总是小于100", "都已经把旧的current值读到了各自的局部变量里",
              "多次自增互相覆盖，最终值比100少")),
            ("具体是什么机制导致的？",
             "很多人以为asyncio是单线程运行，所以不存在多线程那种数据竞争，但根因恰恰在于await这个关键字："
             "只要函数体里出现await，就意味着这里是一个可能被事件循环抢占、切换去执行其他协程的调度点；"
             "increment函数里读current和写回counter这两步之间插入了await asyncio.sleep(0)，这就制造出了一个"
             "和真正的多线程竞态在效果上等价的交错窗口，这是asyncio下经典的看起来是单线程、实际上照样有竞态"
             "条件的陷阱。",
             ("只要函数体里出现await，就意味着这里是一个可能被事件循环抢占、切换去执行其他协程的调度点",
              "看起来是单线程、实际上照样有竞态条件", "和真正的多线程竞态在效果上等价的交错窗口")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复可以用asyncio.Lock把读current、写回counter这整段临界区包起来，保证同一时刻只有一个协程能"
             "执行这段读写逻辑；这类bug最容易在开发者知道asyncio是单线程模型、想当然地认为不需要加锁这种认知"
             "误区下被忽略，功能测试如果只测单个协程调用完全不会暴露问题；系统性预防手段包括code review "
             "checklist里加一条凡是协程之间共享可变状态且读写逻辑中间存在await的地方是否需要asyncio.Lock保护，"
             "单元测试专门用asyncio.gather并发跑多个会修改共享状态的协程、断言最终结果等于预期次数。",
             ("用asyncio.Lock把读current、写回counter这整段临界区包起来", "开发者知道asyncio是单线程模型、想当然地认为不需要加锁",
              "断言最终结果等于预期次数")),
        ),
        pitfall="几乎所有人都知道'asyncio是单线程的'，正因为这个前提太深入人心，很少有人能立刻反应过来await本身就是一个并发切换点，说不出实际结果会小于100这个具体现象。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-12",
        cat=CAT,
        trigger="""面试官说：'这是缓存清理过期条目的函数，先说说它在做什么？'

```python
def purge_expired(cache: dict, now: float) -> None:
    for key, (value, expire_at) in cache.items():
        if expire_at < now:
            del cache[key]
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "purge_expired遍历cache这个字典的每一个(key, (value, expire_at))键值对，检查每一项的过期时间"
             "expire_at是否早于now，如果已经过期就用del cache[key]把这一项从字典里删掉，目的是清理掉所有已"
             "过期的缓存条目。",
             ("遍历cache这个字典的每一个(key, (value, expire_at))键值对", "检查每一项的过期时间expire_at是否早于now",
              "清理掉所有已过期的缓存条目")),
            ("如果cache里有5个条目，其中恰好有2个已经过期，实际调用purge_expired(cache, now)会发生什么？",
             "并不会正常清理掉这2个过期条目然后返回，而是在处理到第一个过期条目、执行del cache[key]删除它之后，"
             "for循环继续尝试从cache.items()这个正在被迭代的视图里取下一项时，会直接抛出RuntimeError: "
             "dictionary changed size during iteration，函数在删掉第一个过期条目后就崩溃退出了，第二个过期"
             "条目根本没有机会被处理到。",
             ("执行del cache[key]删除它之后", "抛出RuntimeError: dictionary changed size during iteration",
              "第二个过期条目根本没有机会被处理到")),
            ("具体是什么机制导致的？",
             "根因是for key, ... in cache.items()这个循环持有的是字典的一个动态视图(view)，Python在迭代字典"
             "时会做修改检测，一旦在迭代过程中字典的大小发生变化，下一次尝试从视图里取值时就会检测到这个不"
             "一致并主动抛异常来防止出现未定义行为；这段代码在for循环体内部直接对正在被迭代的同一个cache字典"
             "做del操作，正是触发这个检测机制的典型写法。",
             ("持有的是字典的一个动态视图(view)", "Python在迭代字典时会做修改检测",
              "在for循环体内部直接对正在被迭代的同一个cache字典做del操作")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复可以先用list(cache.items())拷贝出一份不再和cache共享的独立列表再遍历这份拷贝去做删除，或者"
             "先收集所有要删除的key到一个单独的列表里、遍历结束之后再统一执行删除；这类bug最容易在先写了正常"
             "读取遍历的逻辑、后面顺手在循环体里加了一行删除操作这种增量修改场景被忽略；系统性预防手段是"
             "code review checklist里加一条凡是遍历字典/集合的循环体内是否存在对同一个容器的增删操作，以及"
             "单元测试要专门构造至少存在2个需要被删除的条目这种用例。",
             ("list(cache.items())拷贝出一份不再和cache共享的独立列表", "先写了正常读取遍历的逻辑、后面顺手在循环体里加了一行删除操作",
              "至少存在2个需要被删除的条目")),
        ),
        pitfall="很多人知道'不要在遍历字典时修改它'这条规则，但说不清楚背后是字典的修改检测机制，也容易忽略只测1个待删除条目可能侥幸不报错这个测试设计上的细节。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-13",
        cat=CAT,
        trigger="""面试官说：'这是一个懒加载单例，先说说它在做什么？'

```python
_instance = None

def get_config():
    global _instance
    if _instance is None:
        time.sleep(0.001)  # 模拟磁盘IO耗时
        _instance = load_config_from_disk()
    return _instance
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "get_config实现了一个懒加载单例模式：第一次调用时检查全局变量_instance是否为None，如果是就调用"
             "load_config_from_disk()从磁盘加载配置并赋值给_instance，之后不管调用多少次都直接复用这个已经"
             "加载好的_instance返回，避免每次调用都重复读盘。",
             ("检查全局变量_instance是否为None", "调用load_config_from_disk()从磁盘加载配置", "避免每次调用都重复读盘")),
            ("如果有10个线程在进程刚启动、_instance还是None的时候几乎同时第一次调用get_config()，会发生什么？",
             "10个线程几乎同时执行到if _instance is None这行，由于此时_instance确实还是None，这10个线程会"
             "全部判断条件为True并都进入if代码块；接下来大家都会各自调用一次load_config_from_disk()从磁盘重新"
             "加载配置，最终_instance会被这10次调用中最后一个完成赋值的结果覆盖，配置文件被读了10次而不是"
             "预期的1次。",
             ("这10个线程会全部判断条件为True并都进入if代码块", "各自调用一次load_config_from_disk()从磁盘重新加载配置",
              "配置文件被读了10次而不是预期的1次")),
            ("具体是什么机制导致的？",
             "根因还是TOCTOU竞态：if _instance is None这个检查和_instance = load_config_from_disk()这个赋值"
             "之间存在时间窗口，多个线程可以同时通过检查、都还没来得及赋值；很多人误以为Python有GIL(全局解释"
             "器锁)保护，所以多线程下不会有竞态问题，但GIL只保证单条字节码指令级别的原子性，并不会把检查+赋值"
             "这种跨越多条语句、中间还包含耗时IO调用的复合操作自动打包成一个原子操作。",
             ("if _instance is None这个检查和_instance = load_config_from_disk()这个赋值之间存在时间窗口",
              "误以为Python有GIL(全局解释器锁)保护，所以多线程下不会有竞态问题", "GIL只保证单条字节码指令级别的原子性")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复要引入一把threading.Lock，并且用双重检查锁定(double-checked locking)模式：先不加锁检查一次"
             "_instance是否为None，如果是None再进入with lock加锁之后二次检查一次_instance是否还是None，确认"
             "仍是None才真正调用load_config_from_disk()赋值；这类bug最容易在开发者知道要写懒加载单例、但误以"
             "为Python的GIL天然帮自己处理好了并发安全这种认知误区下被忽略；系统性预防手段包括code review "
             "checklist里加一条凡是模块级懒加载单例是否显式加了锁，以及并发单元测试用多个线程同时首次调用"
             "get_config()、断言load_config_from_disk()只被实际调用了1次。",
             ("双重检查锁定(double-checked locking)模式", "误以为Python的GIL天然帮自己处理好了并发安全",
              "断言load_config_from_disk()只被实际调用了1次")),
        ),
        pitfall="很多人知道'单例要考虑线程安全'，但普遍高估了GIL的保护范围，误以为只要是Python代码多线程就自动安全，说不出GIL只保证单条字节码原子性这个关键限定。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-14",
        cat=CAT,
        trigger="""面试官说：'这是分页翻页的实现，接口文档规定page从1开始编号，先说说它在做什么？'

```python
def get_page(items: list, page: int, page_size: int) -> list:
    # page 从 1 开始计数(接口文档约定)
    start = page * page_size
    end = start + page_size
    return items[start:end]

def get_all_pages(items, page_size):
    pages = []
    page = 1
    while True:
        chunk = get_page(items, page, page_size)
        if not chunk:
            break
        pages.append(chunk)
        page += 1
    return pages
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "get_page计算出该页对应的起始和结束下标，start = page * page_size，end = start + page_size，然后"
             "用切片items[start:end]取出这一页对应的数据；get_all_pages不断递增page去调用get_page，直到某次"
             "返回的chunk是空列表就认为翻到头了，停止循环，把收集到的所有非空页拼成pages列表返回。",
             ("start = page * page_size，end = start + page_size", "用切片items[start:end]取出这一页对应的数据",
              "某次返回的chunk是空列表就认为翻到头了")),
            ("如果items有100条记录，page_size=20，调用方从get_page(items, 1, 20)开始翻页，实际会得到什么结果？",
             "get_page(items, 1, 20)算出start=20, end=40，实际返回的是items[20:40]，也就是原本第2页的数据，"
             "而items[0:20]这真正的第一页数据在整个翻页过程里完全没有被任何一次调用返回过；继续翻页到page=5"
             "时，items[100:120]因为items总共只有100条而变成空列表，循环判定翻到头提前结束，最终get_all_pages"
             "只收集到4页，总共80条记录，比原始的100条少了最前面20条。",
             ("实际返回的是items[20:40]", "items[0:20]这真正的第一页数据在整个翻页过程里完全没有被任何一次调用返回过",
              "总共80条记录，比原始的100条少了最前面20条")),
            ("具体是哪一行、什么机制导致的？",
             "根因在start = page * page_size这一行：这个计算式假设page是从0开始编号的，但接口文档和调用方"
             "约定的是page从1开始编号，page=1时理应对应items[0:page_size]这第一页，可代码里page=1却算出"
             "start=page_size，相当于所有页码都系统性地错位了一页；这种语义错位但语法上完全合法、不抛异常的"
             "bug比直接崩溃更隐蔽。",
             ("这个计算式假设page是从0开始编号的", "接口文档和调用方约定的是page从1开始编号",
              "语义错位但语法上完全合法、不抛异常的bug比直接崩溃更隐蔽")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复要把start的计算改成start = (page - 1) * page_size，让1-indexed的页码正确对应到0-indexed的"
             "切片下标；这类bug最容易在实现函数的人假设是0-indexed分页、写接口文档或者调用这个函数的人假设是"
             "1-indexed分页这种前后端/模块间约定不一致又没有写进代码注释的场景被忽略；系统性预防手段包括"
             "code review checklist里加一条分页函数的页码是0-indexed还是1-indexed是否在函数签名/docstring里"
             "显式写明，以及单元测试要专门断言翻遍所有页之后拼起来的记录总数等于原始items长度、且第一条记录和"
             "最后一条记录都在其中。",
             ("start = (page - 1) * page_size", "前后端/模块间约定不一致又没有写进代码注释",
              "翻遍所有页之后拼起来的记录总数等于原始items长度、且第一条记录和最后一条记录都在其中")),
        ),
        pitfall="很多人测试分页只验证循环能正常结束、没有报错，不会去核对翻完所有页之后总记录数和第一条/最后一条记录是否对得上，这种bug因为不抛异常、看着一切正常，反而比直接崩溃的bug更容易被放过。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-15",
        cat=CAT,
        trigger="""面试官说：'这是一个日志打印函数，先说说它在做什么？'

```python
import time

def log_event(msg, ts=time.time()):
    print(f'[{ts}] {msg}')
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "log_event把msg和一个时间戳ts拼成[ts] msg这样的格式打印出来，函数签名是"
             "def log_event(msg, ts=time.time())，字面上看起来的意图是：如果调用者省略ts，就自动用调用那一刻"
             "的当前时间戳来打时间戳。",
             ("把msg和一个时间戳ts拼成[ts] msg这样的格式打印出来", "def log_event(msg, ts=time.time())",
              "自动用调用那一刻的当前时间戳")),
            ("如果这个函数所在的服务进程在00:00:00启动时被import一次，然后在24小时后的第二天00:00:00才第一次"
             "以log_event('服务正常')这种不传ts的方式被调用，打印出来的[ts]会是什么时间？",
             "打印出来的[ts]仍然是服务进程启动、模块被import、这个函数定义那一刻的时间戳，而不是24小时后真正"
             "调用log_event的那个时刻，也就是说这条服务正常的日志会被打上一个整整晚了24小时的错误时间戳，"
             "看起来像是进程刚启动时打的日志。",
             ("仍然是服务进程启动、模块被import、这个函数定义那一刻", "打上一个整整晚了24小时的错误时间戳",
              "看起来像是进程刚启动时打的日志")),
            ("具体是什么机制导致的？",
             "根因和可变默认参数陷阱背后的机制是同一条Python规则：函数的默认参数值只在def语句被执行的那一刻"
             "求值一次，此后这个函数不管被调用多少次、不管间隔多久，只要调用时省略了ts参数，用的都是定义时就"
             "已经算好并固定下来的那个time.time()返回值；这里ts虽然是不可变的float，但只在定义时求值一次这条"
             "规则本身依然生效，导致它被固化成了一个随进程启动时刻定格的常量，而不是每次调用都重新计算的"
             "当前时间。",
             ("函数的默认参数值只在def语句被执行的那一刻求值一次", "只在定义时求值一次这条规则本身依然生效",
              "随进程启动时刻定格的常量，而不是每次调用都重新计算的当前时间")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复方式和可变默认参数陷阱一样，把默认值改成None，在函数体内部判断if ts is None: ts = time.time()，"
             "保证每次省略该参数调用时都在调用当下重新取一次时间戳；这类bug最容易在长期运行的服务进程里、"
             "函数刚写完马上手动测试一次感觉输出正常这种场景被忽略，因为刚定义完立刻调用时定义时刻和调用时刻"
             "几乎重合，时间戳看起来完全正确；系统性预防手段包括code review checklist里加一条凡是默认参数用了"
             "time.time()/datetime.now()这类会随时间变化的调用作为默认值都要改成None+函数体内部判断，以及"
             "单元测试专门设计先import模块等待几秒钟之后再调用函数、断言返回的时间戳接近调用时刻而不是import时刻"
             "这类用例。",
             ("if ts is None: ts = time.time()", "刚定义完立刻调用时定义时刻和调用时刻几乎重合",
              "先import模块等待几秒钟之后再调用函数、断言返回的时间戳接近调用时刻而不是import时刻")),
        ),
        pitfall="很多人只知道'可变默认参数(列表/字典)有坑'，误以为像time.time()这种返回不可变float的调用作为默认值就安全，没意识到定义时只求值一次这条规则和参数是否可变完全无关。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-16",
        cat=CAT,
        trigger="""面试官说：'这是一个批量解析日志的函数，先说说它在做什么？'

```python
def parse_records(raw_lines: list[str]) -> list[dict]:
    records = []
    for line in raw_lines:
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "parse_records遍历raw_lines里的每一行，尝试用json.loads把这一行解析成字典并追加到records列表里；"
             "如果某一行解析失败抛出异常，就用except Exception: pass直接吞掉这个异常什么都不做，继续处理下"
             "一行，最后把所有成功解析的记录列表返回。",
             ("尝试用json.loads把这一行解析成字典并追加到records列表里", "except Exception: pass直接吞掉这个异常什么都不做",
              "把所有成功解析的记录列表返回")),
            ("如果上游系统在某次故障期间往raw_lines里混入了几行被截断的、不完整的JSON(本该有1万行完整数据，"
             "其中50行因为写入中途进程被杀掉而变成了半截JSON)，parse_records处理完之后，调用方会看到什么现象？",
             "parse_records会正常返回一个长度是9950的records列表，不会抛出任何异常、不会打印任何警告日志，"
             "调用方从函数返回值和调用过程中完全看不出这9950条和预期的10000条之间存在50条数据丢失；这50条被"
             "截断的记录就会在没有任何痕迹的情况下永久性地从数据管道里消失。",
             ("会正常返回一个长度是9950的records列表，不会抛出任何异常", "完全看不出这9950条和预期的10000条之间存在50条数据丢失",
              "在没有任何痕迹的情况下永久性地从数据管道里消失")),
            ("具体是哪一行、什么机制导致的？",
             "根因在except Exception: pass这一行：它捕获了包括json.JSONDecodeError在内的几乎所有可能的异常"
             "类型，捕获之后又只用一个pass什么都不做，既不记录日志、不上报监控指标、也不把失败的原始行保留下"
             "来供事后排查；这是异常处理和异常吞没之间的一个界限问题，把原本应该显式暴露的数据质量问题伪装成了"
             "一切正常。",
             ("捕获了包括json.JSONDecodeError在内的几乎所有可能的异常类型",
              "既不记录日志、不上报监控指标、也不把失败的原始行保留下来供事后排查",
              "把原本应该显式暴露的数据质量问题伪装成了一切正常")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复至少要在except块里加上日志记录，比如logger.warning记录下具体是第几行、原始内容摘要、异常"
             "信息，理想情况下还要维护一个失败计数器并对接监控告警；这类bug最容易在为了让批处理脚本能跑通、"
             "不因为个别脏数据就整体失败而图省事加了一个宽泛的except Exception: pass这种赶工场景下被忽略；"
             "系统性预防手段包括code review checklist里明确一条禁止裸露的except Exception: pass，任何捕获异常"
             "的地方必须至少记录日志或做计数，单元测试要专门构造输入里混入N行损坏数据的用例，断言返回结果的"
             "条数、以及日志/监控指标里记录的失败条数，都和预期的N一致。",
             ("logger.warning记录下具体是第几行、原始内容摘要、异常信息",
              "禁止裸露的except Exception: pass，任何捕获异常的地方必须至少记录日志或做计数",
              "断言返回结果的条数、以及日志/监控指标里记录的失败条数，都和预期的N一致")),
        ),
        pitfall="很多人对着这段代码只会说try except写得挺规范，看不出问题在于pass这个空动作彻底销毁了失败信号，说不出9950条这个具体数字，也想不到要用监控指标或者留存原始失败行来兜底。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-17",
        cat=CAT,
        trigger="""面试官说：'这是双向转账的加锁逻辑，先说说它在做什么？'

```python
lock_a = threading.Lock()
lock_b = threading.Lock()

def transfer_a_to_b(amount):
    with lock_a:
        with lock_b:
            do_transfer(amount)

def transfer_b_to_a(amount):
    with lock_b:
        with lock_a:
            do_transfer(-amount)
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "transfer_a_to_b先获取lock_a再获取lock_b，拿到两把锁之后执行do_transfer(amount)完成转账；"
             "transfer_b_to_a则反过来，先获取lock_b再获取lock_a，拿到两把锁之后执行do_transfer(-amount)，"
             "两个函数各自都用with语句嵌套获取两把锁来保护转账操作的原子性。",
             ("先获取lock_a再获取lock_b", "先获取lock_b再获取lock_a", "各自都用with语句嵌套获取两把锁来保护转账操作的原子性")),
            ("如果线程1调用transfer_a_to_b(100)、线程2几乎同时调用transfer_b_to_a(50)，两个线程分别成功获取"
             "了各自函数里的第一把锁之后，会发生什么？",
             "线程1成功获取了lock_a，正准备获取lock_b；线程2成功获取了lock_b，正准备获取lock_a；此时线程1"
             "想要的lock_b被线程2占着，线程2想要的lock_a被线程1占着，两个线程都会阻塞在各自的第二个with "
             "lock_b/with lock_a这一行，程序表现为整体挂起、既不报错也不继续执行，这两个转账操作都不会完成。",
             ("线程1想要的lock_b被线程2占着，线程2想要的lock_a被线程1占着", "两个线程都会阻塞在各自的第二个with lock_b/with lock_a这一行",
              "程序表现为整体挂起、既不报错也不继续执行")),
            ("具体是什么机制导致的？",
             "根因是transfer_a_to_b和transfer_b_to_a这两个函数对lock_a和lock_b的获取顺序不一致：一个是"
             "lock_a→lock_b，另一个是lock_b→lock_a；当两个线程分别持有对方需要的那把锁、又同时在等待对方释放"
             "时，就形成了经典的循环等待(circular wait)，只有当两个方向的转账函数恰好在时间上交错到各自拿到"
             "第一把锁、都还没拿到第二把锁这个窗口时才会触发，这也是这类bug时有时无、难以稳定复现的原因。",
             ("对lock_a和lock_b的获取顺序不一致", "形成了经典的循环等待(circular wait)", "时有时无、难以稳定复现")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复的标准做法是给所有锁定义一个全局固定的获取顺序，让transfer_b_to_a也改成先获取lock_a再获取"
             "lock_b，从根本上消除循环等待的可能；这类bug最容易在转账/交换这类涉及两个对象加锁的操作、两个"
             "方向分别由不同开发者各自都局部合理地按自己函数里参数的顺序加锁这种场景被忽略；系统性预防手段"
             "包括code review checklist里加一条凡是同时获取多把锁的地方是否所有代码路径都遵循同一个全局锁"
             "顺序，以及使用超时锁(acquire加timeout)加日志告警来兜底。",
             ("给所有锁定义一个全局固定的获取顺序", "各自都局部合理地按自己函数里参数的顺序加锁",
              "使用超时锁(acquire加timeout)加日志告警来兜底")),
        ),
        pitfall="很多人知道'死锁'这个名词，但具体看这段代码时容易忽略两个函数锁的获取顺序是相反的，说不清楚循环等待这个死锁条件在这里被满足的具体机制，也说不出这种bug时有时无的原因是时序窗口。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-18",
        cat=CAT,
        trigger="""面试官说：'这是带重试的数据加载逻辑，先说说它在做什么？'

```python
def parse_lines(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

def load_with_retry(path, max_retries=3):
    lines = parse_lines(path)
    for attempt in range(max_retries):
        try:
            return process(lines)
        except TransientError:
            continue
    raise RuntimeError('加载失败')
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "parse_lines是一个生成器函数，逐行读取path文件并yield去掉首尾空白后的内容；load_with_retry先"
             "调用parse_lines(path)拿到一个生成器对象lines，然后在一个最多重试max_retries次的循环里，每次都"
             "调用process(lines)尝试处理这份数据，如果process抛出TransientError就continue进入下一次重试，"
             "直到某次process成功就直接return结果，如果重试次数用完了都没成功就抛出RuntimeError。",
             ("逐行读取path文件并yield去掉首尾空白后的内容", "调用parse_lines(path)拿到一个生成器对象lines",
              "如果process抛出TransientError就continue进入下一次重试")),
            ("如果process(lines)在第一次调用时因为一次短暂的网络抖动抛出了TransientError(这时候lines这个"
             "生成器其实已经被process内部完整遍历过一遍了)，代码走到第二次重试调用process(lines)时，会发生"
             "什么？",
             "第二次调用process(lines)传进去的还是同一个lines生成器对象，但这个生成器在第一次调用时已经被"
             "完整迭代到耗尽(exhausted)状态；第二次再对它调用process，process内部对lines做遍历时会立刻得到"
             "一个空序列，看起来像是文件里什么内容都没有，而不是重新完整地把文件内容再读一遍重试处理。",
             ("这个生成器在第一次调用时已经被完整迭代到耗尽(exhausted)状态", "process内部对lines做遍历时会立刻得到",
              "看起来像是文件里什么内容都没有")),
            ("具体是什么机制导致的？",
             "根因是lines = parse_lines(path)这一行只在函数最开始创建了一次生成器对象，而重试循环里反复复用"
             "的是这同一个生成器实例；Python的生成器是一次性的惰性迭代器，一旦被完整消费过，它的内部状态就"
             "停在末尾，之后不管再调用多少次遍历它的代码，得到的都是空的遍历结果而不是重新从文件开头读取；"
             "重试这个设计意图默认假设每次重试都能拿到一份完整可用的数据源，但这里被复用的是同一个已经耗尽的"
             "生成器。",
             ("lines = parse_lines(path)这一行只在函数最开始创建了一次生成器对象", "生成器是一次性的惰性迭代器",
              "假设每次重试都能拿到一份完整可用的数据源，但这里被复用的是同一个已经耗尽的生成器")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复要把lines = parse_lines(path)这一行移到重试循环内部，让每次attempt都重新调用parse_lines(path)"
             "得到一个全新的生成器，或者干脆在最外层直接把parse_lines(path)的结果用list()物化成一个真正的"
             "列表再传给process；这类bug最容易在把一个返回列表的函数改造成生成器以节省内存、但忘记检查所有"
             "下游调用方是否假设了这个数据源可以被多次完整遍历这种重构场景被忽略；系统性预防手段包括"
             "code review checklist里加一条凡是把某个数据源函数改成生成器是否核对过所有调用方是否存在多次"
             "遍历同一个生成器实例的重试/重放逻辑，单元测试要专门模拟process第一次调用抛TransientError触发"
             "重试这个场景，断言第二次调用process时拿到的数据和第一次完全一致，而不是空的。",
             ("移到重试循环内部，让每次attempt都重新调用parse_lines(path)得到一个全新的生成器",
              "把一个返回列表的函数改造成生成器以节省内存、但忘记检查所有下游调用方",
              "断言第二次调用process时拿到的数据和第一次完全一致，而不是空的")),
        ),
        pitfall="很多人知道'生成器只能遍历一次'这个知识点，但放到这个带重试逻辑的具体场景里就想不起来去检查lines是不是在循环外面创建的，容易只盯着process内部的重试逻辑对不对，忽视了传进去的数据源本身已经失效。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-19",
        cat=CAT,
        trigger="""面试官说：'这是按VIP标记统计人数的函数，先说说它在做什么？'

```python
def count_by_flag(records: list[dict]) -> dict:
    counts = {}
    for r in records:
        flag = r['is_vip']
        counts[flag] = counts.get(flag, 0) + 1
    return counts
```

（补充说明：上游有的记录用True/False表示is_vip，有的历史遗留记录用1/0表示is_vip）
""",
        chain=(
            ("这段代码整体在做什么？",
             "count_by_flag遍历records里的每一条记录，取出is_vip这个字段的值作为flag，用"
             "counts[flag] = counts.get(flag, 0) + 1这种经典的计数字典写法，把每个不同的flag值出现的次数分别"
             "累加到counts字典里，目的是统计有多少条记录是VIP(is_vip为真)、多少条不是。",
             ("取出is_vip这个字段的值作为flag", "counts[flag] = counts.get(flag, 0) + 1这种经典的计数字典写法",
              "统计有多少条记录是VIP(is_vip为真)、多少条不是")),
            ("如果records里有3条记录的is_vip是True(布尔值)，另外有2条历史遗留记录的is_vip是整数1，最终counts"
             "字典会是什么样？",
             "最终counts字典里只会出现一个键True，对应的计数值是3+2=5，而不是像很多人以为的那样counts里会分别"
             "有True: 3和1: 2这两个独立的键值对，这3条is_vip=True的记录和这2条is_vip=1的记录，在这个统计里被"
             "完全合并成了同一类。",
             ("只会出现一个键True", "对应的计数值是3+2=5", "被完全合并成了同一类")),
            ("具体是什么机制导致的？",
             "根因是Python里bool是int的子类，True的值在数值和哈希上都完全等同于1(True == 1为True，"
             "hash(True) == hash(1))；字典的键查找是基于__eq__和__hash__来判断两个键是否是同一个键的，counts."
             "get(flag, 0)在flag先是True、后是1这两种情况下会认为这是同一个键；很多人下意识以为字典键是靠"
             "类型+值一起区分的，但Python字典键的等价判断只看__eq__和__hash__，不关心两个对象是不是同一个类型。",
             ("bool是int的子类，True的值在数值和哈希上都完全等同于1", "字典的键查找是基于__eq__和__hash__来判断两个键是否是同一个键的",
              "Python字典键的等价判断只看__eq__和__hash__，不关心两个对象是不是同一个类型")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复要在统计之前先把flag统一归一化成同一种类型，比如统一用bool(flag)强制转换成True/False再计数，"
             "或者在数据接入层就把历史遗留的1/0字段清洗成标准的True/False；这类bug最容易在系统迭代过程中数据"
             "格式发生过变化，早期用0/1表示布尔字段、后来改成规范的True/False，但历史数据没有被完全回填清洗"
             "这种场景被忽略；系统性预防手段包括code review checklist里加一条凡是对可能存在类型不一致历史"
             "数据的字段做分组统计是否在统计前做了类型归一化，单元测试专门构造混合True和1、混合False和0这类"
             "用例。",
             ("统一用bool(flag)强制转换成True/False再计数", "早期用0/1表示布尔字段、后来改成规范的True/False",
              "混合True和1、混合False和0")),
        ),
        pitfall="很多人不知道Python里bool是int的子类、True和1在字典键里是同一个键这个具体规则，看到这段代码时完全想不到新老数据格式混用会导致统计结果被静默合并。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-20",
        cat=CAT,
        trigger="""面试官说：'这是计算下一次任务运行时间的函数，先说说它在做什么？'

```python
from datetime import timedelta
import pytz

def schedule_next_run(last_run, tz_name: str):
    tz = pytz.timezone(tz_name)
    local_last_run = last_run.astimezone(tz)
    next_run = local_last_run + timedelta(hours=24)
    return next_run
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "schedule_next_run把last_run这个带时区信息的时间转换到tz_name对应的本地时区得到local_last_run，"
             "然后给它加上timedelta(hours=24)得到next_run，目的是计算从上次运行时间起、24小时之后的下一次"
             "运行时间，用于类似每天定时跑一次的调度场景。",
             ("把last_run这个带时区信息的时间转换到tz_name对应的本地时区", "加上timedelta(hours=24)得到next_run",
              "计算从上次运行时间起、24小时之后的下一次运行时间")),
            ("如果tz_name是'America/New_York'，last_run恰好是当年美国夏令时开始前一天的本地时间(凌晨1点)，"
             "调用schedule_next_run后，next_run对应的本地时间和24小时之后的第二天同一时刻相比，会不会对得上？",
             "next_run算出来的本地时间会比预期晚1个小时：因为timedelta(hours=24)是纯粹的24个物理小时的时间"
             "跨度，而次日凌晨2点前后正好是美国东部夏令时跳过一小时的切换点，本地墙上时钟从这一天开始少了一"
             "个小时，24个物理小时之后对应的本地时刻会显示成比看起来该是的时刻晚1小时，而且这个偏移在代码逻辑"
             "本身完全不报错。",
             ("next_run算出来的本地时间会比预期晚1个小时", "本地墙上时钟从这一天开始少了一个小时",
              "这个偏移在代码逻辑本身完全不报错")),
            ("具体是什么机制导致的？",
             "根因是timedelta(hours=24)操作的是经过的物理时间量，而不是本地墙上时钟的天数，这两者只有在没有"
             "夏令时切换的时间段里才恰好相等；一旦这24小时的区间跨越了DST的春季拨快边界，物理上过去的24小时"
             "对应的本地时间就会比原来同一墙钟时刻晚1小时；这类bug只在调度任务的运行时间恰好落在DST切换日附近"
             "这个特定条件下才会触发，属于典型的低频、时间窗口极窄的隐蔽bug。",
             ("timedelta(hours=24)操作的是经过的物理时间量，而不是本地墙上时钟的天数",
              "一旦这24小时的区间跨越了DST的春季拨快边界", "只在调度任务的运行时间恰好落在DST切换日附近这个特定条件下才会触发")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复应该用支持DST感知的日期运算，比如对日历日期加1天再重新本地化，而不是对已经本地化的时间直接"
             "加timedelta(hours=24)这种物理时间量；这类bug最容易在开发和测试时都在不跨越DST切换日的日子里跑、"
             "服务器又经常用UTC时区所以开发者对本地时区的DST规则没有直觉这种场景被忽略；系统性预防手段包括"
             "code review checklist里加一条凡是对带时区的datetime做加减天数操作是否用的是DST感知的日历运算，"
             "单元测试要专门构造last_run恰好落在DST切换日前一天这个边界用例，断言next_run对应的本地时刻和"
             "预期的本地时刻分钟数完全一致。",
             ("用支持DST感知的日期运算", "服务器又经常用UTC时区所以开发者对本地时区的DST规则没有直觉",
              "last_run恰好落在DST切换日前一天")),
        ),
        pitfall="很多人知道'时区处理容易出bug'这句笼统的话，但具体问到DST切换会不会导致24小时之后本地时刻偏移时说不清楚方向，也想不到这类bug只在春季/秋季两个特定切换日附近才会触发这个低频窗口特征。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-sh-dbg-21",
        cat=CAT,
        trigger="""面试官说：'这是渲染欢迎语模板的函数，先说说它在做什么？'

```python
def render_welcome(template: str, user_data: dict) -> str:
    return template.format(**user_data)

# 调用示例：render_welcome('欢迎你，{nickname}！', user_profile_dict)
```
""",
        chain=(
            ("这段代码整体在做什么？",
             "render_welcome用template.format(**user_data)把user_data字典里的每个键值对拆包成关键字参数，去"
             "填充template这个字符串模板里形如{name}这样的占位符，目的是让调用方可以自定义欢迎语模板，同时把"
             "用户提交的昵称等数据动态填进去。",
             ("用template.format(**user_data)把user_data字典里的每个键值对拆包成关键字参数",
              "去填充template这个字符串模板里形如{name}这样的占位符", "让调用方可以自定义欢迎语模板")),
            ("如果user_profile_dict这个user_data里除了nickname之外还包含一个敏感字段secret_key，而某个用户"
             "故意把自己的nickname设置成字面量字符串'{secret_key}'，之后这个昵称字符串在下游某个日志/展示"
             "环节又被当成新的模板送进了另一次format调用，会发生什么？",
             "第一次render_welcome调用本身只会把nickname的值原样填进{nickname}占位符，正常返回一句包含"
             "字面文本{secret_key}的欢迎语；但当这段包含{secret_key}字面量的字符串在下游被当成模板再次"
             "format且传入的对象里恰好又带有secret_key这个键时，{secret_key}就会被解析成一个真正的占位符，"
             "从而把本不该展示给用户的secret_key值渲染出来，造成信息泄露。",
             ("正常返回一句包含字面文本{secret_key}的欢迎语", "{secret_key}就会被解析成一个真正的占位符",
              "造成信息泄露")),
            ("具体是什么机制导致的？",
             "根因在于template.format(**user_data)这一步把用户完全可控的输入当成了可信的格式化字符串来"
             "处理，Python的str.format语法本身支持{attribute}、{key}这类占位符去访问传入对象的属性或字典的"
             "键，一旦用户输入的内容里恰好包含花括号包裹的合法占位符语法，而这段内容后续又被送进某个format"
             "调用，就会触发意料之外的占位符替换；这类问题的本质是把用户输入的数据和系统信任的格式化模板这"
             "两个应该严格区分的东西混在了同一个字符串里处理。",
             ("把用户完全可控的输入当成了可信的格式化字符串来处理", "一旦用户输入的内容里恰好包含花括号包裹的合法占位符语法",
              "把用户输入的数据和系统信任的格式化模板这两个应该严格区分的东西混在了同一个字符串里处理")),
            ("怎么修复，这类问题什么场景容易被忽略，有没有系统性预防手段？",
             "修复思路是对不可信的用户输入在填入模板前后都不再把它当成模板去format，只把它当成普通字符串值"
             "使用，避免format(**user_data)这种把整个字典无差别解包传入的写法把额外的敏感字段暴露给模板引擎；"
             "这类bug最容易在开发者图方便直接把整个用户/业务对象**解包传给format、而不是显式只传模板真正需要"
             "的那几个字段这种场景被忽略；系统性预防手段包括code review checklist里加一条凡是对外部/用户可控"
             "的字符串调用format或者把它当模板使用之前是否明确了这段输入不会包含未经校验的占位符语法，单元"
             "测试专门构造nickname输入是字面量的{secret_key}这样的字符串这类恶意输入用例。",
             ("对不可信的用户输入在填入模板前后都不再把它当成模板去format",
              "开发者图方便直接把整个用户/业务对象**解包传给format、而不是显式只传模板真正需要的那几个字段",
              "nickname输入是字面量的{secret_key}这样的字符串")),
        ),
        pitfall="很多人看到template.format(**user_data)会觉得这就是标准写法没什么问题，说不出**解包会把user_data里所有字段无差别地暴露给format这个具体风险，也想不到用户输入本身可能在下游被当成模板二次format这种跨模块场景。",
        real_world_link="",
    ),
]


def _self_test() -> None:
    assert 19 <= len(BANK) <= 23, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复id"
    assert all(i.startswith("dp-sh-dbg-") for i in ids), "id前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的条目"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的条目"
    assert all(dp.trigger for dp in BANK), "存在缺失trigger的条目"
    for dp in BANK:
        answers = [ref for (_q, ref, _k) in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 采分关键词未能在参考答案里全部命中: {scores}"
    print(f"[PASS] gate3_debugging_round: {len(BANK)}个DeepPoint 自洽性检查通过")


if __name__ == "__main__":
    _self_test()
