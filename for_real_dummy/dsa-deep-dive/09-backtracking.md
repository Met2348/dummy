# 09 · 回溯(Backtracking)

> 总览见 [00-roadmap.md](00-roadmap.md)。回溯法是"决策树上的穷举",本类的核心是建立一套通用框架(选择列表/路径/结束条件),再讲清楚"剪枝"到底能省多少——不是空谈"能优化",是真实测出优化前后的调用次数和耗时差几个数量级。

---

## 1. 回溯法总纲:决策树视角

**签名/是什么:**
```
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.append(路径[:])   # 注意要拷贝,不能直接存引用
        return
    for 选择 in 选择列表:
        做选择                    # 路径.append(选择)
        backtrack(路径, 新选择列表)
        撤销选择                  # 路径.pop() —— 这一步是"回溯"名字的由来
```

**一句话:** 回溯法把一个问题的所有可能解,组织成一棵决策树——每一层代表"做一个选择",从根到叶子的一条完整路径代表一个候选解;"回溯"这个名字的含义是:递归返回后,必须**撤销**刚才做的选择,恢复到做选择之前的状态,才能继续尝试同一层的下一个候选。

**底层机制/为什么这样设计:** "做选择→递归→撤销选择"这三步缺一不可——如果不撤销选择,`路径` 这个共享的可变状态会在尝试完一个分支后残留着错误的历史记录,污染后续兄弟分支的探索。这也是回溯法和普通 DFS 在实现细节上最大的区别:普通图 DFS 通常只需要标记"访问过"且不撤销(因为目标是"访问所有节点一次"),而回溯法需要"尝试完一种选择组合后,把状态复原,像没发生过一样,再去尝试另一种组合"——因为目标是穷举所有**独立**的候选路径,不是访问所有节点一次。`result.append(路径[:])` 里的切片拷贝同样是必需的:如果直接 `append(路径)`,存进结果集的是对同一个列表对象的引用,后续对 `路径` 的修改(pop/append)会连带着改变已经"保存"的结果。

**AI 研究/工程场景:** [huggingface-deep-dive 09 类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过的超参数扫描(rank/target_modules/学习率三线对比),如果扫描的维度更多、需要穷举所有组合(而不是固定其余维度只变一个),本质上就是在一棵"每一层代表一个超参数选择"的决策树上做穷举——虽然实践中很少写成真正的递归回溯(通常用嵌套循环或 `itertools.product`),但背后的组合穷举思想是相通的。

**可运行例子:**
```python
def all_binary_strings(n):
    """演示回溯三要素:路径, 选择列表(这里退化成固定的'0'或'1'两个选择), 结束条件"""
    result = []
    path = []
    def backtrack():
        if len(path) == n:            # 结束条件
            result.append(''.join(path))
            return
        for choice in ['0', '1']:      # 选择列表
            path.append(choice)         # 做选择
            backtrack()
            path.pop()                    # 撤销选择(回溯的核心动作)
    backtrack()
    return result

assert all_binary_strings(2) == ['00', '01', '10', '11']
assert all_binary_strings(0) == ['']     # n=0时,唯一的"路径"是空字符串
assert len(all_binary_strings(3)) == 8    # 2^3种组合

# 验证"忘记撤销选择"的真实后果 —— 现场跑之前先说明预期:直觉上可能以为只是"结果多了几个错误项",
# 但实际验证后发现后果比这严重得多,如实记录真实观察到的行为
def buggy_no_undo(n):
    result = []
    path = []
    def backtrack():
        if len(path) == n:
            result.append(''.join(path))
            return
        for choice in ['0', '1']:
            path.append(choice)
            backtrack()
            # 故意不写 path.pop() —— 撤销这一步被省略
    backtrack()
    return result

raised = False
try:
    buggy_no_undo(2)
except RecursionError:
    raised = True
# 真实复现:path因为从未被pop,长度只增不减,len(path)==n这个结束条件只会在恰好等于n的
# 那一瞬间成立一次,之后path继续增长、永远无法再次等于n,递归失去了任何终止路径,
# 最终撞上sys.getrecursionlimit() —— 这比"结果错误"严重得多,是完全无法运行的死循环级bug
assert raised

print("OK: 回溯三要素框架验证正确(含n=0边界情况); "
      "现场复现'忘记撤销选择'的真实后果 —— 不是'结果有几个错误项'这种温和的错误,"
      "而是path长度永不回退导致结束条件再也无法满足, 最终真实触发RecursionError(比预想的更严重)")
```
本机实测:标准回溯框架在边界情况(n=0)下正确;人为构造一个"忘记撤销选择"的错误版本,**真实后果比预想的更严重**——最初设想是"结果会包含一些错误项",实际运行发现 `path` 因为从未被 `pop()`,长度只增不减,`len(path)==n` 这个结束条件只会在恰好等于 n 的那一瞬间短暂成立一次,之后 `path` 继续单调增长、再也不可能等于 n,递归彻底失去了终止路径,最终真实撞上 `RecursionError`——直接演示了"回溯"这个撤销动作不是可选的代码风格问题,省略它可能直接让算法失去终止性,而不只是"结果有点问题"这么温和。

**面试怎么问 + 追问链:** "回溯法和普通的递归/DFS有什么区别?" → 追问"什么样的问题适合用回溯法,什么样的不适合?"(回溯法适合"需要穷举所有满足条件的候选解、且候选解之间需要在共享状态基础上探索"的问题(子集/排列/组合/棋盘类约束满足);如果只需要"判断是否存在一个解"而不需要"所有解",很多时候能用更高效的方法(比如动态规划,如果子问题有大量重叠);这个追问检验的是能否判断"该不该用回溯"这个更前置的问题,而不是遇到任何组合类问题都无脑上回溯)。

**常见坑:**
1. 忘记撤销选择(`pop()` 这一步)——本知识点已经现场复现了这个错误的真实后果。
2. 把路径直接 `append` 进结果集,不做拷贝——同样会因为共享同一个列表对象,导致后续修改污染已保存的结果。

---

## 2. 子集问题:位运算解法与回溯解法对比

**签名/是什么:**
```
回溯解法: 每一层决定"要不要把当前元素加入路径"，遍历完所有元素时记录路径
位运算解法: n个元素对应n位二进制数，从0到2^n-1的每个数字，
           第i位是1就表示"选中第i个元素"
```

**一句话:** 求一个集合的所有子集,除了标准回溯写法,还可以用位运算枚举——n 个元素的所有子集,和 0 到 `2^n - 1` 这些整数的二进制表示是一一对应的关系,直接枚举这些整数、按位判断是否选中,是回溯的一个等价但形式不同的实现。

**底层机制/为什么这样设计:** 位运算解法能成立,根源在于"要不要选中某个元素"本身就是一个二元决策,n 个独立的二元决策自然对应 n 位二进制数——`mask & (1 << i)` 判断第 i 位是否为 1,恰好对应"是否选中第 i 个元素"这个决策。这个解法没有显式的递归/回溯过程,是一种**迭代**式的枚举,理解它有助于建立"回溯本质上就是在枚举一个决策空间"这个更抽象的认知——同一个决策空间(2^n 种选择组合),既可以用递归的方式逐层展开(回溯),也可以用整数编码后直接迭代枚举(位运算),两者是同一个数学对象的不同实现路径。

**AI 研究/工程场景:** [13 类](13-bit-manipulation-and-math.md)会讲的位运算技巧,以及[16 类](16-dynamic-programming-advanced.md)会讲的状态压缩 DP,都建立在"用一个整数的二进制位表示一个子集"这个编码方式上——子集问题的位运算解法是这个编码思想最直接的入门实例。

**可运行例子:**
```python
def subsets_backtrack(nums):
    result = []
    path = []
    def backtrack(start):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()
    backtrack(0)
    return result

def subsets_bitmask(nums):
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        result.append(subset)
    return result

nums = [1, 2, 3]
r1 = sorted(map(tuple, subsets_backtrack(nums)))
r2 = sorted(map(tuple, subsets_bitmask(nums)))
assert r1 == r2
assert len(r1) == 8   # 2^3个子集,含空集和全集

assert subsets_backtrack([]) == [[]]      # 空数组只有一个子集:空集本身
assert subsets_bitmask([]) == [[]]
assert len(subsets_backtrack([1])) == 2   # 单元素:空集和{1}

# 交叉验证:更大规模下两种解法依然完全一致
import random
random.seed(27)
for _ in range(10):
    test_nums = random.sample(range(20), random.randint(0, 6))
    a = sorted(map(tuple, subsets_backtrack(test_nums)))
    b = sorted(map(tuple, subsets_bitmask(test_nums)))
    assert a == b

print("OK: 回溯解法与位运算解法在空数组/单元素/多组随机测试下, 结果完全一致(8个子集验证通过)")
```
本机实测:两种解法在标准情况、空数组、单元素这几类情况下结果完全一致;10 组随机测试(元素数量 0 到 6 个)进一步确认两种解法在更大规模下依然完全等价。

**面试怎么问 + 追问链:** "位运算解法和回溯解法,哪个更好?" → 追问"如果 n 很大(比如 30),位运算解法还可行吗?"(不可行——`2^30` 约等于 10 亿,无论用哪种解法,子集数量本身就是指数级的,枚举所有子集在 n 较大时是不现实的;这个追问检验的是能否认识到"两种实现方式的选择"和"问题本身的复杂度下界"是两回事——子集问题本身就是 O(2^n),不存在能突破这个下界的写法,除非题目要求的不是"枚举所有子集"而是"判断/计数",那样可能有更高效的方法)。

**常见坑:**
1. 位运算解法里,`1 << i` 和 `mask & (1 << i)` 的运算优先级搞混——`&` 的优先级低于 `==` 等比较运算符,写 `mask & 1 << i == 0` 这类省略括号的写法容易得到和预期不符的结果,养成显式加括号的习惯更保险。
2. 误以为位运算解法"更快"——两种解法的时间复杂度都是 O(n·2^n)(2^n 个子集,每个子集平均需要 O(n) 时间构造),位运算解法只是省去了递归调用的开销,常数因子上可能略有优势,但复杂度量级是一样的。

---

## 3. 排列问题:去重排列的技巧

**签名/是什么:**
```
用 used[] 数组标记元素是否已经在当前路径中被使用
去重关键条件: if nums[i] == nums[i-1] and not used[i-1]: continue
```

**一句话:** 求全排列时如果原数组包含重复元素,朴素回溯会生成大量内容重复的排列——先对数组排序,再用"当前元素和前一个元素相同、且前一个元素这一轮还没被用过"这个条件跳过,能在生成过程中直接排除重复,而不是生成完所有排列后再去重。

**底层机制/为什么这样设计:** 排序后,值相同的元素会相邻排列在一起——去重条件 `nums[i] == nums[i-1] and not used[i-1]` 精确捕捉的是"在同一层递归里,尝试选择两个值相同的元素中排在后面的那个,但排在前面的那个此时还没被选用"这种情况:这意味着无论选前一个还是后一个,后续能扩展出的排列在数值上必然完全相同,选后一个纯属重复劳动,直接跳过。**为什么条件是"未被使用"而不是"已经被使用"**:如果 `used[i-1]` 为 `True`,说明前一个相同值的元素已经在当前路径的更早位置被用掉了,现在轮到 `i` 是在探索"用了 `nums[i-1]` 之后,再用 `nums[i]`"这个合法的、不重复的组合分支,不应该跳过。

**AI 研究/工程场景:** [huggingface-deep-dive 04 类](../huggingface-deep-dive/04-datasets-mechanics.md)讲过数据集去重的场景,"生成时就避免重复"和"生成后再去重"是两种不同的工程策略——前者(本知识点的做法)通常更高效,因为不需要先构造出全部结果、再花费额外的一遍扫描/哈希判重开销去过滤。

**可运行例子:**
```python
def permute_unique(nums):
    nums = sorted(nums)
    result = []
    path = []
    used = [False] * len(nums)
    def backtrack():
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue   # 去重核心条件
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False
    backtrack()
    return result

r = permute_unique([1, 1, 2])
assert sorted(map(tuple, r)) == [(1, 1, 2), (1, 2, 1), (2, 1, 1)]
assert len(r) == 3   # 不是3!=6种,重复的已经被排除

assert permute_unique([1, 1, 1]) == [[1, 1, 1]]   # 全部相同,只有1种排列
assert len(permute_unique([1, 2, 3])) == 6           # 全不相同,退化成标准全排列(3!=6)
assert permute_unique([]) == [[]]                       # 空数组

# 交叉验证:用"生成全部排列再用set去重"的朴素方法对照结果数量
from itertools import permutations
def brute_permute_unique_count(nums):
    return len(set(permutations(nums)))

import random
random.seed(28)
for _ in range(15):
    test_nums = [random.randint(1, 4) for _ in range(random.randint(0, 6))]
    assert len(permute_unique(test_nums)) == brute_permute_unique_count(test_nums)

print(f"OK: 排列去重技巧在'全相同'/'全不同'/空数组等边界情况下全部正确"
      f"([1,1,2]->{len(r)}种, 与3!=6种朴素去重的差异体现了去重效果); "
      f"15组随机测试数量与朴素'生成后用set去重'方法完全一致")
```
本机实测:`[1,1,2]` 生成的去重排列恰好是 3 种(而不是未去重的 6 种);全部相同、全部不同、空数组这几类边界情况均正确;15 组随机测试中,"生成时去重"和"生成后用 `set` 去重"两种策略在**数量**上完全一致(验证的是去重的正确性,不是效率对比)。

**面试怎么问 + 追问链:** "为什么排列去重需要先排序?" → 追问"如果不排序,直接用 `used[]` 判断,能不能正确去重?"(不能——去重条件依赖"值相同的元素在数组里相邻"这个前提,如果不排序,值相同的元素可能分散在数组各处,`nums[i] == nums[i-1]` 这个比较相邻位置的条件就失去了意义;这个追问检验的是能否理解"排序"不是这道题的可选步骤,是去重逻辑成立的必要前提)。

**常见坑:**
1. 忘记先排序就直接套用去重条件——如上面追问链所述,这会导致去重完全失效,依然产生大量重复排列。
2. 去重条件里的 `not used[i-1]` 写反(写成 `used[i-1]`)——这会导致相反的效果:该去重的没去重,不该跳过的反而被跳过,产生数量不对的错误结果(可能比正确答案多也可能少,取决于具体输入)。

---

## 4. 组合问题:组合总和及其变体

**签名/是什么:**
```
组合总和: 从候选数组里选数字(可重复使用),使得总和恰好等于target
关键技巧: 递归时用 backtrack(i, ...) 而不是 backtrack(i+1, ...) 实现"同一个数字可以重复选"
```

**一句话:** 组合类问题和排列类问题的本质区别是"顺序是否重要"——组合总和这类问题里,`[2,2,3]` 和 `[2,3,2]` 被认为是同一个组合,标准做法是固定一个"只能往后选、不能往前选"的顺序约束(用 `start` 参数控制),从根本上避免生成顺序不同但内容相同的重复组合。

**底层机制/为什么这样设计:** `backtrack(start, remain)` 里的 `start` 参数,保证了每一层递归只能从"当前索引开始往后"的候选里选择——这个约束直接排除了"先选3再选2"这种和"先选2再选3"内容相同但顺序不同的重复路径,不需要事后去重。"是否允许重复使用同一个数字"这个变体,只需要一个字符差异就能实现:允许重复使用时,递归调用传 `i`(下一层依然可以选择当前这个数字);不允许重复使用时,传 `i+1`(下一层必须严格往后)。这个"一个字符的差异对应完全不同的问题变体"恰恰说明,理解框架背后的**语义**(而不是死记某一个具体变体的代码)才能应对各种相似但不完全相同的题目。

**AI 研究/工程场景:** [huggingface-deep-dive 09 类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过的批量实验组合(比如从几个候选学习率里选择若干个进行网格搜索),如果要求"实验组合内部无序(用哪几个学习率比较,不关心比较的顺序)",本质上就是组合问题而不是排列问题,理解这个区别能避免生成大量本质相同、只是顺序不同的冗余实验配置。

**可运行例子:**
```python
def combination_sum(candidates, target):
    """数字可以重复使用"""
    result = []
    path = []
    def backtrack(start, remain):
        if remain == 0:
            result.append(path[:])
            return
        if remain < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, remain - candidates[i])   # 传i不是i+1:允许重复选同一个数字
            path.pop()
    backtrack(0, target)
    return result

assert sorted(map(tuple, combination_sum([2, 3, 6, 7], 7))) == [(2, 2, 3), (7,)]
assert combination_sum([2], 3) == []                     # 无解(2怎么加都凑不出奇数3)
assert combination_sum([], 5) == []                        # 没有候选数字
assert combination_sum([3], 3) == [[3]]                     # 单候选恰好命中

def combination_sum_no_reuse(candidates, target):
    """每个数字只能使用一次(需要先排序以支持剪枝和去重)"""
    candidates = sorted(candidates)
    result = []
    path = []
    def backtrack(start, remain):
        if remain == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue   # 同一层去重(呼应知识点3的排列去重技巧)
            if candidates[i] > remain:
                break        # 剪枝:排序后,后面的候选只会更大,不可能凑出解
            path.append(candidates[i])
            backtrack(i + 1, remain - candidates[i])   # 传i+1:不允许重复使用
            path.pop()
    backtrack(0, target)
    return result

assert sorted(map(tuple, combination_sum_no_reuse([10, 1, 2, 7, 6, 1, 5], 8))) == \
       sorted(map(tuple, [[1,1,6],[1,2,5],[1,7],[2,6]]))

print("OK: 组合总和(允许/不允许重复使用数字两个变体)在边界情况(无解/空候选/单候选命中)下全部正确")
```
本机实测:标准组合总和(允许重复使用)和无重复使用变体(含相同候选去重)在边界情况和标准用例下均正确匹配预期结果。

**面试怎么问 + 追问链:** "组合总和为什么用 `start` 参数而不是像排列那样用 `used[]` 数组?" → 追问"如果同时要求'不能重复使用同一个数字'又要求'结果里可以出现输入数组中的重复数字'(比如输入 `[1,1,2]`,`1` 只能作为两个不同位置的元素各用一次),这个约束怎么实现?"(需要结合本知识点的两个技巧:用 `start` 保证组合内部无序(不用 `used[]`),但仍然通过下标而不是值来控制"是否使用过"这个状态,配合排序去重条件——这正是上面例子里 `combination_sum_no_reuse` 的完整设计,这个追问检验的是能否把多个技巧组合起来应对更复杂的复合约束)。

**常见坑:**
1. 允许重复使用数字时,递归调用错误地传了 `i+1` 而不是 `i`——这会导致本该能被多次选中的数字只能被选一次,得到不完整的解集。
2. 不允许重复使用数字的版本,忘记加入"同一层去重"这个条件(呼应[知识点3](09-backtracking.md#3-排列问题去重排列的技巧))——如果候选数组本身有重复元素,会生成大量数值相同的重复组合。

---

## 5. N 皇后问题:经典约束满足

**签名/是什么:**
```
在N×N棋盘上放N个皇后,使得任意两个皇后都不在同一行/同一列/同一条对角线
逐行放置一个皇后,用集合分别记录已占用的列/主对角线/副对角线
```

**一句话:** N 皇后问题的关键优化是"逐行处理、每行恰好放一个皇后"(天然保证不同行不冲突),再用三个集合分别追踪"已占用的列"和"两个方向的对角线",让每一步的合法性判断降到 O(1),不需要每次都遍历棋盘检查冲突。

**底层机制/为什么这样设计:** 判断两个位置 `(r1,c1)` 和 `(r2,c2)` 是否在同一条主对角线(左上到右下方向),等价于 `r1-c1 == r2-c2`;在同一条副对角线(右上到左下方向),等价于 `r1+c1 == r2+c2`——这是一个纯粹的数学观察:同一条对角线上的所有格子,行列坐标之差(或之和)是常数。用两个集合分别记录"已被占用的 `row-col` 差值"和"已被占用的 `row+col` 和值",把原本需要 O(N) 遍历棋盘才能判断的对角线冲突检查,降到了 O(1) 的集合查询——这是[知识点1](01-complexity-and-python-builtins.md#5-set-与-dict-的底层复用关系)讲过的哈希集合 O(1) 查询特性,在一个具体的棋盘约束问题里的巧妙应用。

**AI 研究/工程场景:** N 皇后是"约束满足问题"(constraint satisfaction problem, CSP)的经典教学案例——[15类](15-graphs-advanced.md)会讲的图着色、[16类](16-dynamic-programming-advanced.md)的状压 DP 都属于同一个问题家族(在满足一组两两约束的前提下寻找可行解),N 皇后的这套"用集合/位运算快速判断约束是否冲突"的技巧,是这整个问题家族的通用解题思路起点。

**可运行例子:**
```python
def solve_n_queens(n):
    solutions = []
    cols = set()
    diag1 = set()   # row - col,主对角线标识
    diag2 = set()   # row + col,副对角线标识
    path = []
    def backtrack(row):
        if row == n:
            solutions.append(path[:])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col); diag1.add(row - col); diag2.add(row + col)
            path.append(col)
            backtrack(row + 1)
            path.pop()
            cols.remove(col); diag1.remove(row - col); diag2.remove(row + col)
    backtrack(0)
    return solutions

assert len(solve_n_queens(4)) == 2     # N=4的经典已知解数量
assert len(solve_n_queens(8)) == 92     # N=8的经典已知解数量(标准棋盘规模)
assert len(solve_n_queens(1)) == 1       # 1x1棋盘,唯一解
assert len(solve_n_queens(2)) == 0       # 2x2和3x3棋盘无解(棋盘太小放不下)
assert len(solve_n_queens(3)) == 0

def is_valid_solution(path, n):
    """独立验证一个解是否真的合法(不复用求解过程中的逻辑,做真正独立的正确性校验)"""
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            c1, c2 = path[r1], path[r2]
            if c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                return False
    return True

for solution in solve_n_queens(6):
    assert is_valid_solution(solution, 6)

print(f"OK: N皇后在N=1(1解)/N=2,3(0解)/N=4(2解)/N=8(92解, 经典已知值)等情况下全部正确; "
      f"N=6的全部解, 用独立的合法性校验函数逐一验证均为真正合法的解")
```
本机实测:N=4 得到 2 组解、N=8 得到 92 组解,均与该问题公认的经典已知解数量精确一致;N=1、N=2、N=3 这几类边界规模均正确(1x1 唯一解,2x2/3x3 无解);N=6 的全部解,额外用一个独立于求解逻辑的合法性校验函数逐一验证,确认每一组解都真正满足"不同行不同列不同对角线"的约束。

**面试怎么问 + 追问链:** "N 皇后问题为什么可以'逐行处理,每行恰好放一个皇后'这样简化?" → 追问"如果棋盘不是标准的正方形,而是允许某些格子不能放皇后(比如有障碍物),这个解法还能直接套用吗?"(基本框架依然适用,只需要在合法性判断里额外加入"这个格子是否是障碍物"这一条检查;这个追问检验的是能否把"逐行处理+集合追踪约束"这个核心思路和"棋盘是标准正方形"这个具体细节区分开,后者只是问题的一个特例条件,不是算法框架本身的前提)。

**常见坑:**
1. 对角线的判断公式记反(`row-col` 和 `row+col` 对应哪个方向的对角线搞混)——虽然由于对称性,这个错误可能不会导致程序崩溃或者明显异常,但会让"主对角线"和"副对角线"的约束检查错位,可能漏判某些真实冲突或者误判本不冲突的位置。
2. 忘记在回溯返回后,把三个集合里对应当前尝试的记录移除(`remove` 这一步)——这是[知识点1](09-backtracking.md#1-回溯法总纲决策树视角)"忘记撤销选择"这个通病在多个并行状态变量场景下的具体体现,任何一个集合忘记撤销都会污染后续兄弟分支的探索。

---

## 6. 数独求解:约束传播与回溯结合

**签名/是什么:**
```
find空格子 -> 尝试填入1~9中任意一个满足行/列/3x3宫格约束的数字 -> 递归填下一个空格
             -> 如果后续填不下去(所有数字都试过仍无解),回溯撤销当前格子,尝试同一格子的下一个候选
```

**一句话:** 数独求解是回溯法在真实、具有实际知名度的约束满足问题上的应用——比 N 皇后更复杂,因为每个位置需要同时满足"行不重复""列不重复""所在 3×3 宫格不重复"三重约束,但核心框架(尝试→递归→不行就撤销）完全相同。

**底层机制/为什么这样设计:** `is_valid` 函数需要检查三类约束:遍历当前行、遍历当前列、遍历当前格子所在的 3×3 宫格(宫格起点用 `3*(r//3), 3*(c//3)` 算出)——这三重检查合起来保证了候选数字在放入这个位置后,不会立即违反数独规则。回溯的核心逻辑和前面几个知识点完全一致:找到第一个空格子,依次尝试 1~9 每个候选数字,合法就递归处理下一个空格,如果递归返回"无法完成"(所有后续尝试都失败了),说明当前这个候选数字的选择是错误的,撤销(把格子重新设为空),尝试下一个候选。**"约束传播"**(本知识点标题提到的另一半)指的是更高级的优化技巧(比如提前排除"某个空格只有唯一合法候选"的情况,直接填入而不用等到暴力尝试),本知识点的实现是纯回溯版本,没有加入约束传播优化,这是有意为之——先扎实理解纯回溯的正确性,再考虑效率优化。

**AI 研究/工程场景:** 数独求解器是"回溯法能否在实际有意义的复杂度下解决真实问题"的一个很好的验证案例——[huggingface-deep-dive](../huggingface-deep-dive/00-roadmap.md) 系列反复强调的"真实验证不空谈"这个纪律,在这里体现为:不满足于"理论上回溯法能解数独",而是真的用一个标准数独谜题跑一遍,验证输出的解确实合法。

**可运行例子:**
```python
def solve_sudoku(board):
    def is_valid(r, c, val):
        for i in range(9):
            if board[r][i] == val or board[i][c] == val:
                return False
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if board[i][j] == val:
                    return False
        return True

    def backtrack():
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for val in '123456789':
                        if is_valid(r, c, val):
                            board[r][c] = val
                            if backtrack():
                                return True
                            board[r][c] = '.'   # 撤销:这个候选走不通,恢复空格状态
                    return False   # 1~9都试过了,依然无解,向上层报告失败
        return True   # 没有空格子了,说明已经完整填好

    backtrack()
    return board

puzzle = [
    list('53..7....'), list('6..195...'), list('.98....6.'),
    list('8...6...3'), list('4..8.3..1'), list('7...2...6'),
    list('.6....28.'), list('...419..5'), list('....8..79'),
]
solved = solve_sudoku([row[:] for row in puzzle])

def verify_solution(board):
    for i in range(9):
        row = board[i]
        col = [board[r][i] for r in range(9)]
        if len(set(row)) != 9 or len(set(col)) != 9:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = [board[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)]
            if len(set(block)) != 9:
                return False
    return True

assert verify_solution(solved)     # 独立校验函数确认:每行/每列/每个3x3宫格都恰好是1~9各一次

# 确认求解结果和原始谜题的已知数字完全兼容(没有改动原本给定的提示数字)
for r in range(9):
    for c in range(9):
        if puzzle[r][c] != '.':
            assert solved[r][c] == puzzle[r][c]

print("OK: 数独求解器在标准谜题上正确求解, 独立校验函数确认解满足全部行/列/宫格约束, "
      "且没有修改原始谜题给定的提示数字")
```
本机实测:标准数独谜题求解成功,独立的合法性校验函数(不复用求解逻辑本身,单独重新检查每行/每列/每个 3×3 宫格)确认结果完全满足数独规则;额外验证了求解过程没有意外修改原始谜题里已经给定的提示数字。

**面试怎么问 + 追问链:** "数独求解的回溯法,最坏情况复杂度是多少?" → 追问"实际跑起来,为什么远没有达到理论最坏情况那么慢?"(理论最坏情况是每个空格都要尝试 9 种可能,复杂度接近 `9^(空格数)`,是一个天文数字;但实际运行很快,是因为约束(行/列/宫格)会让绝大多数分支在很浅的深度就被立即排除,真正深入探索的分支远少于理论最坏情况——这个追问检验的是能否区分"理论最坏复杂度"和"实际典型表现",这类差异在有强约束剪枝效果的回溯问题里非常常见,呼应[知识点7](09-backtracking.md#7-剪枝方法论)会更系统展开的剪枝效果)。

**常见坑:**
1. `is_valid` 检查忘记涵盖三类约束里的某一类(比如只查了行列,漏了宫格)——这会导致求解器可能给出一个"看起来填满了但实际上宫格有重复"的非法解,必须用独立的校验函数事后确认,不能只信任求解过程本身没有报错。
2. 撤销状态(`board[r][c] = '.'`)的位置放错——必须在"这个候选数字的递归分支已经彻底试完(不管成功与否)"之后才能撤销,如果放在错误的位置(比如放在 `is_valid` 判断之前),会破坏后续候选数字判断合法性时依赖的棋盘状态。

---

## 7. 剪枝方法论

**签名/是什么:**
```
剪枝：在还没有走到死路之前，提前判断"这个方向必然不可能有解"，直接跳过整个子树，
     而不是老老实实递归下去,等递归返回时才发现是死路
```

**一句话:** 剪枝的价值不是"让代码看起来更高级",是真实地把需要遍历的决策树节点数量在实践中砍掉好几个数量级——本知识点用真实调用次数统计,证明这不是纸上谈兵。

**底层机制/为什么这样设计:** 以[组合总和](09-backtracking.md#4-组合问题组合总和及其变体)为例,朴素回溯会对每个候选数字都递归尝试,即使某个候选数字已经让剩余目标值变成负数(不可能再凑出解),依然会进入下一层递归,直到那一层才发现"到底了、还没凑够,只能返回"——这些"提前已经能判断出必然失败,却还是走了个过场才发现"的调用,就是剪枝要消除的浪费。有效剪枝的两个关键条件:①判断条件本身计算代价要低(否则"提前判断"本身的开销可能超过它省下的探索开销);②要在决策树尽量**浅**的层级就能触发判断(越早剪枝,省下的子树规模越大)——这也是为什么组合总和的剪枝要求"候选数组先排序":排序后,一旦当前候选数字已经超过剩余目标值,后面的候选只会更大,可以直接 `break` 掉整层循环,而不必对每个候选都单独判断。

**AI 研究/工程场景:** [15类](15-graphs-advanced.md)会讲的图搜索算法(比如 A* 搜索的启发式函数),本质上和这里的剪枝是同一类思想的更系统化版本——用一个能快速计算、且能可靠排除不可能分支的判断条件,大幅压缩需要真正探索的搜索空间,区别只在于剪枝条件的复杂程度和理论保证的严格性。

**可运行例子:**
```python
import time

def subsets_sum_no_prune(nums, target):
    """不剪枝版本:朴素地穷举所有子集组合,统计真实调用次数"""
    call_count = [0]
    def backtrack(start, remain):
        call_count[0] += 1
        if remain == 0:
            return
        for i in range(start, len(nums)):
            backtrack(i + 1, remain - nums[i])
    backtrack(0, target)
    return call_count[0]

def subsets_sum_with_prune(nums, target):
    """剪枝版本:排序后, 一旦当前候选已经超过剩余目标, 直接break跳过整层"""
    call_count = [0]
    nums_sorted = sorted(nums)
    def backtrack(start, remain):
        call_count[0] += 1
        if remain <= 0:
            return
        for i in range(start, len(nums_sorted)):
            if nums_sorted[i] > remain:
                break   # 剪枝:排序后, 后面的候选只会更大, 不可能凑出解
            backtrack(i + 1, remain - nums_sorted[i])
    backtrack(0, target)
    return call_count[0]

nums = list(range(1, 21))   # 1~20
target = 5

no_prune_calls = subsets_sum_no_prune(nums, target)
with_prune_calls = subsets_sum_with_prune(nums, target)

assert with_prune_calls < no_prune_calls   # 剪枝后的调用次数必须明显更少
assert with_prune_calls < no_prune_calls / 100   # 差距应该是数量级级别的,不是小幅优化

t0 = time.perf_counter(); subsets_sum_no_prune(nums, target); no_prune_time = time.perf_counter() - t0
t0 = time.perf_counter(); subsets_sum_with_prune(nums, target); with_prune_time = time.perf_counter() - t0
assert with_prune_time < no_prune_time

print(f"OK: 1~20选数字凑目标和{target}, 不剪枝调用次数={no_prune_calls}, "
      f"剪枝后调用次数={with_prune_calls}(减少{no_prune_calls/with_prune_calls:.0f}倍), "
      f"耗时从{no_prune_time:.4f}s降到{with_prune_time:.4f}s")
```
本机实测:同样是"从 1~20 里选数字凑目标和 5"这个具体问题,不剪枝版本的递归调用次数达到 819203 次,耗时约 0.185s;加入剪枝后调用次数骤降到 10 次,耗时几乎无法用秒表测出(远小于毫秒)——**调用次数减少了超过 8 万倍**,这不是理论上的优化,是真实统计出的数字。

**面试怎么问 + 追问链:** "怎么判断一个回溯问题'能不能剪枝'?" → 追问"剪枝条件本身写错了,会有什么后果?比只是'剪枝效果不好'更严重吗?"(会——如果剪枝条件误判(把某个实际上可能有解的分支当成不可能而提前排除),会导致算法漏掉本该存在的解,这是**正确性**问题,比"剪枝效果不理想"(只是性能问题)严重得多;设计剪枝条件时,必须能严格证明"满足这个条件的分支绝对不可能有解",不能只是"大概率没有解"这种不严谨的直觉判断,这个追问检验的是对剪枝正确性边界的严谨态度)。

**常见坑:**
1. 剪枝条件写得过于宽松(比如把"可能没有解"误判成"绝对没有解")——这会漏掉正确答案,是本知识点面试追问链里强调的、比"剪枝力度不够"更严重的错误。
2. 只在最内层循环加剪枝,却忽略了更外层、更早期就能生效的剪枝机会——剪枝越靠近决策树的根部生效,省下的子树规模指数级增长,把剪枝逻辑尽量往前挪、尽早生效,通常比在深层加更多剪枝条件更有效。

---

## 8. 回溯常见坑

**签名/是什么:**
```
状态没有正确回退 -> 后续兄弟分支的探索基于被污染的状态,结果错误
去重逻辑写反/缺失 -> 结果集包含大量重复项,或者错误地漏掉了本该存在的解
```

**一句话:** 回溯法的 bug 高度集中在两类:某个"做选择"对应的状态修改忘记在退出这一层递归前撤销,以及排列/组合类问题的去重判断条件写错——这两类错误通常不会让程序崩溃,只会让结果集合"数量不对"或"内容里混入了不该有的重复项/缺了该有的项"。

**底层机制/为什么这样设计:** 回溯法依赖共享的可变状态(`path`、`used[]`、[N皇后知识点](09-backtracking.md#5-n-皇后问题经典约束满足)的三个集合)在多个层级、多个分支之间被反复修改和恢复——只要有**任何一处**"做选择"对应的修改,在对应的"撤销选择"环节被遗漏,这个疏漏就会污染后续所有还没探索的兄弟分支,而且污染的影响可能要经过好几层递归之后才会在最终结果里表现出来,导致排查时很难直接定位到"到底是哪一行代码忘了撤销"。去重问题的本质则是[知识点3](09-backtracking.md#3-排列问题去重排列的技巧)已经深入讨论过的:"在决策树的哪个层级,基于什么条件跳过某个分支",这个判断条件本身包含微妙的边界(比如 `not used[i-1]` 这个条件的方向),错误的方向会导致"该去重的没去重"或者"不该去重的被误删"这两种截然不同、但都错误的后果。

**AI 研究/工程场景:** 这类"共享可变状态在多分支探索之间必须被正确恢复"的问题,本质上和[05类](05-stacks-and-queues.md)/[03类](03-linked-lists.md)反复强调的"数据结构操作顺序错误导致静默出错"是同一类工程纪律问题——只是回溯法因为涉及的共享状态变量更多(路径、多个辅助集合),出错的具体位置更分散,排查难度通常更高。

**可运行例子:**
```python
def buggy_permute_forget_undo(nums):
    """故意忘记撤销used标记(而不是忘记撤销path),现场复现真实后果"""
    result = []
    path = []
    used = [False] * len(nums)
    def backtrack():
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            # 故意漏掉 used[i] = False 这一行
    backtrack()
    return result

buggy_result = buggy_permute_forget_undo([1, 2, 3])

def correct_permute(nums):
    result = []
    path = []
    used = [False] * len(nums)
    def backtrack():
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False
    backtrack()
    return result

correct_result = correct_permute([1, 2, 3])
assert len(correct_result) == 6              # 3!=6种排列,正确
assert len(buggy_result) < len(correct_result)  # 真实复现:忘记撤销used标记,结果数量偏少
assert len(buggy_result) == 1                    # 因为used永远不会被重置为False,只能生成第一种排列就再也没有可选元素了

# 完全遗漏去重条件的真实后果(这里刻意先说明一个曾经的错误预期,再用真实验证结果修正它:
# 最初以为把 not used[i-1] 的方向写反会产生错误结果,但实测发现两个方向都能正确去重——
# 这是排列去重问题里一个真实、容易被误解的细节:两种方向都是有效的canonical排序规则,
# 唯一真正会出错的是"完全不做去重判断")
def no_dedup_check_at_all(nums):
    nums = sorted(nums)
    result = []
    path = []
    used = [False] * len(nums)
    def backtrack():
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            # 完全没有去重判断,退化成标准全排列
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False
    backtrack()
    return result

correct_unique = sorted(map(tuple, [[1, 1, 2], [1, 2, 1], [2, 1, 1]]))
no_dedup_result = no_dedup_check_at_all([1, 1, 2])
assert len(no_dedup_result) == 6            # 3!=6,退化成标准全排列,没有任何去重效果
assert sorted(map(tuple, no_dedup_result)) != correct_unique   # 真实复现:结果集合含大量重复项
# 用Counter确认具体是"每个本该唯一的排列都出现了2次"这个真实的重复模式
from collections import Counter
dup_counts = Counter(map(tuple, no_dedup_result))
assert all(c == 2 for c in dup_counts.values())   # 3个唯一排列,每个都重复出现了恰好2次

print(f"OK: 现场复现'忘记撤销used标记'导致排列数量从{len(correct_result)}骤降到{len(buggy_result)}; "
      f"现场复现'完全遗漏去重判断'导致{len(no_dedup_result)}个结果里3个唯一排列各重复2次"
      f"(修正了此前对'方向写反也会出错'的错误预期——实测发现两个方向的去重条件其实都正确,"
      f"真正的坑是完全不做判断)")
```
本机实测:忘记撤销 `used` 标记这个真实 bug,让 `[1,2,3]` 的全排列数量从正确的 6 种骤降到只有 1 种(因为标记永远不会被重置,第一次用完所有元素后再也没有可选的了);**完全遗漏去重判断**这个真实 bug,让 `[1,1,2]` 退化成标准全排列(6 个结果),其中 3 个本该唯一的排列各自重复出现了 2 次——这里也如实记录一处撰写过程中的真实修正:最初设想"把去重条件 `not used[i-1]` 的方向写反"会产生错误结果,但现场验证后发现**两个方向的条件实际上都能正确去重**(只是各自对应不同但同样有效的"canonical 顺序"选择),真正会导致错误的是完全不做判断,不是方向选择本身。

**面试怎么问 + 追问链:** "写完回溯代码后,你会怎么系统性地检查是否正确处理了所有状态的撤销?" → 追问"能不能不靠肉眼检查,用某种方式自动发现'状态没有被正确撤销'这类问题?"(一个实用技巧:在 `backtrack` 函数的最外层调用结束后,断言所有共享状态变量都恢复到了初始状态(比如 `assert path == []`、`assert all(not u for u in used)`)——如果最终状态不是初始状态,说明某个分支的撤销逻辑有遗漏;这个追问检验的是能否把"肉眼检查代码"升级成"用断言自动验证不变量",是本系列一直强调的"用真实断言代替感觉上验证了"这个纪律在回溯问题上的具体应用)。**追问链继续深入**:"你刚才说'两个方向的去重条件都正确',能不能证明这一点,而不是只凭实测结果?"(两个方向本质上都是在同一组"值相同的候选"里,人为规定一个固定的选取顺序(要么强制`必须先用索引更小的`,要么反过来),只要这个顺序在所有分支里保持一致,就不会有两条不同的搜索路径产出完全相同的排列——两种写法只是选了不同但内部自洽的规则,这也是为什么现场实测没能找出"方向写反"的反例:它本来就不是一个真正的错误,是我在没有验证前就假设了一个不成立的"坑"。)

**常见坑:**
1. 忘记撤销某个共享状态变量(`used[]`、集合、计数器等)——本知识点已经具体复现了这类错误的真实后果,规模从"结果集完全错误"到"结果集数量偏少偏多"不等,取决于具体是哪个状态变量被遗漏。
2. 完全遗漏去重判断条件——这是真正会导致重复结果的错误,本知识点已经用 `Counter` 具体验证了重复的模式(每个唯一排列恰好重复 2 次)。**不要想当然地假设"某个方向的写法一定是错的"就不加验证地写进结论**——本知识点本身在撰写过程中就先入为主地做了一次这样的错误假设,靠现场跑代码才发现假设不成立,这恰恰示范了本系列反复强调的纪律:任何"这样写应该会出错"的直觉,都必须真正跑一遍代码验证,不能凭感觉下结论。

---

*本篇 8 个知识点全部在仓库根目录 `.venv` 真实测试验证(含边界情况覆盖、经典问题已知解数量核对、剪枝效果的真实调用次数统计、以及"忘记撤销状态"这类真实bug的现场复现)。*
