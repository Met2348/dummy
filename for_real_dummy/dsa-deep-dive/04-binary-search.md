# 04 · 二分查找(Binary Search)

> 总览见 [00-roadmap.md](00-roadmap.md)。二分查找看起来是"背了模板就会"的题材,但终面追问链最喜欢在边界写法上刁难——`lo <= hi` 还是 `lo < hi`、`hi = mid` 还是 `hi = mid - 1`,选错一个都可能死循环或漏解。本类不背模板,讲清楚每种写法背后的循环不变量。

---

## 1. 标准二分查找模板与循环不变量

**签名/是什么:**
```
lo, hi = 0, len(nums) - 1
while lo <= hi:
    mid = lo + (hi - lo) // 2
    ...
```

**一句话:** 二分查找每一步用中点把搜索区间砍掉一半,靠的是数组**有序**这个前提——通过比较中点值和目标值,可以确定目标不可能在被排除的那一半里,这样 O(n) 的线性搜索被压缩成 O(log n)。

**底层机制/为什么这样设计:** `mid = lo + (hi - lo) // 2` 而不是直接写 `(lo + hi) // 2`,是为了避免 `lo` 和 `hi` 都很大时两者相加可能溢出(虽然 Python 的整数没有固定位宽溢出问题,但这是一个在 C/Java 等有固定整型位宽的语言里必须注意的真实陷阱,写成前一种形式的习惯在任何语言下都安全,是通用最佳实践)。循环不变量是"答案如果存在,一定落在 `[lo, hi]` 这个闭区间内"——每次根据中点比较结果收缩区间,这个不变量始终保持,直到 `lo > hi`(区间为空)证明答案不存在,或者中点直接命中。

**AI 研究/工程场景:** [huggingface-deep-dive 11 类](../huggingface-deep-dive/11-hub-and-sharing.md)讲过按 `revision`/时间戳做版本查找的场景,如果一批模型版本按时间有序排列,要定位"最接近某个时间点的版本",本质就是在做二分查找。

**可运行例子:**
```python
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

assert binary_search([1, 3, 5, 7, 9, 11], 7) == 3
assert binary_search([1, 3, 5, 7, 9, 11], 4) == -1   # 不存在
assert binary_search([], 1) == -1                      # 空数组
assert binary_search([5], 5) == 0                      # 单元素命中
assert binary_search([5], 3) == -1                      # 单元素不命中

# 与暴力线性查找交叉验证,随机测试
import random
random.seed(0)
def linear_search(nums, target):
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1

for _ in range(50):
    arr = sorted(random.sample(range(200), random.randint(0, 20)))
    target = random.choice(arr) if arr and random.random() < 0.7 else random.randint(0, 200)
    bs_result = binary_search(arr, target)
    ls_result = linear_search(arr, target)
    # 二分和线性查找找到的位置可能不同(重复元素时),但"是否存在"的判断必须一致
    assert (bs_result != -1) == (ls_result != -1)
    if bs_result != -1:
        assert arr[bs_result] == target  # 二分找到的位置,值必须确实等于target

print("OK: 标准二分查找在边界情况下全部正确, 50组随机测试与线性查找的'是否存在'判断完全一致")
```
本机实测:边界情况(空数组、单元素命中/不命中)全部正确;50 组随机测试中,二分查找和线性查找对"目标是否存在"的判断完全一致,且二分查找返回的位置上的值确实等于目标值。

**面试怎么问 + 追问链:** "写出标准二分查找,并说明为什么用 `lo <= hi` 而不是 `lo < hi` 作为循环条件。" → 追问"如果数组存在重复元素,标准二分查找返回的是哪一个位置?"(不确定——标准写法只保证"找到某一个值等于 target 的位置",具体是重复值里的第几个取决于比较顺序,不是稳定的;如果需要"第一个"或"最后一个",需要[知识点2](04-binary-search.md#2-二分边界变体)的变体写法,这个追问检验的是能否区分"能找到答案"和"找到指定的那一个答案"这两个不同要求)。

**常见坑:**
1. 循环条件用 `lo <= hi` 时,收缩区间必须用 `mid + 1` 和 `mid - 1`(排除掉 `mid` 本身);如果误用 `lo < hi` 却依然写 `hi = mid - 1`,会错误地排除掉本该继续检查的位置。
2. `mid` 计算写成 `(lo + hi) // 2` 在 Python 里虽然不会溢出,但换到其他语言习惯性写法容易埋雷——养成 `lo + (hi - lo) // 2` 的写法习惯,不依赖"这门语言恰好没有整型溢出"这个具体环境特性。

---

## 2. 二分查找边界变体:找第一个 / 最后一个满足条件的位置

**签名/是什么:**
```
find_first: hi = mid (不是 mid - 1)，收缩区间时不排除mid本身，持续向左找
find_last:  lo = mid + 1，持续向右找，最终lo-1是答案
```

**一句话:** 当数组存在重复元素,需要精确找到"第一个等于 target 的位置"或"最后一个等于 target 的位置"时,不能用标准二分查找的"找到就返回"写法,而要用一种"找到后不立即返回,继续向指定方向收缩"的变体写法。

**底层机制/为什么这样设计:** 找第一个满足条件的位置,可以把问题转化成"找到第一个满足 `nums[i] >= target` 的位置"——这是一个关于下标的**单调**谓词(一旦某个位置满足,它后面的位置也都满足),二分查找本质上就是在一个单调谓词上找"分界点"。写法上用左闭右开区间 `[lo, hi)`,当 `nums[mid] >= target` 时,`mid` 本身可能就是答案,所以不能排除它,只能收缩 `hi = mid`;只有 `nums[mid] < target` 时才能确定 `mid` 及其左边都不是答案,收缩 `lo = mid + 1`。这个"要不要把 mid 本身排除出下一轮搜索区间"的判断,是所有边界变体写法的核心区别所在。

**AI 研究/工程场景:** [huggingface-deep-dive 01 类](../huggingface-deep-dive/01-tokenizer-internals.md)讲过 tokenizer 的 offset mapping,如果要在一批按位置排序的 token 边界里,找到"第一个边界位置 >= 某个字符下标"的 token,就是这里讲的边界变体二分的直接应用。

**可运行例子:**
```python
import bisect

def find_first(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(nums) and nums[lo] == target:
        return lo
    return -1

def find_last(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    if lo - 1 >= 0 and nums[lo - 1] == target:
        return lo - 1
    return -1

arr = [5, 7, 7, 7, 8, 8, 10]
assert find_first(arr, 7) == 1 and find_last(arr, 7) == 3
assert find_first(arr, 9) == -1 and find_last(arr, 9) == -1   # 不存在
assert find_first([], 1) == -1 and find_last([], 1) == -1      # 空数组
assert find_first([5], 5) == 0 and find_last([5], 5) == 0      # 单元素

# 交叉验证:结果必须和Python标准库bisect模块给出的边界一致
assert find_first(arr, 7) == bisect.bisect_left(arr, 7)
assert find_last(arr, 7) == bisect.bisect_right(arr, 7) - 1

print("OK: 边界变体二分在重复元素/不存在/空数组/单元素情况下全部正确, "
      "且与标准库bisect模块给出的边界位置完全一致")
```
本机实测:`[5,7,7,7,8,8,10]` 中查找 7,`find_first` 返回下标 1,`find_last` 返回下标 3,和标准库 `bisect.bisect_left`/`bisect_right` 的结果精确一致;边界情况(不存在、空数组、单元素)也全部正确。

**面试怎么问 + 追问链:** "找第一个满足条件的位置和标准二分有什么本质区别?" → 追问"能不能用同一个统一的模板,通过传入不同的判断函数,同时支持找第一个和找最后一个?"(可以——把"找第一个满足 `nums[i] >= target`"抽象成"找第一个满足谓词 `p(i)` 为真的位置",这个模板本身和具体判断哪个值无关,只依赖谓词单调;这个抽象正是 Python 标准库 `bisect` 模块内部的实现思路,也是这类问题的更通用理解方式)。

**常见坑:**
1. 在 `nums[mid] >= target` 时错误地写成 `hi = mid - 1`——这会把 `mid` 本身（可能正是答案）排除出搜索区间,导致漏掉正确答案。
2. 循环结束后忘记检查 `lo`(或 `lo-1`)位置的值是否真的等于 `target`——二分查找收缩出的位置只是"谓词分界点",不代表这个位置上一定存在目标值(比如目标值根本不在数组里时,`lo` 会停在"如果存在应该插入的位置",但那个位置的值和 target 无关)。

---

## 3. 旋转排序数组的二分查找

**签名/是什么:**
```
数组本身有序，但被从某个位置"旋转"过，如 [4,5,6,7,0,1,2]
关键判断：mid 把数组分成两半，其中必有一半是"局部有序"的
```

**一句话:** 旋转数组虽然整体不是单调有序,但从任意一个中点切开,左右两半里**必然有一半依然保持严格有序**——先判断哪一半有序,再看目标值是否落在这个有序区间内,依然能把二分查找的思路套用上去。

**底层机制/为什么这样设计:** 如果 `nums[lo] <= nums[mid]`,说明左半部分 `[lo, mid]` 没有发生"旋转断裂",是有序的;这时候只需要判断 `target` 是否落在 `[nums[lo], nums[mid])` 这个区间内——如果是,答案只可能在左半部分,收缩到左边;如果不是,答案必然在右半部分(哪怕右半部分本身不是全局有序的,只要确认目标不在有序的那一半里,就能安全排除它)。如果 `nums[lo] > nums[mid]`,则反过来,右半部分 `[mid, hi]` 才是有序的。这个"先判断哪一半有序,再判断目标是否落在有序区间"的两层逻辑,是旋转数组二分查找和标准二分查找本质的区别。

**AI 研究/工程场景:** 这类"整体不是全局单调,但可以拆分成若干个局部单调区间"的结构,在处理**环形缓冲区**(circular buffer,比如某些流式数据管道用环形数组实现)的查找场景中会真实出现——旋转数组问题本质上就是在给环形结构的查找算法打基础。

**可运行例子:**
```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:          # 左半部分有序
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                # 右半部分有序
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1

assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1   # 不存在
assert search_rotated([1], 1) == 0                        # 单元素
assert search_rotated([], 1) == -1                        # 空数组
assert search_rotated([1, 2, 3, 4, 5], 3) == 2            # 完全没旋转(等价于标准数组)

# 交叉验证:对所有可能的旋转位置,结果必须和"先还原成有序数组再线性查找"一致
base = [10, 20, 30, 40, 50]
for rotate_at in range(len(base)):
    rotated = base[rotate_at:] + base[:rotate_at]
    for target in [10, 30, 50, 99]:
        expected_found = target in base
        result = search_rotated(rotated, target)
        assert (result != -1) == expected_found
        if result != -1:
            assert rotated[result] == target

print("OK: 旋转数组二分查找在单元素/空数组/未旋转等边界情况下全部正确, "
      "对全部5种旋转位置x4个查询目标的组合测试均与预期一致")
```
本机实测:标准情况和边界情况(单元素、空数组、完全未旋转)全部正确;对基准数组的全部 5 种旋转位置分别测试 4 个查询目标(共 20 组组合),结果与预期完全一致。

**面试怎么问 + 追问链:** "旋转数组二分查找,判断哪一半有序的依据是什么?" → 追问"如果数组里存在重复元素(比如 `[1,1,1,0,1]`),这个算法还能正确工作吗?"(不一定——当 `nums[lo] == nums[mid]` 时,无法判断到底是左半有序还是右半有序(因为重复元素让"比较大小"这个判断失效),标准做法是这种情况下退化成 `lo += 1` 逐步收缩,牺牲最坏情况下的 O(log n) 变成 O(n);这个追问检验的是能否发现"数组无重复"是这个算法能保持 O(log n) 的一个隐藏前提)。

**常见坑:**
1. 判断"目标是否落在有序区间"时,边界条件用错(比如该用 `<=` 的地方用了 `<`)——特别是数组只有两三个元素时,这类边界错误最容易暴露,写完后应该用小规模例子手工验算。
2. 误以为只要数组"看起来乱序"就不能用二分——旋转数组不是完全随机乱序,依然具有"两段有序拼接"这个结构性质,这个结构性质才是二分查找依然适用的关键,不能一概而论地认为"没排序就不能二分"。

---

## 4. 二分答案:在答案空间上二分

**签名/是什么:**
```
不在数组下标上二分，而是在"可能的答案取值范围"上二分，
要求：判断"某个候选答案是否可行"这件事本身具有单调性
```

**一句话:** 二分查找不是只能用在有序数组上——只要能构造出一个"候选答案是否满足要求"的单调判断函数(某个阈值以下都不满足,以上都满足,或反过来),就可以在**答案的取值范围**上做二分,这类问题统称"二分答案"。

**底层机制/为什么这样设计:** 以经典的"珂珂吃香蕉"问题为例(每小时选择一个恒定速度吃香蕉,要在 h 小时内吃完所有堆,求最小可行速度):速度越快,吃完所需的总时间必然越短(或不变)——这是一个关于"速度"这个候选答案的严格单调关系,速度取值范围是 `[1, max(piles)]`,可以在这个范围上二分,每次用 `check(speed)` 判断"这个速度能否在 h 小时内吃完",根据判断结果收缩范围。二分答案的技巧价值在于:它把一个"求最优解"的问题转化成了"反复判断某个候选解是否可行"的问题,只要"是否可行"这个判断本身好写、且具有单调性,就能用二分把候选空间的搜索复杂度从线性降到对数级。

**AI 研究/工程场景:** [huggingface-deep-dive 08 类](../huggingface-deep-dive/08-quantization-bitsandbytes.md)讲过的显存/精度权衡场景,如果要为一个训练任务找到"能放进显存的最大 batch size",本质上就是二分答案的真实应用——batch size 越大显存占用必然越高(单调),可以二分搜索,而不需要从 1 开始线性尝试。

**可运行例子:**
```python
def min_eating_speed(piles, h):
    def hours_needed(speed):
        return sum((p + speed - 1) // speed for p in piles)  # 向上取整

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if hours_needed(mid) <= h:
            hi = mid          # mid可行,但可能不是最小的,继续向左找
        else:
            lo = mid + 1       # mid不可行,答案必须更大
    return lo

assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
assert min_eating_speed([1], 1) == 1                    # 单堆,时间恰好够
assert min_eating_speed([5], 100) == 1                  # 时间远远充裕,最小速度1即可

# 交叉验证:用线性扫描找最小可行速度,结果必须和二分答案一致
def brute_min_speed(piles, h):
    def hours_needed(speed):
        return sum((p + speed - 1) // speed for p in piles)
    for speed in range(1, max(piles) + 1):
        if hours_needed(speed) <= h:
            return speed
    return max(piles)

import random
random.seed(1)
for _ in range(20):
    piles = [random.randint(1, 30) for _ in range(random.randint(1, 6))]
    h = random.randint(len(piles), 50)
    assert min_eating_speed(piles, h) == brute_min_speed(piles, h)

print("OK: 二分答案在边界情况下全部正确, 20组随机测试与线性扫描找最小可行解结果完全一致")
```
本机实测:边界情况(单堆恰好够时间、时间远远充裕)全部正确;20 组随机测试中,二分答案和线性扫描找最小可行速度的结果完全一致。

**面试怎么问 + 追问链:** "什么样的问题可以考虑用'二分答案'这个技巧?" → 追问"怎么判断一个'求最优解'的问题是否具有二分答案需要的单调性?"(先问自己:如果候选答案 x 是可行的,那么比 x 更"宽松"方向的所有候选(比如本例中更快的速度)是否也必然可行?如果这个单调性成立,就可以二分答案;如果不成立(比如某个中间值可行,但更大和更小的值都不可行),二分答案会给出错误结果——这个追问检验的是能否在拿到新题目时,主动检验单调性这个前提,而不是看到"求最小/最大值"就无脑套用二分模板)。

**常见坑:**
1. 没有验证单调性就直接套用二分答案模板——如果"是否可行"这个判断不满足单调性,二分会过早收缩到错误的区间,得到错误答案且不会有任何报错提示。
2. `check` 函数(本例中的 `hours_needed`)本身如果实现得低效(比如内部又用了一次不必要的排序),会拖累整体复杂度——二分答案的总复杂度是"二分的轮数(对数级)乘以每轮 check 函数的复杂度",check 函数本身的效率同样重要,不能只关注"外层用了二分"就忽视内层实现。

---

## 5. 浮点数二分查找

**签名/是什么:**
```
lo, hi = 0.0, x
while hi - lo > eps:      # 用精度阈值代替整数二分的 lo <= hi
    mid = (lo + hi) / 2
    ...
```

**一句话:** 在实数范围内二分查找(比如手写开平方根),不存在整数二分里"最终收缩到唯一一个下标"这种终止条件,只能用"区间宽度小于某个精度阈值"作为循环终止条件,得到的是一个足够接近真实答案的近似值。

**底层机制/为什么这样设计:** 整数二分依赖"区间越缩越小,最终必然收缩到空或者恰好一个元素"这个离散性质;实数区间理论上可以无限细分,不存在这样一个天然的终止点,所以必须人为设定一个精度阈值 `eps`,当区间宽度小于这个阈值时,认为已经"足够精确"并停止。这个精度阈值的选择本身是一个工程权衡——阈值越小,结果越精确,但需要的迭代次数也越多(每次迭代区间宽度减半,达到阈值 `eps` 需要约 `log2((hi-lo)/eps)` 次迭代,是对数级别,所以即使 `eps` 设得很小,实际迭代次数依然可控)。

**AI 研究/工程场景:** [huggingface-deep-dive 09 类](../huggingface-deep-dive/09-finetuning-comparison-lab.md)讲过的学习率超参数搜索,如果要在一个连续的学习率区间里,用二分思路搜索"满足某个训练稳定性判据的最大学习率",用的正是这里讲的浮点数二分查找思路(前提依然是"是否稳定"这个判断具有单调性,呼应知识点4)。

**可运行例子:**
```python
import math

def my_sqrt(x, eps=1e-9):
    if x < 0:
        raise ValueError("负数没有实数平方根")
    lo, hi = 0.0, max(1.0, x)   # x<1时,真实平方根比x本身大,右边界要保证覆盖到答案
    while hi - lo > eps:
        mid = (lo + hi) / 2
        if mid * mid < x:
            lo = mid
        else:
            hi = mid
    return lo

assert abs(my_sqrt(2) - math.sqrt(2)) < 1e-6
assert abs(my_sqrt(0.25) - math.sqrt(0.25)) < 1e-6   # x<1的情况,验证右边界处理正确
assert abs(my_sqrt(0) - 0) < 1e-6                      # 0的平方根是0
assert abs(my_sqrt(1) - 1) < 1e-6                      # 1的平方根是1

raised = False
try:
    my_sqrt(-4)
except ValueError:
    raised = True
assert raised   # 负数应该明确报错,而不是返回一个无意义的数字

# 交叉验证:对一批随机正数,和math.sqrt结果的误差应该稳定小于精度阈值
import random
random.seed(2)
max_error = 0.0
for _ in range(30):
    x = random.uniform(0, 1000)
    error = abs(my_sqrt(x) - math.sqrt(x))
    max_error = max(max_error, error)
assert max_error < 1e-5

print(f"OK: 浮点二分开平方在x<1/x=0/x=1/负数报错等边界情况下全部正确, "
      f"30组随机正数测试最大误差={max_error:.2e}(远小于精度阈值)")
```
本机实测:边界情况(`x<1`、`x=0`、`x=1`、负数应报错)全部正确;30 组随机正数测试中,和标准库 `math.sqrt` 的最大误差在 `1e-5` 量级以下,远小于设定的精度阈值。

**面试怎么问 + 追问链:** "浮点数二分和整数二分的终止条件为什么不一样?" → 追问"如果 `eps` 设置得过大或过小,分别会有什么后果?"(`eps` 过大,结果精度不够,可能不满足题目要求的误差范围;`eps` 过小,虽然结果更精确,但由于浮点数本身存在精度极限(64 位浮点数大约有 15~17 位有效十进制数字),`eps` 小到一定程度后,`hi - lo > eps` 这个判断可能因为浮点数精度问题永远无法在数值上真正小于极小的 `eps`,导致循环次数远超预期甚至理论上的死循环——这个追问检验的是对浮点数本身精度局限的理解,不是单纯的算法逻辑问题)。

**常见坑:**
1. `x < 1` 时,如果二分区间右边界直接设成 `x` 本身(而不是 `max(1.0, x)`),会导致真实平方根(比如 `sqrt(0.25)=0.5 > 0.25`)落在搜索区间之外,得到错误结果——这是本知识点例子里特意设计的一个边界测试点。
2. 精度阈值 `eps` 设置得脱离实际需求(比如题目只要求保留 4 位小数,却设了 `1e-15` 的阈值)——不仅浪费多余的迭代次数,还可能撞上上面追问链提到的浮点数精度极限问题。

---

## 6. 二维矩阵二分查找

**签名/是什么:**
```
矩阵每行从左到右递增，每列从上到下递增（或整体拉平后单调）
把二维坐标 (row, col) 映射成一维下标: row = idx // cols, col = idx % cols
```

**一句话:** 如果一个二维矩阵"拉平"成一维数组后依然保持整体有序(常见约定:每行递增,且下一行首元素大于上一行末元素),就可以直接复用一维二分查找,只需要把一维下标换算成对应的行列坐标。

**底层机制/为什么这样设计:** 这类矩阵的"整体有序"性质,本质上和一个真正的一维有序数组没有区别,只是存储时按行拆开摆放——`idx // cols` 和 `idx % cols` 这两个整数除法/取模运算,精确地把一维下标还原成对应的二维坐标,不需要真的把矩阵拉平复制成一维数组(那样会引入 O(rows×cols) 的额外空间和时间开销)。**需要特别注意**:并非所有"每行递增、每列递增"的矩阵整体拉平后都单调有序——如果矩阵的约定只是"每行从左到右递增、每列从上到下递增"但**没有**"下一行首元素大于上一行末元素"这个额外保证,拉平后就不是整体单调的,这种情况需要用另一套技巧(从右上角或左下角开始,每步排除一行或一列),不能直接套用一维二分。

**AI 研究/工程场景:** [huggingface-deep-dive 04 类](../huggingface-deep-dive/04-datasets-mechanics.md)讲过 Arrow 格式数据在内存里的连续布局,如果一个二维形状的数据集(比如按 `(batch, position)` 排列)整体按某个字段有序存储,查找特定值时用的正是这里"二维坐标映射一维下标"的思路,不需要真的对每一行分别做二分。

**可运行例子:**
```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target:
            return True
        elif val < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False

matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
assert search_matrix(matrix, 3) is True
assert search_matrix(matrix, 13) is False   # 不存在
assert search_matrix([], 1) is False          # 空矩阵
assert search_matrix([[]], 1) is False        # 只有空行
assert search_matrix([[5]], 5) is True         # 单元素矩阵命中

# 交叉验证:拉平成一维数组后用标准二分查找,结果应该完全一致
def flatten_and_check(matrix, target):
    flat = [v for row in matrix for v in row]
    return target in flat   # 用最朴素的方式交叉验证(不追求效率,只追求正确性对照)

import random
random.seed(3)
base_vals = sorted(random.sample(range(200), 24))
mat = [base_vals[i*4:(i+1)*4] for i in range(6)]   # 6行4列,整体有序拉平
for target in [base_vals[0], base_vals[-1], base_vals[12], 999]:
    assert search_matrix(mat, target) == flatten_and_check(mat, target)

print("OK: 二维矩阵二分在空矩阵/单元素/不存在等边界情况下全部正确, "
      "多组查询与'拉平后朴素判断是否存在'的交叉验证结果完全一致")
```
本机实测:边界情况(空矩阵、只有空行、单元素矩阵)全部正确;对一个真实构造的 6×4 整体有序矩阵,4 个不同查询目标的结果和"拉平后朴素判断"完全一致。

**面试怎么问 + 追问链:** "二维矩阵二分查找,把一维下标换算成二维坐标的公式是什么,为什么这样换算是对的?" → 追问"如果矩阵只保证'每行递增、每列递增',但不保证'整体拉平后有序',这道题还能用一维二分吗?"(不能——这种情况下矩阵不满足整体单调这个前提,需要改用"从右上角开始,比较后每步排除一行或一列"的 O(rows+cols) 解法,这个追问检验的是能否准确识别"一维二分适用"这个前提条件的边界,而不是看到"矩阵+有序"就无脑套用同一个模板)。

**常见坑:**
1. 把"每行递增、每列递增"和"整体拉平后单调有序"这两种不同的矩阵约定搞混——两者的解法完全不同,题目描述里的具体约定必须仔细确认,不能想当然。
2. 一维下标换算成二维坐标时,行列计算写反(比如把 `row = idx // cols` 和 `col = idx % cols` 弄反成 `row = idx // rows`)——这类错误在小规模测试里可能因为矩阵恰好是方阵而被掩盖,必须用非方阵(行数列数不同)的矩阵测试才能可靠暴露。

---

## 7. 二分查找常见坑

**签名/是什么:**
```
死循环陷阱：mid的计算方式和区间收缩方式不匹配，导致lo/hi不再变化
边界写法不一致：循环条件、mid计算、区间收缩三者必须成套使用，不能混搭
```

**一句话:** 二分查找的所有变体(`[lo,hi]` 闭区间 vs `[lo,hi)` 左闭右开区间)各自有一套自洽的写法,循环条件、`mid` 的取整方式、区间收缩方式必须三者匹配;混搭不同变体的写法片段,最容易导致的后果是**死循环**,而不是简单的逻辑错误。

**底层机制/为什么这样设计:** 死循环的典型场景:用左闭右开区间 `[lo, hi)`,`mid = (lo+hi)//2`(下取整),如果收缩区间时错误地写成 `lo = mid`(而不是 `lo = mid + 1`)——当 `hi = lo + 1` 时,`mid` 恰好等于 `lo`,`lo = mid` 这一步让 `lo` 的值完全没有变化,区间永远不会缩小,循环条件 `lo < hi` 也永远成立,程序卡死。这个陷阱的根源是:下取整的 `mid` 在区间只剩两个元素时会等于 `lo`,如果收缩逻辑没有把 `mid` 本身排除出去,`lo` 就有可能原地不动。

**AI 研究/工程场景:** 这类"边界条件写法不匹配导致死循环"的问题不是二分查找独有的——任何依赖"每次迭代状态必须真实推进"的循环结构(比如状态机的转移逻辑),都可能因为类似的边界疏漏产生同样的死循环故障模式,排查思路是通用的:检查每一条分支是否都能保证状态真实变化。

**可运行例子:**
```python
import signal

def buggy_search_potential_infinite_loop(nums, target, max_iterations=10000):
    """故意写一个有死循环风险的二分查找,用最大迭代次数做安全阀而不是真的死等"""
    lo, hi = 0, len(nums)   # 左闭右开区间[lo, hi)
    iterations = 0
    while lo < hi:
        iterations += 1
        if iterations > max_iterations:
            return "INFINITE_LOOP_DETECTED"
        mid = (lo + hi) // 2   # 下取整
        if nums[mid] < target:
            lo = mid + 1
        elif nums[mid] > target:
            hi = mid            # 正确
        else:
            lo = mid             # bug:命中时错误地写成lo=mid而不是直接return,
                                  # 且这个写法在特定情况下会导致lo不再推进
    return -1

# 构造一个真实触发死循环风险的场景:两元素区间,mid命中导致lo不再变化
result = buggy_search_potential_infinite_loop([3, 5], 3)
assert result == "INFINITE_LOOP_DETECTED"  # 真实复现了死循环风险,而不是理论上的担忧

def correct_search(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        elif nums[mid] > target:
            hi = mid
        else:
            return mid   # 命中立即返回,不存在"继续收缩"这一说
    return -1

assert correct_search([3, 5], 3) == 0
assert correct_search([3, 5], 5) == 1

# 混搭不同区间约定的反例:用闭区间的循环条件(lo<=hi),却用了左闭右开的收缩方式(hi=mid)
def mixed_convention_bug(nums, target, max_iterations=10000):
    lo, hi = 0, len(nums) - 1  # 声明是闭区间[lo, hi]
    iterations = 0
    while lo <= hi:            # 闭区间的循环条件
        iterations += 1
        if iterations > max_iterations:
            return "INFINITE_LOOP_DETECTED"
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid            # 错误:闭区间收缩应该用 mid-1,这里混用了左闭右开的写法
    return -1

result2 = mixed_convention_bug([1, 2], 1)
assert result2 == "INFINITE_LOOP_DETECTED"  # 真实复现:两种区间约定混搭导致死循环

print("OK: 现场复现两类真实的二分查找死循环场景(命中后错误收缩; 两种区间约定混搭), "
      "并验证对应的正确写法不会有这个问题")
```
本机实测:两个精心构造的"死循环风险"场景(命中后错误地继续收缩、混搭两种区间约定的写法)都真实触发了迭代次数暴涨,用最大迭代次数做安全阀成功检测到了这个问题——如果不加这个安全阀,这两个 bug 版本会真的卡死不返回。

**面试怎么问 + 追问链:** "如果你的二分查找代码在某些输入下死循环,你会怎么排查?" → 追问"能不能不靠运气,而是通过检查代码结构本身就发现这类隐患?"(检查方法论:先明确自己用的是哪种区间约定(闭区间还是左闭右开),再逐一核对循环条件、`mid` 计算方式、每个分支的区间收缩是否都严格符合这个约定下的标准写法——三者中只要有一处用了另一种约定的写法,就是潜在的死循环隐患;这比"多测几个例子祈祷不出问题"可靠得多,也是终面里能体现"系统性调试能力"而不只是"会写代码"的一个关键区分点)。

**常见坑:**
1. 在同一个函数里混用了两种区间约定的写法片段(比如循环条件是闭区间的 `lo <= hi`,但收缩逻辑用了左闭右开的 `hi = mid`)——通常是从两个不同地方抄来的代码片段拼接导致,写二分查找前先明确选定一种约定,通篇保持一致。
2. 只用"能正常返回结果"的测试用例验证代码,没有专门测试"两元素""目标在边界"这类最容易触发死循环的小规模场景——死循环风险往往在区间缩小到只剩 1~2 个元素时才会暴露,大规模测试用例反而不容易踩中这个陷阱。

---

*本篇 7 个知识点全部在仓库根目录 `.venv` 真实测试验证(含边界情况覆盖、随机交叉验证、以及两类死循环场景的真实复现)。*
