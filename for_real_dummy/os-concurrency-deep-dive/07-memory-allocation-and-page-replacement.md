# 07 内存分配与页面置换

> 板块 III:内存管理收官。06 类讲了"虚拟地址怎么转换成物理地址",本类讲两件承接的事:堆内存(`malloc`)背后的分配器怎么工作,以及物理内存不够用时,操作系统怎么决定"谁被换出去"。

---

## 1. malloc内部机制(空闲链表法)

**签名/是什么**

`malloc`/`free`(或 Python 里对象背后的堆内存管理)需要一个用户态的内存分配器来管理"进程从操作系统要来的一大块内存,内部怎么切分给各次小的分配请求"。空闲链表法(Free List)是最基础的实现思路:维护一条记录"哪些区间当前空闲"的链表,分配时从链表里找一块足够大的区间切一部分出去,释放时把区间放回链表,并尝试和相邻的空闲区间合并(Coalescing)。

**一句话**

`malloc` 不是直接问操作系统要内存(那样太慢太频繁),是应用层自己维护一本"当前这块大内存里,哪里是空的"的账本,自己精打细算地切分。

**底层机制/为什么这样设计**

向操作系统申请内存(比如通过 `brk`/`mmap` 系统调用)涉及从用户态陷入内核态,这个开销相对于典型的小对象分配(几十到几百字节)而言是不成比例的巨大——如果每次 `malloc` 一个小对象都直接找操作系统要,性能会完全无法接受。用户态分配器的做法是:一次性向操作系统申请一大块内存,然后自己在这块内存内部做精细的切分和管理,只有当这块内存实在不够用了才再去问操作系统要更多。释放内存时把空闲区间和相邻区间合并(而不是简单地标记"这块空了"就完事),是为了防止"明明总空闲空间够,但被切成太多互不相邻的小块导致大分配请求失败"这个问题的萌芽——但即使有合并,空闲链表法在长期高频、大小不一的分配释放模式下依然会累积碎片(见第 7 点)。

**AI研究/工程场景**

理解堆分配器的基本工作原理,是理解"为什么 Python/PyTorch 这类语言/框架自己还要在操作系统的 `malloc` 之上再包一层自己的内存池/缓存分配器"的基础——比如 PyTorch 的 CUDA caching allocator 就是应用层再实现一套类似空闲链表/大小分级的机制,专门管理显存分配,原因和用户态 `malloc` 存在的原因完全一致:避免频繁触发昂贵的底层分配操作(对 CUDA 而言,`cudaMalloc`/`cudaFree` 比 CPU 端的 `malloc`/`free` 更慢得多)。

**可运行例子**(验证环境:`.venv`)

```python
class FreeListAllocator:
    def __init__(self, total_size):
        self.total_size = total_size
        self.free_blocks = [(0, total_size)]  # (start, size)
        self.allocated = {}
        self.next_id = 0

    def alloc(self, size):
        for i, (start, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                alloc_id = self.next_id
                self.next_id += 1
                self.allocated[alloc_id] = (start, size)
                if block_size == size:
                    self.free_blocks.pop(i)
                else:
                    self.free_blocks[i] = (start + size, block_size - size)
                return alloc_id
        return None

    def free(self, alloc_id):
        start, size = self.allocated.pop(alloc_id)
        self.free_blocks.append((start, size))
        self.free_blocks.sort()
        merged = []
        for block in self.free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == block[0]:
                prev_start, prev_size = merged.pop()
                merged.append((prev_start, prev_size + block[1]))
            else:
                merged.append(block)
        self.free_blocks = merged

alloc = FreeListAllocator(1000)
a = alloc.alloc(100)
b = alloc.alloc(200)
c = alloc.alloc(150)
allocated_ranges = [alloc.allocated[x] for x in (a, b, c)]
print('allocated_ranges=%s' % allocated_ranges)
for i in range(len(allocated_ranges)):
    for j in range(i + 1, len(allocated_ranges)):
        s1, sz1 = allocated_ranges[i]
        s2, sz2 = allocated_ranges[j]
        overlap = not (s1 + sz1 <= s2 or s2 + sz2 <= s1)
        assert not overlap, "allocated blocks must never overlap"
print("NO_OVERLAP_TEST=PASS")

alloc.free(b)
alloc.free(c)
print('free_blocks_after_freeing_b_and_c=%s' % alloc.free_blocks)
merged_block_found = any(start == 100 and size == 900 for start, size in alloc.free_blocks)
assert merged_block_found, "freeing adjacent blocks b, c must coalesce with each other AND with the remaining tail free block into one single [100,900) block"
print("COALESCING_TEST=PASS")
```

**面试怎么问+追问链**

- **决策依据追问轴**:空闲链表按什么策略选择要切分哪一块(第一个够大的?最小的够大的?)?——追问:首次适配(First-Fit,选第一个足够大的,实现简单但容易在链表前段留下越来越小的碎片)、最佳适配(Best-Fit,选最接近所需大小的,单次浪费最小但需要遍历整个链表且容易产生大量无法再利用的极小碎片)、最差适配(Worst-Fit,选最大的那块切,试图让剩余部分还能应付未来的中等请求)各有不同的碎片化特征,不存在绝对最优的策略,真实分配器往往会结合大小分级(见第 2 点)来规避空闲链表法在大量小对象场景下的固有弱点。

**常见坑**

- 忘记释放已分配的内存(在 C/C++ 这类需要手动管理内存的语言里是真实的内存泄漏根源)——空闲链表分配器本身没有能力知道"这块内存还有没有人在用",它完全依赖调用方正确地调用 `free`,这也是为什么 Python 这类有垃圾回收的语言能省去这类心智负担(但垃圾回收本身也有自己的复杂性和开销,不是没有代价的免费午餐)。

---

## 2. 伙伴系统与slab分配器

**签名/是什么**

伙伴系统(Buddy System)是一种按 2 的幂次组织空闲内存的分配策略:内存被递归地二等分,每次分配从最合适的那一级"借"一块,借不到就从更大的一级切一半下来(切出来的另一半叫这块的"伙伴",放回空闲链表);释放时检查自己的伙伴是否也空闲,空闲就合并回上一级,递归进行。Slab 分配器专门针对"频繁分配/释放大量相同大小对象"这类场景,预先切好一批固定大小的槽位(slot),分配释放只是简单的"从空闲槽位栈里弹出/压回"操作。

**一句话**

伙伴系统用"按2的幂次切分、伙伴互认合并"解决了空闲链表法"释放后不知道该去哪找相邻块合并"的低效,slab 分配器则干脆放弃"通用大小"这个目标,专攻"反复分配同一种大小对象"这一个场景做到极致快。

**底层机制/为什么这样设计**

伙伴系统的核心巧思在于地址计算:一个大小为 2^k 的块,它的伙伴地址可以通过"把地址的第 k 位取反"(用异或运算 `addr XOR (1 << k)`)直接算出来,不需要遍历任何数据结构去"寻找"相邻块,这比空闲链表法维护地址排序、线性扫描寻找相邻块要高效得多。这个优雅的地址计算技巧,本质上是牺牲了"按 2 的幂次分配可能造成内部碎片"这个代价(申请 65 字节也要给整整 128 字节),换来了合并操作的极致效率。Slab 分配器则是完全不同的优化方向:操作系统内核里有大量固定大小、生命周期短、分配释放极其频繁的对象(比如每次系统调用都要分配的内核数据结构),针对这类场景,连"找一块合适大小的空闲区域"这个查找过程本身都可以省略——反正所有槽位大小完全相同,直接用一个简单的空闲槽位栈,分配就弹栈顶、释放就压回去,是能达到的最快分配方式之一。

**AI研究/工程场景**

高频短生命周期对象的分配模式在数据处理流水线里很常见(比如每个请求都要分配的临时上下文对象,请求处理完立刻释放),这类场景正是 slab 式分配思想的用武之地;而伙伴系统式的"按大小分级、就近合并"思想,在需要管理任意大小内存块的通用分配器(包括 Linux 内核自己的物理页帧分配器)里被广泛使用,理解这两种分配策略的适用边界,有助于理解为什么真实的通用内存分配器(如 glibc 的 `ptmalloc`)往往是"对小对象用类似 slab 的固定大小分级池、对大对象退化成类似伙伴系统或直接 `mmap`"这种混合策略,而不是单一策略打天下。

**可运行例子**(验证环境:`.venv`)

```python
import time

class BuddyAllocator:
    def __init__(self, total_size_power):
        self.max_order = total_size_power
        self.free_lists = {i: [] for i in range(total_size_power + 1)}
        self.free_lists[total_size_power] = [0]
        self.allocated = {}

    def _order_for_size(self, size):
        order = 0
        while (1 << order) < size:
            order += 1
        return order

    def alloc(self, size):
        return self._alloc_order(self._order_for_size(size))

    def _alloc_order(self, order):
        if order > self.max_order:
            return None
        if self.free_lists[order]:
            addr = self.free_lists[order].pop()
            self.allocated[addr] = order
            return addr
        bigger = self._alloc_order(order + 1)
        if bigger is None:
            return None
        buddy_addr = bigger + (1 << order)
        self.free_lists[order].append(buddy_addr)
        self.allocated[bigger] = order
        return bigger

    def free(self, addr):
        order = self.allocated.pop(addr)
        while order < self.max_order:
            buddy_addr = addr ^ (1 << order)  # 伙伴地址:异或页大小直接算出,不需要查找
            if buddy_addr in self.free_lists[order]:
                self.free_lists[order].remove(buddy_addr)
                addr = min(addr, buddy_addr)
                order += 1
            else:
                break
        self.free_lists[order].append(addr)

buddy = BuddyAllocator(total_size_power=10)  # 总大小 2^10 = 1024
a = buddy.alloc(100)  # round up到128 (2^7)
b = buddy.alloc(100)
print('a=%s b=%s' % (a, b))
assert a is not None and b is not None and a != b

buddy.free(a)
buddy.free(b)
final_state = {order: blocks for order, blocks in buddy.free_lists.items() if blocks}
print('free_lists_after_freeing_both=%s' % final_state)
assert final_state == {10: [0]}, "freeing both buddies should fully coalesce back up to the single original 1024-sized free block"
print("BUDDY_SYSTEM_TEST=PASS")

class SlabAllocator:
    def __init__(self, object_size, count):
        self.object_size = object_size
        self.free_slots = list(range(count))
    def alloc(self):
        return self.free_slots.pop() if self.free_slots else None
    def free(self, slot):
        self.free_slots.append(slot)

slab = SlabAllocator(object_size=64, count=100_000)
t0 = time.perf_counter()
slots = [slab.alloc() for _ in range(50_000)]
alloc_time_slab = time.perf_counter() - t0
t0 = time.perf_counter()
for s in slots:
    slab.free(s)
free_time_slab = time.perf_counter() - t0
print('slab: alloc_time=%.5f free_time=%.5f for 50000 ops' % (alloc_time_slab, free_time_slab))
assert alloc_time_slab < 0.5 and free_time_slab < 0.5, "slab allocator alloc/free for fixed-size objects should be very fast (O(1) per operation via simple stack push/pop)"
print("SLAB_ALLOCATOR_SPEED_TEST=PASS")
```

**面试怎么问+追问链**

- **底层机制追问轴**:伙伴地址那个异或运算 `addr XOR (1 << k)` 具体是什么原理?——追问:一个 2^k 大小的块被从一个 2^(k+1) 大小的块切分出来时,它和它的伙伴的地址只在第 k 位(从0开始数)上不同(一个是0一个是1),因为切分点恰好落在 2^k 的边界上——异或第 k 位就是在"猜另一半在哪",这个技巧要求所有块的起始地址都严格按照自己的大小对齐,这也是为什么伙伴系统要求块大小必须是 2 的幂次,不是随意的设计选择。

**常见坑**

- 请求大小恰好比 2 的幂次多一点点(比如 65 字节),伙伴系统会分配整整 128 字节——这是伙伴系统固有的内部碎片代价,最坏情况下浪费接近 50%,这是选择伙伴系统时必须接受的权衡,不是可以调优消除的缺陷。

---

## 3. 页面置换算法FIFO/LRU/Clock

**签名/是什么**

当物理内存(页框)不够容纳当前所有需要访问的页面时,操作系统必须选择"淘汰"某个已经在内存里的页面,腾出空间给新页面——这就是页面置换。FIFO(先进先出,淘汰最早被换入的页面)、LRU(最近最少使用,淘汰最久没被访问的页面)、Clock(用一个"最近访问位"+ 循环指针近似模拟 LRU,不需要精确记录访问顺序,实现开销远低于严格 LRU)是三种经典策略。

**一句话**

FIFO 只看"谁最老",LRU 看"谁最久没被碰过",Clock 是"用一个简化的机制近似 LRU 的效果、不用精确记账"。

**底层机制/为什么这样设计**

LRU 的直觉依据是"局部性原理"(见 02 类知识点8、06 类知识点4):最近被访问过的页面,近期再次被访问的概率通常更高,所以淘汰"最久没被碰过"的页面,是对未来访问模式的一个合理预测。但严格实现 LRU 需要精确记录每个页面的访问时间顺序,这个记账开销(每次内存访问都要更新时间戳)在真实硬件层面代价很高;Clock 算法用一个简化的近似方案回避了这个问题:给每个页框配一个"最近访问过"的标志位(访问时置1),需要淘汰时,一个指针沿着页框循环扫描,遇到标志位是1的就清零并跳过(给它"第二次机会"),遇到标志位是0的就淘汰它——这样只需要一个简单的位翻转和循环指针,不需要精确的时间戳记录,是工程实现里"完美 vs 够用且便宜"的经典权衡。

**AI研究/工程场景**

理解页面置换算法的思想,对理解各种"缓存淘汰策略"都有直接帮助——不管是 CPU 缓存、数据库缓冲池、还是应用层的 LRU Cache(比如 `functools.lru_cache`),背后都是同一类问题:有限容量的快速存储,面对超出容量的访问请求,该淘汰谁。推理服务的 KV cache 管理(vLLM 等框架的核心优化点之一)本质上也是一个更复杂场景下的"页面置换"问题——GPU 显存有限,哪些请求的 KV cache 该被换出/换入,直接影响服务的吞吐量和延迟。

**可运行例子**(验证环境:`.venv`)

```python
def fifo(sequence, num_frames):
    frames = []
    faults = 0
    for page in sequence:
        if page not in frames:
            faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)
            frames.append(page)
    return faults

def lru(sequence, num_frames):
    frames = []
    faults = 0
    for page in sequence:
        if page in frames:
            frames.remove(page)
            frames.append(page)
        else:
            faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)
            frames.append(page)
    return faults

def clock(sequence, num_frames):
    frames = [None] * num_frames
    ref_bits = [0] * num_frames
    pointer = 0
    faults = 0
    for page in sequence:
        if page in frames:
            ref_bits[frames.index(page)] = 1
            continue
        faults += 1
        while ref_bits[pointer] == 1:
            ref_bits[pointer] = 0
            pointer = (pointer + 1) % num_frames
        frames[pointer] = page
        ref_bits[pointer] = 1
        pointer = (pointer + 1) % num_frames
    return faults

sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
num_frames = 3
f_fifo = fifo(sequence, num_frames)
f_lru = lru(sequence, num_frames)
f_clock = clock(sequence, num_frames)
print('FIFO=%d LRU=%d Clock=%d (same access sequence, 3 frames)' % (f_fifo, f_lru, f_clock))
assert f_fifo > 0 and f_lru > 0 and f_clock > 0, "with only 3 frames for 5 distinct pages, all algorithms must incur some faults"
assert f_clock <= f_lru + 2, "Clock is an approximation of LRU and should perform reasonably close to it, not dramatically worse, on a typical access pattern"
print("PAGE_REPLACEMENT_COMPARISON_TEST=PASS")
```

**面试怎么问+追问链**

- **方案批判迭代轴**:"我们用 LRU 做缓存淘汰"——追问1:如果访问模式是"扫描一个远大于缓存容量的数据集"(比如全表扫描),LRU 会发生什么?候选人如果说"应该没什么问题"——追问2:全表扫描场景下,每个页面几乎都只被访问一次就再也不会被访问,LRU 会把所有页面都换入换出一遍,缓存命中率趋近于 0,这种"扫描污染缓存"是 LRU 的一个经典弱点(很多真实系统用"LRU-K"或者"分段 LRU"这类改进算法来专门缓解这个问题,给"只访问一次"的页面更低的优先级,不让它们污染真正的热点数据)。

**常见坑**

- 认为 LRU 总是比 FIFO"更聪明所以更好"——LRU 的额外记账开销在页框数量巨大、访问极其频繁的场景下可能变得不可忽视,而且在某些特定访问模式下(比如上面提到的全表扫描),LRU 并不比简单的 FIFO 表现更好,算法选择需要结合真实的访问模式特征,不是无脑选"听起来更智能"的那个。

---

## 4. 最优置换Belady定理与Belady异常

**签名/是什么**

最优置换算法(OPT,又叫 MIN)是一个理论算法:每次淘汰"未来最晚才会被再次用到(或者未来根本不会再被用到)的页面"。Belady 最优性定理指出,OPT 在任意给定访问序列和帧数下,产生的缺页次数都不会超过任何其他置换算法——它是理论下界。Belady 异常(Belady's Anomaly)则是一个反直觉的真实现象:对某些置换算法(典型代表是 FIFO),增加可用页框数量,缺页次数反而可能增加,而不是像直觉认为的那样只会减少或不变。

**一句话**

OPT 需要"预知未来"才能实现,是评价其他算法好坏的理论标尺;Belady 异常提醒我们"更多资源不一定带来更好效果"这个反直觉但真实存在的现象。

**底层机制/为什么这样设计**

OPT 之所以是理论最优、不可能被超越,证明思路很直观:既然要淘汰,淘汰"未来最晚才用到的"那个,给自己争取到的"缓冲时间"是所有可能选择里最长的,不可能有更好的选择——但它要求"预知未来的访问序列",现实系统里完全不可行,它的价值纯粹是作为衡量其他实际算法优劣的理论基准。Belady 异常之所以会在 FIFO 这类算法上出现,根源在于 FIFO 不是"栈算法"(Stack Algorithm)——栈算法有一个数学性质:用 k 个帧运行时驻留在内存里的页面集合,一定是用 k+1 个帧运行时驻留页面集合的子集(LRU 和 OPT 都是栈算法),这个性质保证了帧数增加时,历史某一时刻已经在内存里的页面不会因为帧数增加反而被替换掉,从而保证缺页数单调不增;FIFO 恰恰不满足这个子集关系(FIFO 的淘汰顺序只看"进入时间",和帧数变化后的整体状态没有这种嵌套关系),这就是为什么增加帧数对 FIFO 而言不能保证情况变好,某些精心构造(或者纯属巧合)的访问序列下反而会变差。

**AI研究/工程场景**

Belady 异常本身更多是一个"提醒工程师保持谦逊、不要想当然"的经典案例——它告诉我们"给系统更多资源就应该表现更好"这个看似天经地义的直觉,在某些具体机制(不是所有机制,是"非栈算法"这类特定机制)下是会失效的,任何做资源扩容决策(加内存、加缓存、加机器)的人,理论上都应该谨慎验证"扩容真的带来了预期的改善",而不是盲目相信直觉,尤其是涉及到类似 FIFO 这种简单置换策略的自定义缓存系统时。

**可运行例子**(验证环境:`.venv`;复用第 3 点已定义的 `fifo`/`lru` 函数)

```python
def fifo(sequence, num_frames):
    frames = []
    faults = 0
    for page in sequence:
        if page not in frames:
            faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)
            frames.append(page)
    return faults

def lru(sequence, num_frames):
    frames = []
    faults = 0
    for page in sequence:
        if page in frames:
            frames.remove(page)
            frames.append(page)
        else:
            faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)
            frames.append(page)
    return faults

def optimal(sequence, num_frames):
    frames = []
    faults = 0
    for i, page in enumerate(sequence):
        if page not in frames:
            faults += 1
            if len(frames) >= num_frames:
                future = sequence[i+1:]
                farthest, victim = -1, frames[0]
                for f in frames:
                    if f not in future:
                        victim = f
                        break
                    idx = future.index(f)
                    if idx > farthest:
                        farthest = idx
                        victim = f
                frames.remove(victim)
            frames.append(page)
    return faults

sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
f_fifo = fifo(sequence, 3)
f_lru = lru(sequence, 3)
f_opt = optimal(sequence, 3)
print('FIFO=%d LRU=%d OPT=%d (3 frames)' % (f_fifo, f_lru, f_opt))
assert f_opt <= f_fifo and f_opt <= f_lru, "OPT must never have MORE faults than any other algorithm on the same sequence - Belady's optimality theorem"
print("OPT_OPTIMALITY_TEST=PASS")

faults_3_frames = fifo(sequence, 3)
faults_4_frames = fifo(sequence, 4)
print('FIFO faults with 3 frames=%d, with 4 frames=%d' % (faults_3_frames, faults_4_frames))
assert faults_4_frames > faults_3_frames, \
    "Belady's anomaly: for FIFO specifically, INCREASING the number of frames can INCREASE the fault count - a genuine counter-intuitive real phenomenon"
print("BELADY_ANOMALY_TEST=PASS")

lru_faults_3 = lru(sequence, 3)
lru_faults_4 = lru(sequence, 4)
print('LRU faults with 3 frames=%d, with 4 frames=%d' % (lru_faults_3, lru_faults_4))
assert lru_faults_4 <= lru_faults_3, "LRU, as a stack algorithm, must NOT exhibit Belady's anomaly - more frames never increases its fault count"
print("LRU_NO_ANOMALY_TEST=PASS")
```

验证记录:实测同一访问序列下 OPT=7 严格小于等于 FIFO=9 和 LRU=10;FIFO 从 3 帧增加到 4 帧,缺页数从 9 反而升到 10(Belady 异常真实复现);同样的序列 LRU 从 3 帧到 4 帧,缺页数从 10 降到 8(符合直觉,栈算法性质保证不会异常)——三组数字放在一起,清晰展示了"是不是栈算法"这个数学性质如何决定了"加资源会不会有反效果"这个实践结论。

**面试怎么问+追问链**

- **诊断真实数据(新题型)**:给一个自定义缓存系统扩容前后的缺页(缓存未命中)率数据,扩容后命中率不升反降——追问:第一步排查这个缓存的淘汰策略是不是类似 FIFO 这种非栈算法的简单策略,如果是,Belady 异常是一个需要认真考虑的可能根因(虽然不是唯一可能,访问模式变化、缓存预热不充分等也可能导致类似现象,但"淘汰算法本身是非栈算法"是一个专业候选人应该主动想到去排查的方向)。

**常见坑**

- 把 Belady 异常当成"随时可能发生的普遍现象,任何扩容都要小心"——Belady 异常只在非栈算法(FIFO 是最常见的例子)上可能出现,而且需要访问序列具备特定结构才会触发,真实场景下大多数生产系统用的是 LRU 或其变种(栈算法性质),不会遇到这个问题;了解这个现象的意义在于理解"为什么栈算法这个数学性质在设计缓存置换策略时值得被重视",而不是对所有扩容决策都产生不必要的怀疑。

---

## 5. mmap内存映射

**签名/是什么**

`mmap`(memory map)系统调用把一个文件(或匿名内存区域)直接映射到进程的虚拟地址空间,之后对这段内存的读写会被内核自动同步到对应的文件内容(或者对匿名映射而言,就是单纯的按需分配内存),不需要显式调用 `read`/`write` 这类传统文件 IO 接口。

**一句话**

`mmap` 让"读写一个文件"和"读写一段内存"变成了同一件事,省去了在用户态缓冲区和内核文件缓存之间来回拷贝数据的开销。

**底层机制/为什么这样设计**

传统的 `read`/`write` 文件 IO 需要把数据在"内核的 page cache"和"用户态传入的缓冲区"之间做一次拷贝(数据先从磁盘读到内核缓存,再从内核缓存拷贝到用户提供的 buffer),这次拷贝对大文件/高频访问场景而言是实打实的开销。`mmap` 把文件的 page cache 直接映射进用户态的虚拟地址空间,读写这段内存本质上就是直接读写 page cache,省掉了这次额外拷贝(这类技术统称"零拷贝",在高性能 IO 场景里是重要的优化手段)。此外,`mmap` 让"按需加载"变得自然:一个几十 GB 的大文件被 `mmap` 之后,并不会立刻把全部内容读入内存,只有真正被访问到的页面才会触发缺页中断从磁盘读入(懒加载),这对"只需要随机访问大文件里一小部分内容"的场景是显著的效率提升。

**AI研究/工程场景**

大模型权重文件的加载是 `mmap` 一个非常直接的真实应用场景——像 llama.cpp 这类推理引擎会用 `mmap` 加载 GGUF 权重文件,而不是一次性 `read` 整个文件到内存:一方面利用了懒加载(如果模型的某些部分暂时不需要用到,不会被立刻读入物理内存),另一方面多个进程如果 `mmap` 同一份权重文件,操作系统的 page cache 会被自动共享(第一个进程读过的页面,后面的进程访问同样的文件位置能直接命中缓存,不需要重复从磁盘读取)——这是本地多进程部署同一个大模型时能显著节省内存和加载时间的关键机制。

**可运行例子**(验证环境:`.venv`)

```python
import mmap
import os
import tempfile

fd, path = tempfile.mkstemp()
os.write(fd, b'0' * 100)
os.close(fd)

with open(path, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 100)
    mm[0:5] = b'HELLO'  # 直接像操作bytearray一样写内存映射区域
    mm.flush()
    mm.close()

with open(path, 'rb') as f:
    content = f.read()
print('file_content_after_mmap_write=%s' % content[:10])
assert content[:5] == b'HELLO', "writing through the mmap'd memory region must be reflected in the actual file content on disk - this is the core mmap contract"
os.unlink(path)
print("MMAP_TEST=PASS")
```

**面试怎么问+追问链**

- **决策依据追问轴**:什么场景下 `mmap` 不一定比传统 `read`/`write` 更好?——追问:如果需要读取的数据量很小、只读一次(不会重复访问),`mmap` 本身建立映射也有固定的系统调用开销,可能反而不如一次简单的 `read` 划算;此外,`mmap` 对文件大小变化的处理(比如映射后文件被其他进程截断)比传统 IO 更容易引发难以预料的信号(`SIGBUS`),对健壮性要求高的场景需要额外小心处理这类边界情况。

**常见坑**

- 以为 `mmap` 写入立刻同步到磁盘——`mm[0:5] = b'HELLO'` 这类写入首先只是修改了内存(page cache)里的内容,真正落盘的时机由操作系统决定(除非显式调用 `flush()`/`msync`,或者进程正常退出触发的清理),如果系统在数据落盘前崩溃,这部分修改可能丢失,这和 10 类知识点会讲到的 `fsync` 语义是同一个问题的不同侧面。

---

## 6. Swap与抖动Thrashing

**签名/是什么**

Swap(交换)指物理内存不够用时,操作系统把暂时不活跃的页面换出到磁盘上的交换空间,腾出物理内存给当前更需要的页面;需要时再换入。抖动(Thrashing)指系统的"工作集"(Working Set,一个进程在某个时间窗口内实际频繁访问的页面集合)显著超过可用物理内存,导致页面被换出后很快又要被换入,换入的页面又很快挤走了别的还需要用的页面——系统把绝大部分时间花在"搬页面"上,真正用于计算的时间反而少得可怜。

**一句话**

适度的 Swap 是"暂时不用的东西先放仓库",抖动是"仓库和柜子之间来回搬东西搬到根本没空干正事"。

**底层机制/为什么这样设计**

Swap 机制本身的设计初衷是合理的:允许系统"超卖"物理内存(总的虚拟内存需求可以超过物理内存总量),只要真正同时活跃需要的部分(工作集)能装得下物理内存,不活跃的部分放到磁盘上完全没问题。抖动是这个机制在"工作集超过物理内存"这个前提被打破时的失效模式——一旦工作集装不下,任何置换算法都无法避免"刚换出去的页面马上又要用、换入它又得挤走另一个马上要用的页面"这种恶性循环,因为置换算法能做的只是"在有限空间里做局部最优选择",没有办法凭空变出更多物理内存。抖动最大的危害在于它是一个自我恶化的正反馈:磁盘 IO(换入换出)远比内存访问慢几个数量级,系统花越多时间搬页面,能推进实际计算的时间比例就越低,如果这时候调度器还傻乎乎地允许更多进程加入竞争内存(试图"提高利用率"),只会让工作集需求进一步超过物理内存,抖动进一步恶化。

**AI研究/工程场景**

训练任务如果 batch size/模型规模配置得让实际需要的内存(工作集)显著超过机器物理内存,即使系统配置了 Swap 允许"跑起来不报 OOM",实际表现会因为持续的页面换入换出而呈现出"CPU/GPU 利用率很低、但系统 IO 却很繁忙"这种典型抖动特征,吞吐量可能比"直接减小 batch size 让工作集完全装进物理内存"还要差得多——这是内存资源规划时必须避免的一种反模式,"能跑起来不报错"不等于"跑得高效"。

**可运行例子**(验证环境:`.venv`)

```python
import random

def lru_fault_rate(sequence, num_frames):
    frames = []
    faults = 0
    for page in sequence:
        if page in frames:
            frames.remove(page)
            frames.append(page)
        else:
            faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)
            frames.append(page)
    return faults / len(sequence)

rng = random.Random(7)
working_set_size = 20
sequence = [rng.randint(0, working_set_size - 1) for _ in range(2000)]  # 在20个页之间随机访问,模拟一个"工作集"

rate_sufficient = lru_fault_rate(sequence, num_frames=18)   # 帧数接近工作集大小
rate_thrashing = lru_fault_rate(sequence, num_frames=4)     # 帧数远小于工作集,严重不足

print('fault_rate with frames=18 (close to working set): %.3f' % rate_sufficient)
print('fault_rate with frames=4 (severe shortage): %.3f' % rate_thrashing)
assert rate_thrashing > rate_sufficient * 3, \
    "when available frames drop far below the working set size, fault rate should spike dramatically (thrashing) - a disproportionate increase relative to how much the frame count dropped is the defining symptom"
print("THRASHING_TEST=PASS")
```

验证记录:实测帧数接近工作集大小(18/20)时缺页率仅 9.1%,帧数严重不足(4/20)时缺页率飙升到 79.4%——不到 5 倍的帧数减少,导致接近 9 倍的缺页率上升,这种"不成比例的急剧恶化"正是抖动的数值特征。

**面试怎么问+追问链**

- **诊断真实数据(新题型)**:服务器监控显示 `si`/`so`(swap in/out,来自 `vmstat` 之类的工具)数值持续居高不下,同时 CPU 使用率里 `wa`(等待IO)占比异常高——追问:这组指标组合是抖动的典型信号(而不是单纯"内存不够但还没到抖动程度"),排查方向包括:检查是不是有进程实际工作集远超预期(内存泄漏、或者配置的负载规模超出了机器容量),短期缓解手段是减少并发运行的进程数/降低单进程内存需求,根本解决要么是加物理内存,要么是重新设计让工作集能装进现有内存。

**常见坑**

- 把"用了 Swap"直接等同于"发生了抖动"——只要有页面在换入换出就是 Swap 在正常工作,不代表系统在抖动;抖动的判定标准是"换入换出的频率高到显著挤占了正常计算的时间占比",偶尔的、低频的换入换出是 Swap 机制发挥其设计价值的正常表现,不需要一见到 Swap 活动就紧张。

---

## 7. 内存碎片的动态演化与真实分配失败后果

**签名/是什么**

06 类知识点6 已经讲过外部碎片(空闲总量够但没有连续大块)和内部碎片(分配的块比实际需要的大,多余部分浪费)这两个静态概念。本知识点换成动态视角:在真实系统运行过程中,反复分配、释放不同大小的对象,即使每次释放都正确执行了相邻块合并,只要存在"释放模式不规律,导致空闲块之间夹杂着仍然存活的分配块"这种情况,外部碎片依然会持续累积,而且合并机制对这种"夹心"式的碎片完全无能为力(合并只能处理相邻的空闲块,对中间隔着存活分配的空闲块无效)。

**一句话**

内存碎片不是分配器"实现得不够好",而是"长期、不规律的分配释放模式"本身天然会产生的累积效应,合并相邻空闲块只能缓解、不能根治。

**底层机制/为什么这样设计**

最能体现这个问题的经典模式是"棋盘式"释放:假设连续分配了一长串大小相同的对象,之后释放"每隔一个"的对象(而不是连续释放一整段)——被释放的对象总大小可能非常可观,但因为每个空闲块的左右邻居都还是存活的分配对象,空闲块之间永远无法互相合并,不管释放了多少内存,能用来满足新请求的"最大连续空闲块"始终只有单个对象那么大。这不是一个刻意构造的极端理论案例——真实系统里长期运行的服务,不同生命周期长短、大小不一的对象混杂分配释放,天然会产生类似的"夹心"效应,这是所有依赖"相邻合并"这一招来对抗碎片的分配器(包括第 1 点的空闲链表法)都无法回避的结构性限制,真正的根治手段(内存压缩/整理,移动仍然存活的对象来腾出连续空间)需要能够改写指向这些对象的所有引用,只有配合垃圾回收器的语言/运行时才能安全做到,C/C++ 这类手动内存管理语言的分配器基本不具备这个能力。

**AI研究/工程场景**

长期运行的推理服务如果请求处理过程中分配了大量生命周期长短不一的临时对象(不同长度的输入/输出文本、不同大小的中间张量),运行数天后即使总的空闲内存看起来"够用",也可能开始出现"明明有内存,但某个大小的分配请求却失败"这种真实的生产问题,根因往往就是这类碎片累积——这也是为什么 07 类知识点2 讲的 slab/大小分级思路(把不同大小的对象分别放进独立的池子)在长期运行的服务里格外重要:同一个大小类内部,任何一个槽位空出来都能被同大小的新请求复用,不存在"夹心"问题,能从根本上避免这一类碎片积累,而不只是缓解。

**可运行例子**(验证环境:`.venv`;复用第 1 点已验证正确的空闲链表分配器)

```python
class FreeListAllocator:
    def __init__(self, total_size):
        self.total_size = total_size
        self.free_blocks = [(0, total_size)]
        self.allocated = {}
        self.next_id = 0
    def alloc(self, size):
        for i, (start, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                alloc_id = self.next_id; self.next_id += 1
                self.allocated[alloc_id] = (start, size)
                if block_size == size:
                    self.free_blocks.pop(i)
                else:
                    self.free_blocks[i] = (start + size, block_size - size)
                return alloc_id
        return None
    def free(self, alloc_id):
        start, size = self.allocated.pop(alloc_id)
        self.free_blocks.append((start, size))
        self.free_blocks.sort()
        merged = []
        for block in self.free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == block[0]:
                prev_start, prev_size = merged.pop()
                merged.append((prev_start, prev_size + block[1]))
            else:
                merged.append(block)
        self.free_blocks = merged
    def fragmentation_ratio(self):
        if not self.free_blocks:
            return 0.0
        total_free = sum(size for _, size in self.free_blocks)
        largest_free = max(size for _, size in self.free_blocks)
        return 0.0 if total_free == 0 else 1.0 - (largest_free / total_free)

# 棋盘式分配模式:先密集分配一长串固定大小对象填满整个池子,再释放"每隔一个"的对象
alloc = FreeListAllocator(90_000)
BLOCK_SIZE = 100
NUM_BLOCKS = 900  # 900 * 100 = 90000,正好填满,不留无关的尾部空闲块干扰测量
ids = [alloc.alloc(BLOCK_SIZE) for _ in range(NUM_BLOCKS)]

for i in range(0, NUM_BLOCKS, 2):  # 释放偶数下标,制造棋盘式空洞(奇数下标的块仍然存活,挡在中间)
    alloc.free(ids[i])

frag = alloc.fragmentation_ratio()
total_free = sum(size for _, size in alloc.free_blocks)
largest_free = max(size for _, size in alloc.free_blocks)
print('total_free=%d largest_contiguous_free=%d fragmentation_ratio=%.3f' % (total_free, largest_free, frag))
assert total_free >= BLOCK_SIZE * (NUM_BLOCKS // 2) * 0.99, "roughly half the blocks were freed, so total free space should be close to half the allocated total"
assert largest_free <= BLOCK_SIZE * 2, \
    "the checkerboard freeing pattern means no two freed blocks are adjacent (the odd-indexed ones remain allocated in between) - the largest contiguous free block stays tiny no matter how much total free space exists"
assert frag > 0.9, "with huge total free space but only tiny contiguous chunks, the fragmentation ratio should be severe (close to 1.0)"
print("CHECKERBOARD_FRAGMENTATION_TEST=PASS")

big_request = alloc.alloc(BLOCK_SIZE * 3)
print('big_request_of_3_blocks_result=%s (total_free=%d is far more than 3 blocks worth, yet the request fails)' % (big_request, total_free))
assert big_request is None, "a request needing 3 contiguous blocks must fail even though total free space is far more than 3 blocks worth - this IS the real, practical cost of external fragmentation, not an abstract concern"
print("FRAGMENTATION_CAUSES_REAL_ALLOC_FAILURE_TEST=PASS")
```

验证记录:实测总空闲空间 45000 字节(相当于 450 个 block),但最大连续空闲块仅 100 字节(1 个 block),碎片化程度 0.998(接近极限);紧接着一个只需要 300 字节(3 个 block)的请求,尽管总空闲远超所需,依然分配失败——用具体数字直观展示了"总量够用"和"能不能用"完全是两回事。

**面试怎么问+追问链**

- **真实性验证轴**:"我们的服务运行几天后开始偶发内存分配失败,重启就好了"——追问:有没有验证过是"总内存不够"还是"碎片化导致的分配失败"?正确的排查方法包括:查看分配失败时刻的详细内存统计(不只是总空闲量,还要看空闲块的大小分布),如果总空闲量充足但请求持续失败,基本可以确认是碎片问题;"重启就好"这个现象本身也是碎片问题的典型指纹(重启清空了所有历史分配状态,碎片自然消失,直到下一轮长期运行重新累积)——如果是单纯内存不足,重启通常不能"根治"、很快会复现同样问题,两种故障模式的时间尺度特征不同。

**常见坑**

- 遇到"内存够但分配失败"的问题,第一反应是加内存——加内存确实能缓解(给碎片更多"腾挪空间"),但如果分配释放的模式本身没有改变(依然是不规律的棋盘式模式),碎片化问题迟早会在更大的内存池上重新累积到同样麻烦的程度,只是需要更长时间;真正的根治手段是改变分配策略本身(比如第 2 点的大小分级/slab思路,或者定期重启释放压力大的服务组件作为工程上可接受的权宜之计),单纯堆内存是治标不治本。

---

*本文件 7 个知识点,验证环境:全部 `.venv`(纯内存分配器/页面置换算法模拟 + 标准库 `mmap` 真实文件映射验证)。*
