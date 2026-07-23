# 01 · Tokenizer 核心机制(Tokenizer Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。本篇例子统一以 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 的 tokenizer 为对象,已在仓库根目录 `.venv` 真实跑通(`transformers==5.10.2`)。**本篇多处例子的输出含非 ASCII 字符(如 SentencePiece 的 `▁` 前缀),运行时请设置 `PYTHONUTF8=1`,否则中文 Windows 默认 GBK 控制台会把这些字符打印成乱码**(不是代码错误,是控制台编码问题,呼应 00-roadmap.md 的环境声明)。

---

## 1. `AutoTokenizer.from_pretrained` 内部机制

**签名/是什么:**
```
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```
`AutoTokenizer` 不是一个具体的 tokenizer 类,是一个**工厂**:传一个模型名/路径,它会自动判断该实例化哪个具体的 tokenizer 类,返回那个类的实例。

**一句话:** `from_pretrained` 做的事情是"下载/定位配置文件 → 读取 `tokenizer_config.json` 决定具体类 → 用该类去加载 vocab/merges/tokenizer.json → 拼出一个可用的 tokenizer 对象",不是一个黑盒魔法。

**底层机制/为什么这样设计:** 具体过程分四步:① 把 `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` 解析成 Hub 仓库,检查本地缓存(`~/.cache/huggingface/hub`)有没有,没有就下载 `tokenizer_config.json`/`tokenizer.json`/`special_tokens_map.json` 等文件;② 读 `tokenizer_config.json` 里的 `"tokenizer_class"` 字段(TinyLlama 这里是 `"LlamaTokenizer"`);③ `AutoTokenizer` 内部维护一张"模型类型 → tokenizer 类"的映射表,按这个字段名反射出具体的 Python 类;④ 调用该类自己的 `from_pretrained`,真正解析 vocab 数据构造出对象。这一整套"配置驱动分发"的设计,是为了让用户不用记住"LLaMA 系模型该用哪个 tokenizer 类"这种细节——**这也是为什么本篇后面几乎所有例子都用 `AutoTokenizer` 而不是具体类名**。

**AI 研究/工程场景:** 写训练/推理脚本时,模型名往往是配置文件里的一个字符串变量,如果要手动 `import` 对应的具体 tokenizer 类,换一个底座模型就要跟着改 import 语句;用 `AutoTokenizer.from_pretrained(cfg.model_name)`,换模型只改一个字符串,这也是 07/09 类"用同一套代码跑不同实验配置"的基础前提。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

assert type(tok).__name__ == "LlamaTokenizer"
assert tok.vocab_size == 32000
assert tok.name_or_path == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("OK: AutoTokenizer 分发到了 LlamaTokenizer,vocab_size=32000")
```
本机实测:`type(tok).__name__` 确实是 `'LlamaTokenizer'`(读的是 `tokenizer_config.json` 里的 `tokenizer_class` 字段反射出来的)。

**面试怎么问 + 追问链:** "`AutoTokenizer.from_pretrained` 内部做了什么?" → 追问"如果我传一个本地目录路径而不是 Hub 仓库名,行为有什么不同?"(答:跳过网络请求/缓存查找,直接读本地目录里的同名文件,其余步骤一样) → 再追问"`tokenizer_config.json` 缺失 `tokenizer_class` 字段会怎样?"(答:`AutoTokenizer` 会退化到根据 `config.json` 里的 `model_type` 字段做映射,这是二级兜底逻辑)。

**常见坑:**
1. 网络不通时 `from_pretrained` 会卡住重试很久才报错——生产环境建议显式设置超时或提前 `local_files_only=True` 排除网络因素排查。
2. 本地缓存损坏(下载中断)会导致奇怪的 JSON 解析报错,报错信息经常不会直接提示"缓存坏了",要学会往 `~/.cache/huggingface/hub` 里手动排查。

---

## 2. BPE 分词算法实操演示

**签名/是什么:**
```
tok.tokenize("unbelievability")  # -> ['▁un', 'bel', 'iev', 'ability']
```
`tokenize()` 返回的是**子词(subword)列表**,不是整词、也不是单字符——这就是 BPE(Byte-Pair Encoding,及其变体如 SentencePiece 使用的 unigram/BPE)分词的直接可见结果。

**一句话:** BPE 不是查一本"词到子词"的固定字典,而是把词汇表里最高频的字符/子词对,反复合并成更长的子词单元,直到达到目标词表大小;推理时用这份"合并规则"贪心地把新词切成已知子词的组合。

**底层机制/为什么这样设计:** 传统"整词切分"遇到没在词表里出现过的词(out-of-vocabulary,OOV)只能标 `<unk>`,信息全部丢失;纯字符级切分能覆盖任意词,但序列长度爆炸、语义单元太碎。BPE 是这两者的折中:训练阶段统计语料里所有相邻符号对的共现频率,每轮把频率最高的一对合并成新符号,重复数万次;最终词表里既有常见整词(如`"the"`)也有高频子词片段(如`"ing"`/`"tion"`),生僻词会被拆成几个子词的组合而不是变成 `<unk>`。TinyLlama 用的是 SentencePiece 风格,词首会加一个 `▁`(U+2581)前缀标记"这是一个新词的开始"(把空格也编码进 token 本身,而不是单独处理空格字符)。

**上面这段"统计频率 → 合并最高频对 → 重复"是文字描述,容易停留在抽象层面——下面用一个和 tokenizer 训练同源的最小例子把这套贪心算法真正跑一遍。注意这和下面"可运行例子"不是同一件事:下面的例子是拿 TinyLlama **已经训练好**的 tokenizer 做分词(推理阶段);这里是往前退一步,演示"分词规则本身是怎么被训出来的"(训练阶段):**

```python
import re
from collections import Counter

# 玩具语料:词 -> 出现频率(数字刻意选得很小,方便手工核对,不是真实语料)
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

# 第0步:每个词拆成"单字符序列 + 词尾标记 </w>"
# (</w> 用来区分"词尾的t"和"词中的t",否则不同词里的同一个字符会被无差别对待)
vocab = {" ".join(list(word)) + " </w>": freq for word, freq in corpus.items()}

def get_pair_counts(vocab):
    """统计当前词表里,每一种"相邻符号对"一共出现了多少次(按词频加权求和)。"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """把词表里所有出现这个相邻对的地方,合并成一个新符号。"""
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged = "".join(pair)
    return {pattern.sub(merged, word): freq for word, freq in vocab.items()}

merge_history = []
for _ in range(4):
    pairs = get_pair_counts(vocab)
    best_pair = max(pairs, key=pairs.get)        # 贪心:每轮只合并当前共现频率最高的那一对
    merge_history.append((best_pair, pairs[best_pair]))
    vocab = merge_vocab(best_pair, vocab)

assert merge_history[0] == (('e', 's'), 9)         # newest(freq6)和widest(freq3)都含"es",6+3=9,当前最高
assert merge_history[1] == (('es', 't'), 9)        # 上一轮刚合出的es,紧接着又和t合并
assert merge_history[2] == (('est', '</w>'), 9)    # est再和词尾标记合并
assert merge_history[3] == (('l', 'o'), 7)         # 前3轮都在合并newest/widest共有的部分,第4轮才轮到low/lower
print("OK: 4轮贪心合并顺序 =", merge_history)
```
实测(纯 Python 标准库,不依赖任何 tokenizer 库,CPU 瞬间跑完):4 轮合并顺序精确是 `[(('e','s'),9), (('es','t'),9), (('est','</w>'),9), (('l','o'),7)]`——**第一轮该合并谁,看的是整个语料的共现总数,不是单个词内部的频率**:`newest` 和 `widest` 各自只出现 6 次和 3 次,但两个词都含有"es"这个相邻对,`(e,s)` 的共现次数是两者相加的 `9`,比"看起来更直觉"的 `(l,o)`(只在 low/lower 里出现,7 次)更高,所以先被合并——这也是为什么 BPE 训练出来的子词,经常是"横跨多个不同词、但共享某段拼写"的片段,不是按单个词的直觉去猜的。

**这 4 轮合并,`newest` 这个词的符号序列是怎么一步步变短的(第4轮合并的是`l,o`,`newest`里没有这两个字符,不受影响,停在第3轮的结果):**
```
n e w e s t </w>          第0步:7个符号(6个字符+1个词尾标记)
n e w [es] t </w>         第1轮 (e,s)->es  :变成6个符号
n e w [est] </w>          第2轮 (es,t)->est :变成5个符号
n e w [est</w>]           第3轮 (est,</w>)->est</w> :变成4个符号
```
"BPE 把最高频的字符/子词对反复合并"这句话,具体到 `newest` 这一个词上,就是上面这 3 次"两个相邻符号变成一个新符号"的操作——合并次数越多,这个词被切分成的 token 就越少(越接近"整词"),这正是"常见片段被合并、生僻片段保留原样(拆得更碎)"这个机制的最小可验证例子。真实的 tokenizer 训练(比如 TinyLlama 用的 SentencePiece)在几十万字符的真实语料上重复这个"统计 → 合并"过程数万次,规模不同,但每一轮"找最高频相邻对、合并、重复"的算法逻辑和上面这个玩具版本完全一样。

**AI 研究/工程场景:** 理解 BPE 直接决定了你能不能读懂模型的"有效上下文长度"——同样 1000 个英文单词,不同 tokenizer 编码出的 token 数可能相差 30% 以上,这直接影响 `max_length` 该设多少、API 按 token 计费的账单、以及为什么生僻专业词汇(化学分子名、小众编程语言关键字)会被切得特别碎、模型理解起来更吃力。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tokens = tok.tokenize("unbelievability")
assert tokens == ['▁un', 'bel', 'iev', 'ability']
assert len(tokens) == 4  # 一个词被切成4个子词,不是1个也不是15个字符
assert tokens[0].startswith('▁')  # ▁ 前缀标记"新词开始"

# 常见词几乎不被拆分,生僻长词被拆得更碎
common = tok.tokenize("hello world")
rare = tok.tokenize("supercalifragilisticexpialidocious")
assert len(common) == 2       # "hello" "world" 各自一个token
assert len(rare) > len(common)  # 生僻长词切分出的子词数量明显更多
print(f"OK: 'unbelievability'->{tokens}, 生僻词切出{len(rare)}个子词")
```
本机实测:`unbelievability` → `['▁un', 'bel', 'iev', 'ability']`(4个子词),`supercalifragilisticexpialidocious` → 11 个子词。**运行本例的终端/脚本必须设置 `PYTHONUTF8=1`,否则打印 `▁` 会花屏(不影响 assert 本身,只影响你肉眼看到的输出)。**

**面试怎么问 + 追问链:** "BPE 和 WordPiece(BERT用)、Unigram(部分 T5/ALBERT用)有什么区别?" → 追问"为什么现在主流 LLM(LLaMA/GPT系)几乎都用 BPE 或它的变体,而不是纯字符级或纯词级?"(答:字符级序列太长训练/推理都慢,词级 OOV 问题在开放域文本上无法回避,BPE 是当前的工程最优解)→ 深挖"词表大小(vocab_size)怎么选?太大太小分别有什么代价?"(词表越大,罕见 token 的 embedding 训练不充分;词表越小,序列越长,计算量越大——这是一个真实的超参权衡)。

**常见坑:**
1. 中文/日文这类没有天然空格分词的语言,BPE 的"▁词首标记"语义会变得模糊,很多中文优化过的 tokenizer 会有专门处理,不能想当然套用英文的直觉。
2. 同一段文本,不同大小写、不同前导空格,可能切分结果完全不同(`"World"` 和 `" World"` 大概率不是同一组 token id)——写 prompt 拼接逻辑时空格位置很容易踩这个坑。

---

## 3. fast(Rust) vs slow(Python) tokenizer:历史设计与当前版本的真实现状

**签名/是什么:**
```
tok_fast = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tok_slow = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
```
`use_fast` 参数历史上用来在"Rust 实现的高速 tokenizer"和"纯 Python 实现"之间二选一。**这个知识点的价值恰好在于:实测发现当前版本(transformers 5.10.2)的真实行为,和大多数教程描述的历史行为已经不一样了。**

**一句话:** 历史上 `PreTrainedTokenizerFast`(基于 Rust `tokenizers` 库,速度快)和 `PreTrainedTokenizer`(纯 Python 实现,速度慢但依赖少)是两条并行的类继承体系;**实测在 5.10.2 版本,`AutoTokenizer` 对 Llama、BERT 这些主流模型,无论 `use_fast=True` 还是 `False`,返回的对象 `is_fast` 都是 `True`**——也就是说对这些常见模型,`use_fast=False` 在当前版本已经不能再产出一个真正的纯 Python 实现了。

**底层机制/为什么这样设计:** 实测确认基类层面区分依然存在——`PreTrainedTokenizer` 现在的真实实现是 `transformers.tokenization_python.PythonBackend`,`PreTrainedTokenizerFast` 是 `transformers.tokenization_utils_tokenizers.TokenizersBackend`,两者是不同的类(`PreTrainedTokenizer is PreTrainedTokenizerFast` 为 `False`)。但对 `LlamaTokenizer`/`BertTokenizer` 这些具体模型类而言,**它们的 `from_pretrained` 在当前版本内部统一走向了 `TokenizersBackend`**(`LlamaTokenizer` 的 MRO 里能看到 `TokenizersBackend`——MRO 即方法解析顺序,[torch-deep-dive/03](../torch-deep-dive/03-nn-module-internals.md) 已经讲过 Python 按什么顺序在多重继承的父类里依次查找属性/方法,这里直接复用不重复讲),`use_fast=False` 这个参数对它们已经名存实亡。这反映了 transformers 库的一个真实演进趋势:早期"每个模型手写一个纯 Python tokenizer + 一个对应的 Fast 版本"的双轨维护成本越来越高,库正在朝着"绝大多数模型统一走 Rust 后端,只保留极少数没有 Rust 实现的模型走 Python 后端"收敛。旧教程里"fast 比 slow 快 N 倍"的性能对比 demo,在这些主流模型上已经无法复现——不是 benchmark 写错了,是被对比的对象事实上已经是同一个东西。

**AI 研究/工程场景:** 写涉及 tokenizer 选择的代码时,不要再假设 `use_fast=False` 能拿到一个"更兼容但更慢"的降级选项去规避 Rust 后端的某个 bug——至少对主流模型家族,这条退路在新版本里可能已经不存在了,真遇到 Rust 后端的兼容性问题,该去 transformers/tokenizers 的 issue 追踪,而不是切 `use_fast=False` 掩盖问题。

**可运行例子:**
```python
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 基类层面依然是两个不同的类
assert PreTrainedTokenizer is not PreTrainedTokenizerFast
assert PreTrainedTokenizer.__module__ == "transformers.tokenization_python"
assert PreTrainedTokenizerFast.__module__ == "transformers.tokenization_utils_tokenizers"

# 但具体模型类在当前版本,use_fast=True/False 都落到同一个后端
tok_fast = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tok_slow = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
assert tok_fast.is_fast is True
assert tok_slow.is_fast is True   # 关键断言:False 请求并没有拿到真正的slow实现
assert type(tok_fast).__name__ == type(tok_slow).__name__ == "LlamaTokenizer"

# 两者分词结果自然完全一致(本来就是同一套后端)
ids_fast = tok_fast("hello world")["input_ids"]
ids_slow = tok_slow("hello world")["input_ids"]
assert ids_fast == ids_slow
print("OK: use_fast=False 在当前版本对 LlamaTokenizer 已不产生真正的慢速实现")
```
本机实测:`LlamaTokenizerFast` 这个类在当前版本**已经不存在**了(`from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast` 直接 `ModuleNotFoundError`),`sentencepiece` 包本身是装了的(0.2.1),排除了"因为缺依赖被迫降级"这个可能性,这是库自身的架构收敛,不是本机环境缺东西。

**面试怎么问 + 追问链:** "fast 和 slow tokenizer 有什么区别,该怎么选?" 这是一道经典老题,**扎实的回答应该先讲历史设计动机(Rust 并行处理+零拷贝,能给长文本批量编码带来数量级的速度提升),再补一句"但在当前版本的很多主流模型上,这个选择已经被库收敛掉了,`use_fast=False` 不一定还能拿到真正的纯 Python 路径,建议实际验证 `.is_fast` 属性而不是想当然"**——这种"讲清楚历史 + 验证过当前真实行为"的回答,比死背一个可能已经过时的性能对比数字更能体现真实的工程判断力。

**常见坑:**
1. 网上大量教程(包括很多正式书籍)展示"fast 比 slow 快 10 倍"的 benchmark,复现时如果用的是主流模型+新版本 transformers,可能压根量不出这个差距——这不代表教程当年是错的,是库演进了,遇到这种"和资料对不上"的情况,第一反应应该是查当前版本的真实行为,而不是怀疑自己代码写错了。
2. 判断"这个 tokenizer 是不是 fast 版本"永远用 `tok.is_fast` 这个属性现场查,不要用 `use_fast=` 参数猜或者用类名里有没有"Fast"字样猜(如本例所示,`LlamaTokenizer` 这个名字里没有 Fast,但 `is_fast` 是 `True`)。

---

## 4. Special Tokens 机制

**签名/是什么:**
```
tok.special_tokens_map   # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'}
tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id
```
Special tokens 是词表里承担"结构性角色"而不是"语义"的特殊符号:句子开始(BOS)、句子结束(EOS)、未知词(UNK)、填充(PAD)等。

**一句话:** 这些符号具体是哪个字符串、对应哪个 id,**因模型而异**,必须现场读该模型的 `special_tokens_map`,不能凭"BOS 一般是 `<s>`"这种经验硬编码。

**底层机制/为什么这样设计:** 模型训练时,BOS/EOS 这些符号被当成普通 token 一起参与了 embedding 训练,模型学到了"看到 EOS 就该停止生成"这种关联——如果推理时用错了 token 字符串(比如手滑输错拼写),tokenizer 会把它当成未登录的普通文本重新拆分成好几个不相关的子词,模型完全学不到"这是停止信号"的含义,生成会失控停不下来。**Special tokens 是模型和 tokenizer 之间的一份隐式契约,契约的内容必须以模型自己声明的为准**,这也是为什么 `special_tokens_map.json` 要和模型权重一起打包发布。

**AI 研究/工程场景:** 手写生成循环判断"是否该停止"时,必须用 `tok.eos_token_id` 现场取值去比较 `next_token_id == tok.eos_token_id`,而不是硬编码某个整数;换一个底座模型,这个 id 大概率会变,硬编码的代码会静默出错(生成不停或过早停止)而不是报错提醒你。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

assert tok.bos_token == "<s>" and tok.bos_token_id == 1
assert tok.eos_token == "</s>" and tok.eos_token_id == 2
assert tok.unk_token == "<unk>" and tok.unk_token_id == 0
# TinyLlama-Chat 这个模型的 pad_token 默认就复用了 eos_token,不是 None
assert tok.pad_token == "</s>" and tok.pad_token_id == 2
assert tok.pad_token_id == tok.eos_token_id  # 两者实际是同一个token,这是本模型的真实设计

assert set(tok.all_special_tokens) == {"<s>", "</s>", "<unk>"}
print("OK: special tokens 现场核实,pad复用了eos,不是None")
```
本机实测:很多教程会提醒"LLaMA 系 tokenizer 默认没有 `pad_token`,要手动设置",但 **TinyLlama-Chat 这个具体 checkpoint 实测已经预置了 `pad_token='</s>'`**(复用 eos)——这正说明"要现场查,不要凭经验假设"这条原则本身,同一句"LLaMA 没有 pad_token"的经验放到不同的具体 checkpoint 上可能就不成立。

**面试怎么问 + 追问链:** "为什么很多 LLaMA 系模型要手动把 `pad_token` 设成 `eos_token`?会有什么副作用?" → 这是非常高频的面试题,答案:因为基座 LLaMA 原始词表没有专门的 pad 符号,一种常见 workaround 是复用 eos;副作用是**如果训练时不小心把 pad 位置的 loss 也算进去,模型会被"教会"提前生成 eos**(因为 pad 位置的 label 恰好也是 eos 的 id),必须配合正确的 `attention_mask`/`labels=-100` 屏蔽逻辑,这是一个真实的、容易致命的坑,09 类微调实战会再次遇到。

**常见坑:**
1. 不要假设所有模型的 special tokens 字符串都一样,`<s>`/`<|endoftext|>`/`<|im_end|>` 在不同模型家族里各不相同,复制别的模型的 prompt 模板过来不改 special tokens 是经典错误。
2. `pad_token=eos_token` 这个常见 workaround,如果训练代码没有正确屏蔽 pad 位置的 loss,会产生前面提到的"模型学会提前停止"的隐蔽 bug,现象是模型输出越来越短、越来越敷衍,排查起来容易走弯路。

---

## 5. Padding 策略与 `padding_side`

**签名/是什么:**
```
tok.padding_side = "left"   # 或 "right"
batch = tok(["hi", "hello there friend"], padding=True, return_tensors="pt")
```
一个 batch 里多条文本长度不一致时,`padding=True` 会把短的那条用 pad token 补齐到和最长的一样长;`padding_side` 决定补在左边还是右边。

**一句话:** **做生成(generation)任务几乎总是要用左填充(`padding_side="left"`),做分类/编码这类"整句一次性喂进去"的任务通常用右填充(默认值)**——选错方向不会报错,但生成结果会是错的。

**底层机制/为什么这样设计:** 自回归生成时,模型是"看最后一个位置的输出,预测下一个 token"。如果用右填充,batch 里较短的那条序列,它的"最后一个真实 token"不在整个序列的最后一个位置(后面跟着几个 pad),模型在"整个序列的最后一个位置"上算出来的其实是基于 pad token 的续写,不是你想要的续写——除非你手动对齐每条序列各自的真实结尾位置,非常麻烦还容易出错。左填充把 pad 都堆在前面,不管这条序列本来多长,"整个序列的最后一个位置"永远是它自己的真实最后一个 token,直接从这个位置续写就是对的。

**AI 研究/工程场景:** 批量跑评测(比如同时给模型 100 道题,等它逐条生成答案)几乎总是"生成任务",这个坑最容易在写评测脚本时踩到——12 类"批量推理的padding与attention_mask处理"知识点会用真实生成结果的质量差异(而不只是理论描述)进一步验证这一点;09 类微调实战对比里所有涉及批量生成的环节,也统一遵守这条规则。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tok.padding_side = "left"
batch_left = tok(["hi", "hello there friend"], padding=True, return_tensors="pt")
# "hi" 编码后比 "hello there friend" 短,pad(id=2)应该堆在左边
assert batch_left["input_ids"][0].tolist() == [2, 2, 1, 7251]
assert batch_left["attention_mask"][0].tolist() == [0, 0, 1, 1]
# 关键性质:不管padding_side,每一行"最后一个位置"应该是这行的真实最后一个token
assert batch_left["input_ids"][0][-1].item() == 7251  # "hi" 的真实内容在最后一位

tok.padding_side = "right"
batch_right = tok(["hi", "hello there friend"], padding=True, return_tensors="pt")
assert batch_right["input_ids"][0].tolist() == [1, 7251, 2, 2]
assert batch_right["input_ids"][0][-1].item() == 2  # 右填充下,"hi"这行最后一位是pad,不是真实内容!

print("OK: 左填充保证每行最后一位是真实token,右填充不保证")
```
本机实测:两种 padding_side 下 `input_ids`/`attention_mask` 的真实数值如上,和预期完全一致。

**面试怎么问 + 追问链:** "batch 生成时为什么要用左填充?" → 追问"那 `attention_mask` 在这里起了什么作用,如果我忘记传会怎样?"(答:pad 位置本身也会参与 self-attention 计算,`attention_mask=0` 的作用是告诉模型"这些位置的 key/value 在算 attention 分数时要被屏蔽掉"——忘记传等于让模型把无意义的 pad 也当成上下文的一部分,输出质量会下降,序列越短、batch 里长度差异越大,影响越明显)。

**常见坑:**
1. 用 `Trainer`/`SFTTrainer` **训练**时,`padding_side` 通常应该保持默认的右填充(训练是整句一次性算 loss,不存在"从哪个位置续写"的问题);只有做**生成/推理**才需要切成左填充——很多人把这两个场景的设置搞反,或者全局只设置一次就忘了区分。
2. 切换 `padding_side` 是这个 tokenizer 对象的一个可变属性,如果代码里有多处共享同一个 tokenizer 实例、又在不同地方分别设置过,容易出现"以为设的是 left,实际因为后面某处代码又改回了 right"的隐蔽 bug。

---

## 6. Truncation 策略与 `max_length`

**签名/是什么:**
```
tok(text, truncation=True, max_length=10)
```
当输入文本编码后的 token 数超过 `max_length`,`truncation=True` 会把超出部分**直接砍掉**,而不是报错。

**一句话:** `truncation` 是"防止模型收到超过它上下文窗口长度的输入而崩溃或产生未定义行为"的安全阀,默认从序列尾部砍(也可以配置从头部砍或者砍中间保留头尾)。

**底层机制/为什么这样设计:** 模型的位置编码(不管是绝对位置还是 RoPE 这类相对位置编码)通常是在训练时的固定/主要长度范围内学习的,超出这个范围模型的行为在训练分布之外,效果通常会明显变差甚至输出乱码。`truncation=True` + `max_length` 是在数据进入模型前就做好长度控制,防止把这种"分布外"的风险传给模型本身;不设置的话,遇到一条异常长的输入,轻则显存爆炸,重则模型输出质量骤降却没有任何报错提示你原因。

**AI 研究/工程场景:** 处理真实世界的长文档(论文、网页、日志)时几乎必须配合 `truncation=True` 使用,否则单条超长样本就可能让整个 batch 的显存需求失控;09 类微调时数据集里长短不一的样本,统一 `max_length` 截断也是控制训练显存可预测性的关键一步。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

long_text = "word " * 100  # 明显超长的重复文本

no_trunc = tok(long_text)["input_ids"]
trunc = tok(long_text, truncation=True, max_length=10)["input_ids"]

assert len(no_trunc) > 10       # 不截断,长度远超10
assert len(trunc) == 10         # 截断后严格等于max_length
assert trunc == no_trunc[:10]   # 默认截断策略:从尾部砍,保留开头部分
print(f"OK: 不截断{len(no_trunc)}个token,截断到{len(trunc)}个,内容是原序列的前10个token")
```
本机实测:未截断 102 个 token(100 个"word "重复+首尾特殊 token),截断后精确为 10。

**面试怎么问 + 追问链:** "`max_length` 该怎么设置?" → 追问"截断策略除了默认从尾部砍,还有哪些选择,分别适合什么场景?"(`truncation_side="left"` 从头部砍——比如聊天记录场景,通常更想保留"最近"的对话而不是最早的;还有一些任务会做"掐头去尾保留中间"或者"分段处理"的自定义策略,标准 API 不直接支持,需要自己写预处理逻辑)。

**常见坑:**
1. `max_length` 不设置时,`truncation=True` 会用模型的 `model_max_length`(如果 tokenizer 知道这个值)兜底,但有些 tokenizer 这个值配置得不准确(比如被错误设成一个极大的默认值),容易造成"以为设置了截断,实际上没有真正生效"的误解——生产代码里 `max_length` 建议显式传值,不要依赖隐式默认。
2. 截断是无损信息丢弃,不是"智能摘要"——超出部分的信息对模型来说是完全不存在的,不要把截断和摘要/压缩这类保留语义的操作混为一谈。

---

## 7. 批量编码机制

**签名/是什么:**
```
batch = tok(["short", "a bit longer sentence"], padding=True, return_tensors="pt")
# batch.keys() == ['input_ids', 'attention_mask']
```
`tokenizer.__call__` 传入一个字符串列表(而不是单个字符串),会返回一个"批量"的编码结果——所有字段都是二维张量(`[batch_size, seq_len]`),而不是一维。

**一句话:** 批量编码不是简单地把每条文本单独编码后再拼起来,而是内部先分别 tokenize、再统一按 batch 里最长的那条做 padding 对齐,最终打包成规整的矩形张量,这样才能喂进模型做并行计算。

**底层机制/为什么这样设计:** GPU 的并行计算优势建立在"同一个 batch 内所有样本形状一致"这个前提上——如果每条文本长度不同,没法直接堆叠成一个张量。批量编码把"对齐"这一步封装掉了:你不需要自己算 batch 里最长的序列是多少、手动补多少 pad,`padding=True` 会自动算出这个 batch 的最大长度并统一对齐。`attention_mask` 则是"这个对齐动作产生的副作用需要被记录下来"——模型需要知道哪些位置是真实内容、哪些是凑数补的 pad。

**AI 研究/工程场景:** `DataLoader` 配合 `DataCollator`(05 类会展开)本质上就是在反复调用这套批量编码机制,把不同长度的样本动态拼成规整 batch;理解这一层,才能理解为什么"动态 padding"(每个 batch 按自己内部最长的对齐,而不是全局统一填到数据集最大长度)能显著减少训练中浪费在 pad token 上的无效计算。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

single = tok("short")
assert isinstance(single["input_ids"], list)      # 单条:list[int]
assert isinstance(single["input_ids"][0], int)

batch = tok(["short", "a bit longer sentence"], padding=True, return_tensors="pt")
assert set(batch.keys()) == {"input_ids", "attention_mask"}
assert batch["input_ids"].dim() == 2               # 批量:2维张量
assert batch["input_ids"].shape[0] == 2            # batch_size=2
assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]  # 两个字段seq_len一致
# 两条长度不同的文本,padding后seq_len应该都等于batch内最长那条的长度
longer_len = len(tok("a bit longer sentence")["input_ids"])
assert batch["input_ids"].shape[1] == longer_len

print(f"OK: 单条返回list,批量返回shape={tuple(batch['input_ids'].shape)}的2维张量")
```
本机实测:两条文本 padding 对齐后 `input_ids`/`attention_mask` 的 shape 都是 `[2, 5]`。

**面试怎么问 + 追问链:** "为什么批量编码要统一按 batch 内最长对齐,而不是全局固定长度?" → 追问"动态 padding 和静态 padding(全部统一填到 `max_length`)各自的取舍是什么?"(动态 padding 省计算但每个 batch 形状不同、某些追求极致吞吐的固定 shape 优化场景不友好;静态 padding 计算量固定但会在短样本上浪费大量算力在 pad 上——这是一个真实的工程权衡,`DataCollatorWithPadding` 默认是动态的)。

**常见坑:**
1. 不传 `return_tensors="pt"` 时返回的是嵌套 Python list,不是 tensor,直接喂给 PyTorch 模型会报类型错误——这个参数经常被新手漏掉。
2. `padding=True` 只保证同一个 `__call__` 调用内部的 batch 互相对齐,不同批次之间的 `seq_len` 可能不一样(这是正常的,动态 padding 本来就是逐 batch 独立计算的)。

---

## 8. Chat Template 机制

**签名/是什么:**
```
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
`apply_chat_template` 把"结构化的对话消息列表"渲染成"这个具体模型训练时见过的那种纯文本格式"。

**一句话:** 不同模型训练时用的对话格式(角色标记怎么写、轮次之间怎么分隔)完全不同,`apply_chat_template` 把"格式差异"这件事从用户代码里剥离出来,你只需要给结构化的 `role`/`content` 列表,格式细节由模型自带的 Jinja2 模板决定。

**底层机制/为什么这样设计:** 每个 chat 模型在 `tokenizer_config.json` 里都存了一段 `chat_template`(Jinja2 模板字符串),`apply_chat_template` 本质上是拿 `messages` 这个数据结构去渲染这段模板。这样设计的意义在于:如果没有这层抽象,想换一个底座模型,你的 prompt 拼接代码(手写字符串拼接 `f"<|user|>\n{content}"` 这种)要跟着重写;有了这层抽象,只要模型自带了正确的 `chat_template`,换模型时你的业务代码(构造 `messages` 列表的逻辑)完全不用动。

**AI 研究/工程场景:** 09 类微调用的 `openassistant-guanaco` 数据集因为本身已经是纯文本格式,不需要这一步;但如果换成结构化的对话数据集(比如后面会提到的 `no_robots`,`messages` 字段是结构化的),就必须先 `apply_chat_template` 转成纯文本才能喂给 `SFTTrainer`,这是 10 类会展开的内容。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

assert rendered == (
    "<|system|>\nYou are a helpful assistant.</s>\n"
    "<|user|>\nWhat is 2+2?</s>\n"
    "<|assistant|>\n"
)
# add_generation_prompt=True 会在结尾追加"该轮到assistant说话了"的标记,方便直接喂给generate()
assert rendered.endswith("<|assistant|>\n")

# tokenize=True(默认)不是返回裸token id列表,而是返回 BatchEncoding(含input_ids+attention_mask)
from transformers.tokenization_utils_base import BatchEncoding
enc = tok.apply_chat_template(messages, add_generation_prompt=True)
assert isinstance(enc, BatchEncoding)
assert set(enc.keys()) == {"input_ids", "attention_mask"}
assert isinstance(enc["input_ids"], list) and isinstance(enc["input_ids"][0], int)

# 配合 return_tensors="pt" 可以直接产出能喂给 model.generate(**enc) 的批量张量,免去手动再调一次 tokenizer
enc_pt = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
assert enc_pt["input_ids"].dim() == 2  # 已经是 [batch, seq_len] 的张量,可以直接解包传给 generate()
print("OK: chat template 渲染出 TinyLlama 自己的 <|role|> 格式(Zephyr风格),tokenize=True默认返回BatchEncoding而非裸list")
```
本机实测:TinyLlama-Chat 用的是类 Zephyr 格式(`<|system|>`/`<|user|>`/`<|assistant|>` + `</s>` 分隔),渲染结果和断言完全一致。

**面试怎么问 + 追问链:** "`add_generation_prompt=True` 是做什么用的?" → 追问"如果模型没有内置 `chat_template` 会怎样?"(会抛 `ValueError`,提示你没有 chat template 可用——这种情况下要么这个模型本来就不是对话模型,要么需要手动指定一个模板字符串)→ 深挖"训练(09/10类)和推理阶段,`add_generation_prompt` 应该分别怎么设?"(训练时通常不需要这个尾巴,因为完整的对话(含assistant回复)本身就是训练目标;推理/生成时需要,用来提示模型"轮到你了")。

**常见坑:**
1. 手写字符串拼接 prompt(不用 `apply_chat_template`)是新手最容易踩的坑——一旦拼接格式和模型训练时见过的哪怕有一个换行符/空格的差异,模型的对话能力都会明显下降,且不会有任何报错提示你格式错了。
2. 不同模型的 `chat_template` 天差地别,把 A 模型的 prompt 模板字符串抄给 B 模型用是常见错误——永远用 `apply_chat_template` 让 tokenizer 自己处理,不要手抄别的模型的格式。

---

## 9. Tokenizer 保存与自定义词表增量添加

**签名/是什么:**
```
tok.add_tokens(["<MY_CUSTOM_TOKEN>"])
model.resize_token_embeddings(len(tok))
```
`add_tokens` 往词表里追加新符号;但**光加 tokenizer 的词表是不够的**,必须同步调用模型的 `resize_token_embeddings`,否则 tokenizer 能编出新 id、但模型的 embedding 矩阵根本没有这一行,会直接越界报错。

**一句话:** Tokenizer 的词表和模型的 embedding 矩阵是两个独立存储的东西,靠"行数必须一致"这个隐式契约绑在一起,新增 token 必须两边同步更新,只改一边会导致索引越界。

**底层机制/为什么这样设计:** 模型的 embedding 层本质是一个 `[vocab_size, hidden_size]` 的查找表矩阵,`input_ids` 里的每个整数就是这个矩阵的行索引。`add_tokens` 只是往 tokenizer 自己维护的"字符串 → id"映射表里插入新条目,分配的新 id 通常紧接在原 `vocab_size` 后面(本例中 `32000`);但模型的 embedding 矩阵行数还是原来的 `vocab_size`,没人告诉它"词表变大了"。`resize_token_embeddings(new_size)` 做的事情就是把 embedding 矩阵(以及最后输出层的 lm_head,如果没有 tie weights 的话——tie weights 即权重共享/权重绑定,让 embedding 层和输出层的 `.weight` 指向同一个 `nn.Parameter` 对象,[torch-deep-dive/03](../torch-deep-dive/03-nn-module-internals.md) 第10点已经讲过这个机制,这里不重复)扩容到新的行数,新增的行用随机初始化或者某种启发式方式填充——**这也是为什么新增 special token 之后,模型对这个新 token 的理解是从随机初始化开始学的,不是自带语义,需要用一定量的微调数据"教会"模型这个新符号该怎么用**。

**AI 研究/工程场景:** 给基座模型新增领域专属的特殊标记(比如给代码模型加 `<FIM_HOLE>` 这类 fill-in-the-middle 标记,或者给对话模型加自定义的工具调用标记)是真实的工程需求,这个"tokenizer 加词表 + 模型 resize embedding + 用数据微调让模型学会新 token 的含义"三部曲缺一不可。

**可运行例子:**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

before = len(tok)
assert before == 32000

num_added = tok.add_tokens(["<MY_CUSTOM_TOKEN>"])
assert num_added == 1
assert len(tok) == 32001

new_id = tok.convert_tokens_to_ids("<MY_CUSTOM_TOKEN>")
assert new_id == 32000  # 新token紧接在原vocab_size后面分配id

# 重复添加同一个token,不会重复计数(去重逻辑)
num_added_again = tok.add_tokens(["<MY_CUSTOM_TOKEN>"])
assert num_added_again == 0
assert len(tok) == 32001  # 长度没有变化

print(f"OK: 新增token前vocab_size={before},新增后={len(tok)},新token id={new_id}")
```
本机实测:新增前 32000,新增后 32001,新 token id 恰好是 32000(数组下一个可用行索引)。**本例没有加载模型演示 `resize_token_embeddings`,是刻意的取舍**——加载 1.1B 参数模型只为演示这一行 API 成本太高,读者理解"必须同步 resize,否则行数不匹配"这个原理后,`model.resize_token_embeddings(len(tok))` 是一行足够直白的 API 调用,02 类会在真实加载模型的场景里配合演示。

**面试怎么问 + 追问链:** "新增 token 之后不 resize 模型 embedding 会发生什么?" → 会在 forward 时的 embedding lookup 那一步直接 `IndexError`(索引越界),因为 tokenizer 编出了 32000 这个 id,但模型 embedding 矩阵只有 32000 行(合法索引 0~31999)。→ 追问"新 token 的初始 embedding 是怎么来的,会不会影响模型原有能力?"(默认随机初始化或复制某个已有 token 的向量做启发式初始化,如果不做任何微调直接使用,这个新 token 的表现是未定义的乱码级别;原有 token 的 embedding 不受影响,模型原有能力理论上不会因为纯粹"扩容"而退化,除非后续微调过程中学习率设置不当造成灾难性遗忘,这个话题在 09 类会真实观察)。

**常见坑:**
1. 忘记 `resize_token_embeddings` 是新手接入自定义 token 最常见的报错来源,报错信息是底层的 CUDA 索引越界或者 `IndexError`,不会直接提示你"是不是忘记 resize 了",第一次遇到容易一头雾水。
2. `add_tokens` 添加的是"普通词表 token",如果是想加"结构性"的 special token(比如自定义的角色标记,希望它不被当作普通文本处理),应该用 `add_special_tokens` 而不是 `add_tokens`,两者语义不同,08/10 类如果涉及自定义标记时需要留意选对方法。

---

*本篇 9 个知识点全部在仓库根目录 `.venv` 真实验证通过。*
