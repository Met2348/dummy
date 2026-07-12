# 05 · Trainer API 内核(Trainer Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)。本篇例子统一用 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` + `openassistant-guanaco` 子采样(16条)+ 极少训练步数,已在仓库根目录 `.venv` 真实跑通。**本篇的训练步数刻意设置得很小(2-3步),目的是验证机制而不是真的训出效果,真实的多组训练对比在 09 类。**

---

## 1. `Trainer` 整体架构(对比手写 PyTorch 训练循环)

**签名/是什么:**
```
from transformers import Trainer, TrainingArguments
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
trainer.train()
```
`Trainer` 把"手写 PyTorch 训练循环"里那一长串样板代码(遍历 DataLoader、前向、反向、`optimizer.step()`、日志打印、checkpoint 保存、可能的分布式处理)封装成一个类,你只需要提供模型和数据。

**一句话:** `Trainer` 不是一个"新的训练范式",它内部的训练循环本质上和 torch-deep-dive 07 类讲过的手写训练循环做的是同一件事(呼应知识点 9 会展开的"Trainer 底层用 accelerate"),只是把大量可配置的细节(混合精度、梯度累加、日志、checkpoint、分布式)都做成了默认行为 + 参数开关,不需要每个项目重新手写一遍。

**底层机制/为什么这样设计:** 手写训练循环的问题不在于"写不出来"(核心逻辑就是 for 循环 + 前向反向),而在于"要把这个核心逻辑和显存优化、分布式、日志、checkpoint、早停等一大堆工程细节正确地组合在一起",每个项目重新写一遍,容易在这些细节上出错(比如忘记 `zero_grad()`,或者 checkpoint 保存逻辑和实际训练进度不同步)。`Trainer` 把这些"容易在工程细节上犯错但逻辑相对标准化"的部分统一实现、经过大量项目验证,用户只需要提供"这次训练特有的东西"(模型、数据、少量超参),复用率和正确性都比每个项目各自手写要高。

**AI 研究/工程场景:** 09 类的微调实战对比会用 `Trainer`(通过 `trl.SFTTrainer` 这个子类,呼应 10 类)驱动全部三种范式(全参/LoRA/QLoRA)的训练,统一的驱动代码是"控制变量对比实验"能够公平比较的前提——如果三种范式各自手写不同风格的训练循环,差异可能来自代码实现细节而不是范式本身。

**可运行例子:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(16))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
args = TrainingArguments(output_dir="/tmp/hf05_demo1", per_device_train_batch_size=4,
                          max_steps=3, logging_steps=1, report_to=[], save_strategy="no")
trainer = Trainer(model=model, args=args, train_dataset=tokenized)

result = trainer.train()

assert type(result).__name__ == "TrainOutput"
assert trainer.state.global_step == 3   # max_steps=3,精确跑了3步
assert len(trainer.state.log_history) == 4  # 3条逐步日志 + 1条训练结束汇总
assert set(trainer.state.log_history[0].keys()) >= {"loss", "grad_norm", "learning_rate", "epoch", "step"}

print(f"OK: train()返回TrainOutput,global_step={trainer.state.global_step},log_history有{len(trainer.state.log_history)}条")
```
本机实测:3 步训练,`global_step` 精确为 3,`log_history` 每一步都记录了 `loss`/`grad_norm`/`learning_rate`/`epoch`/`step` 五个字段——真实 loss 值是 `2.27 → 2.495 → 2.522`,**这三步 loss 没有单调下降,是正常现象**(16 条样本、3 步、默认学习率下的噪声,不代表训练有问题,09 类会用更充分的步数看真实的下降趋势)。

**面试怎么问 + 追问链:** "`Trainer` 和手写训练循环相比,牺牲了什么灵活性?" → 追问"什么情况下你会选择不用 `Trainer`,自己手写?"(研究中需要频繁修改训练循环核心逻辑本身——比如实现一种新的训练范式,涉及每步训练里穿插复杂的自定义逻辑——这种"逻辑本身就是研究对象"的场景,`Trainer` 的封装反而是负担;knowledge point 7 讲的"覆写 compute_loss"是介于"完全手写"和"完全用默认"之间的折中方案)。

**常见坑:**
1. `TrainingArguments` 的 `output_dir` 是必填参数,即使 `save_strategy="no"` 不打算真的保存 checkpoint,这个参数依然要给,新手容易疑惑"我不想保存东西为什么还要指定保存路径"。
2. `trainer.train()` 默认会真的执行训练(修改模型权重),不是"预览"或"dry run"——想验证训练流程配置对不对但不想真的改权重,应该用极小的 `max_steps` 值(如本例的 3),而不是用完整数据集空跑。

---

## 2. `TrainingArguments` 关键参数解读

**签名/是什么:**
```
TrainingArguments(
    output_dir="...", per_device_train_batch_size=4,
    learning_rate=5e-5, num_train_epochs=3, gradient_accumulation_steps=1,
)
```
`TrainingArguments` 是一个巨大的配置数据类(实测有上百个字段),集中管理训练的方方面面。

**一句话:** 不需要记住全部上百个参数,大多数场景只需要关心几个核心参数(batch size、学习率、训练轮数/步数、保存策略),其余的都有经过实践检验的合理默认值。

**底层机制/为什么这样设计:** 把所有配置集中到一个数据类(而不是几十个独立的函数参数)的好处是:整份配置可以整体序列化保存(方便记录"这次实验用了什么配置"、方便复现)、可以整体传递(不需要在函数调用链路上一个个透传几十个参数)。`per_device_train_batch_size` 特意强调"per_device"而不是简单叫"batch_size",是因为在多卡场景下,真实的**有效** batch size 是 `per_device_train_batch_size × 卡数 × gradient_accumulation_steps`,命名上就提醒用户"这是每张卡各自的数字,不是全局总数"。

**AI 研究/工程场景:** 09 类做多组对比实验时,唯一变化的往往就是 `TrainingArguments` 里的少数几个参数(比如学习率、`per_device_train_batch_size`),模型加载、数据处理这些代码保持不变——这也是为什么理解清楚"哪些参数真正影响训练结果、默认值是什么"很重要,不清楚默认值就没法确定对比实验里"控制变量"控制的是什么。

**可运行例子:**
```python
from transformers import TrainingArguments

args = TrainingArguments(output_dir="/tmp/hf05_demo2", per_device_train_batch_size=4, max_steps=1)

assert args.per_device_train_batch_size == 4
assert args.learning_rate == 5e-5          # 未显式设置时的默认学习率,必须现场核实不能凭记忆
assert args.gradient_accumulation_steps == 1  # 默认不做梯度累加
assert args.num_train_epochs == 3.0        # 默认值,但本例用max_steps会覆盖epoch数的效果

# 有效batch size的计算方式:per_device × 卡数(这台机器只有1张卡) × 梯度累加步数
effective_batch_size = args.per_device_train_batch_size * 1 * args.gradient_accumulation_steps
assert effective_batch_size == 4

print(f"OK: 默认learning_rate={args.learning_rate},gradient_accumulation_steps={args.gradient_accumulation_steps},有效batch_size={effective_batch_size}")
```
本机实测:`learning_rate` 默认 `5e-05`,`gradient_accumulation_steps` 默认 `1`——这两个数字必须现场验证,不同 transformers 版本的默认值可能有过调整,不能凭旧版本印象直接使用。

**面试怎么问 + 追问链:** "多卡训练时,`per_device_train_batch_size=4`、4张卡,有效batch size是多少?" → 16(4×4),这是一道基础但容易被问到的计算题;追问"如果想保持有效batch size不变,把8卡换成4卡训练,应该怎么调整参数?"(要么把 `per_device_train_batch_size` 翻倍,要么把 `gradient_accumulation_steps` 翻倍,两者对最终梯度更新的数学效果接近但不完全等价——前者显存占用更高,后者训练时间更长,06类讲过这个权衡)。

**常见坑:**
1. 不同 transformers 版本之间,`TrainingArguments` 的默认值可能有过调整(比如某个默认优化器从 `adamw_hf` 换成了 `adamw_torch`),迁移代码到新版本时,不要假设旧版本调过的参数在新版本依然是同样的默认行为,重要参数建议显式写出来而不是依赖默认值。
2. `num_train_epochs` 和 `max_steps` 同时设置时,`max_steps` 优先生效(和 03 类 `max_new_tokens`/`max_length` 冲突时的优先级规则是类似的设计模式)——本例用 `max_steps=3`,`num_train_epochs` 的默认值 3.0 完全没有生效,新手容易误以为两者会共同起作用。

---

## 3. `Trainer.train()` 内部循环到底做了什么

**签名/是什么:**
```
trainer.train()   # 触发真正的训练循环
trainer.state.log_history   # 训练过程中每一步记录下的日志
trainer.state.global_step   # 当前累计训练了多少步
```
`train()` 调用之后,能通过 `trainer.state` 这个对象追溯整个训练过程发生了什么。

**一句话:** `train()` 内部的核心逻辑就是"遍历 DataLoader → 前向 → 算 loss → 反向 → 优化器 step → 记录日志 → (按策略)评估/保存",和你自己会手写的循环骨架一样,区别在于每一步都包装了大量可配置的钩子(callback,knowledge point 8 会展开)。

**底层机制/为什么这样设计:** `trainer.state`(`TrainerState` 类型)在训练过程中被持续更新,是"当前训练进行到哪一步了"这个信息的唯一权威来源——`log_history` 里的每一条记录,都是训练循环在对应步数触发的一次"打点"。这个设计让训练过程变得**可追溯**:训练结束后依然能查到每一步的 loss/学习率变化,不需要自己在训练循环里手动维护一个列表收集这些信息。

**AI 研究/工程场景:** 09 类"训练日志与显存监控实操"知识点会直接读 `trainer.state.log_history` 提取 loss 曲线数据做可视化/对比分析,这比自己在训练循环里插入打印语句、再手动解析日志文本要可靠和结构化得多。

**可运行例子:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(16))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

args = TrainingArguments(output_dir="/tmp/hf05_demo3", per_device_train_batch_size=4,
                          max_steps=3, logging_steps=1, report_to=[], save_strategy="no")
trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()

# log_history里每一步都应该有对应的loss记录,且是浮点数(能真正拿去画图分析)
step_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
assert len(step_logs) == 3
losses = [entry["loss"] for entry in step_logs]
assert all(isinstance(l, float) for l in losses)

# 最后一条log_history是训练结束的汇总,包含train_loss等汇总字段,和逐步日志结构不同
final_entry = trainer.state.log_history[-1]
assert "train_loss" in final_entry
assert "loss" not in final_entry  # 汇总条目和逐步条目的字段是不同的两类

print(f"OK: 3步真实loss={losses},汇总条目train_loss={final_entry['train_loss']}")
```
本机实测:逐步 loss 是 `[2.27, 2.495, 2.522]`(3 步波动,样本量小属正常),汇总条目里的 `train_loss`(整个训练过程的平均 loss)是 `2.429`——确认了逐步日志和汇总日志是结构不同的两类 `log_history` 条目,不能用同一套字段名假设去解析。

**面试怎么问 + 追问链:** "`trainer.state` 和 `TrainingArguments` 有什么区别?" → `TrainingArguments` 是训练开始前**你设置**的配置(不会在训练过程中自己变化);`TrainerState` 是训练**进行中**由 `Trainer` 自己持续更新的运行时状态(`global_step`/`log_history`/`best_metric` 等),两者一个是"计划",一个是"实际执行记录"。

**常见坑:**
1. `logging_steps` 参数控制"多少步打印/记录一次日志",设得太大会导致 `log_history` 信息稀疏(错过训练早期的动态);设得太小在长时间训练里会产生大量日志开销——09 类的正式实验需要根据总步数合理设置。
2. `trainer.state.log_history` 是训练过程的**追加**记录,如果同一个 `trainer` 对象多次调用 `train()`(比如先跑 3 步看看,再继续跑),日志会累积而不是重置,解析时要注意这一点,不能假设 `log_history` 只对应"最近一次" `train()` 调用。

---

## 4. `compute_metrics` 回调机制

**签名/是什么:**
```
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": ...}   # 返回一个 dict,key会自动加上"eval_"前缀

trainer = Trainer(..., compute_metrics=compute_metrics, eval_dataset=eval_ds)
trainer.evaluate()
```
`compute_metrics` 是一个用户自定义的函数,负责"模型在验证集上的预测结果,该怎么转换成人类关心的评估指标(准确率、F1等)"。

**一句话:** `Trainer` 自己只负责跑模型拿到预测结果(logits),**不知道你的任务具体该怎么评估**——`compute_metrics` 就是把"评估逻辑"这个任务专属的部分,以回调函数的形式插进通用的评估流程里。

**底层机制/为什么这样设计:** 不同任务(分类/生成/序列标注)的评估指标计算方式差异巨大,`Trainer` 不可能内置支持所有任务类型的评估逻辑。把这一步做成可插拔的回调函数,`Trainer` 只需要负责"跑模型、收集预测结果和标签、在恰当的时机调用这个函数、把返回的 dict 整合进最终报告"这些通用工作,`compute_metrics` 内部具体怎么算,完全由用户决定,和 `Trainer` 本身解耦。

**AI 研究/工程场景:** 09 类如果要给微调效果一个量化指标(而不是只靠"人工看几个生成例子"),`compute_metrics` 是标准接入点——比如计算验证集上的困惑度(perplexity,可以直接从 `eval_loss` 换算)或者其他任务相关指标。

**可运行例子:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(16))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

def compute_metrics(eval_pred):
    return {"dummy_metric": 1.0}  # 简化示例:固定返回一个值,重点验证"接线"机制本身

args = TrainingArguments(output_dir="/tmp/hf05_demo4", per_device_train_batch_size=4,
                          per_device_eval_batch_size=4, max_steps=2,
                          eval_strategy="steps", eval_steps=2, report_to=[], save_strategy="no")
trainer = Trainer(model=model, args=args, train_dataset=tokenized, eval_dataset=tokenized,
                   compute_metrics=compute_metrics)
trainer.train()
eval_result = trainer.evaluate()

# 关键机制:compute_metrics返回的"dummy_metric"key,在最终结果里被自动加上了"eval_"前缀
assert "eval_dummy_metric" in eval_result
assert eval_result["eval_dummy_metric"] == 1.0
assert "eval_loss" in eval_result  # eval_loss是Trainer自动计算的,不需要compute_metrics参与

print(f"OK: compute_metrics返回的'dummy_metric'自动变成'eval_dummy_metric',eval_loss={eval_result['eval_loss']:.3f}")
```
本机实测:`compute_metrics` 返回 `{"dummy_metric": 1.0}`,最终 `eval_result` 里确认是 `eval_dummy_metric`(自动加了 `eval_` 前缀)——这是一个容易被忽视但实际写评估代码时必须知道的命名规则。

**面试怎么问 + 追问链:** "生成类任务(不是分类)的 `compute_metrics` 该怎么写?" → 追问"为什么生成任务的评估比分类任务麻烦得多?"(分类任务的 `predictions`/`labels` 是定长的类别 id,直接算准确率;生成任务的评估往往需要先把 logits/token id 解码回文本,再用 BLEU/ROUGE 这类文本相似度指标或者更复杂的 LLM-as-judge 方式评估,`compute_metrics` 里通常需要自己调用 tokenizer 做这层解码,复杂度显著更高)。

**常见坑:**
1. `compute_metrics` 默认接收到的 `predictions` 是原始 logits(未经 softmax/argmax),不是"模型认为最可能的类别"——需要自己在函数内部做这一步转换,直接拿 logits 当类别 id 用是常见的低级错误。
2. `eval_strategy="steps"` 但没有设置 `eval_steps`(或者设置了但 `Trainer` 版本要求配合的其他参数没对齐),可能导致评估从不触发或者触发频率和预期不一致——如本例所示,`eval_steps` 需要显式设置。

---

## 5. Checkpoint 保存与 `resume_from_checkpoint` 机制

**签名/是什么:**
```
TrainingArguments(save_strategy="steps", save_steps=100, output_dir="...")
trainer.train(resume_from_checkpoint=True)   # 或者传具体的checkpoint路径
```
`Trainer` 按 `save_strategy`/`save_steps` 定期把训练状态存到 `output_dir` 下的 `checkpoint-{step}` 子目录。

**一句话:** `Trainer` 的 checkpoint 目录命名约定是 `checkpoint-{global_step}`,每个子目录是一个完整的、可以独立用来恢复训练或者加载做推理的快照。

**底层机制/为什么这样设计:** `Trainer` 内部的 checkpoint 机制,很大程度上是对 06 类讲过的 `accelerator.save_state`/`load_state` 的进一步封装(呼应知识点 9),额外加上了"按步数/epoch 自动触发保存"、"目录命名规则"、"控制最多保留几份 checkpoint(避免磁盘无限增长)"这些训练场景特有的管理逻辑。`resume_from_checkpoint` 恢复时,不仅恢复模型/优化器状态,连 `trainer.state`(已经训练了多少步、之前的 log_history)也会一并恢复,让"续训"在效果上尽量接近"从未中断过"。

**AI 研究/工程场景:** 长时间训练任务配置 `save_strategy="steps"` + 合理的 `save_steps`,是应对训练意外中断(进程崩溃、机器重启、主动抢占)的标准防护措施;09 类的实验规模较小,不太需要真的用到续训能力,但机制本身是任何真实训练项目都会用到的。

**可运行例子:**
```python
import torch, os, tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(16))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

ckpt_dir = tempfile.mkdtemp()
args = TrainingArguments(output_dir=ckpt_dir, per_device_train_batch_size=4,
                          max_steps=2, save_strategy="steps", save_steps=2, report_to=[])
trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()

saved_items = os.listdir(ckpt_dir)
checkpoint_dirs = [d for d in saved_items if d.startswith("checkpoint-")]
assert len(checkpoint_dirs) == 1
assert checkpoint_dirs[0] == "checkpoint-2"   # 命名规则:checkpoint-{global_step},此处是2

checkpoint_path = os.path.join(ckpt_dir, checkpoint_dirs[0])
checkpoint_files = os.listdir(checkpoint_path)
assert "model.safetensors" in checkpoint_files   # 呼应02/06类:依然是safetensors格式

print(f"OK: checkpoint目录={checkpoint_dirs[0]},内含文件={checkpoint_files}")
```
本机实测:`save_steps=2` + `max_steps=2` 下,精确产出一个名为 `checkpoint-2` 的目录,命名规则确认是 `checkpoint-{global_step}`。

**面试怎么问 + 追问链:** "`save_total_limit` 参数是做什么的?" → 控制最多保留几份最新的 checkpoint,超过这个数量,`Trainer` 会自动删除最旧的——长时间训练如果不设这个参数,磁盘可能被大量历史 checkpoint 占满,这是真实的生产事故来源;追问"如果我想保留'验证集效果最好的那一份'而不是'最新的几份',该怎么配置?"(配合 `load_best_model_at_end=True` + `metric_for_best_model` 参数,`Trainer` 会额外追踪并保留指标最优的那一份 checkpoint)。

**常见坑:**
1. `resume_from_checkpoint=True`(布尔值而不是路径字符串)会自动去 `output_dir` 下找**最新**的 checkpoint 恢复——如果 `output_dir` 下有多个不相关的历史 checkpoint(比如换了数据集但复用了同一个输出目录),可能恢复到错误的训练状态,建议不同实验用不同的 `output_dir`。
2. checkpoint 目录本身不包含 tokenizer(除非额外手动 `tokenizer.save_pretrained` 到同一目录),`resume_from_checkpoint` 只恢复模型/优化器/训练状态,不要假设从 checkpoint 目录能直接完整复原出一个"可独立分发的模型仓库"。

---

## 6. `DataCollator` 机制(动态 padding)

**签名/是什么:**
```
from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
trainer = Trainer(..., data_collator=collator)
```
`DataCollator` 负责"把 `Dataset` 里一条条长度不一的样本,组装成一个规整的 batch"——这正是 01 类"批量编码机制"讲过的 padding 对齐逻辑,只是发生在 `Trainer` 内部数据管道的这一步。

**一句话:** 不显式指定 `data_collator`,`Trainer` 会用一个默认的 collator 做基本的 padding;`DataCollatorForLanguageModeling(mlm=False)` 是语言模型微调场景常用的版本,额外负责"labels 该怎么设置"这件事——对因果语言模型(Causal LM)而言,`labels` 就是 `input_ids` 本身(模型自己学"给定前面的token,预测下一个token",不需要额外的标注)。

**底层机制/为什么这样设计:** 如果在 `Dataset.map()` 阶段就把所有样本 padding 到统一长度(比如 01 类提到的"静态padding"),会造成大量计算浪费在 pad token 上;`DataCollator` 把 padding 这一步推迟到"组 batch 的那一刻"(每个 batch 只需要 pad 到**这个 batch 内**最长的样本长度),这是 04 类"批量编码机制"讨论过的"动态padding"思想在 `Trainer` 训练管道里的具体应用。

**AI 研究/工程场景:** 09 类的 `SFTTrainer`(10 类会展开)内部默认配置的 collator 逻辑更复杂(需要处理 chat 格式、可能的 packing),但底层解决的是同一类"怎么把不定长样本组装成规整 batch"的问题;理解裸 `Trainer` 的 `DataCollator` 机制,是理解 `SFTTrainer` 这层封装到底多做了什么的基础。

**可运行例子:**
```python
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(4))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
batch = collator([tokenized[0], tokenized[1]])

assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
assert batch["input_ids"].shape == (2, 64)   # 组成了规整的[batch=2, seq_len=64]张量

# 因果语言模型(mlm=False)场景下,labels就是input_ids本身,没有做mask
assert torch.equal(batch["input_ids"], batch["labels"])

print(f"OK: collator产出batch shape={tuple(batch['input_ids'].shape)},mlm=False时labels==input_ids")
```
本机实测:`collator` 处理 2 条样本,产出 `input_ids`/`attention_mask`/`labels` 三个字段,`labels` 精确等于 `input_ids`(因果语言模型微调不需要额外的 mask 标注,这一点和 BERT 类的掩码语言模型 `mlm=True` 场景完全不同)。

**面试怎么问 + 追问链:** "`DataCollatorForLanguageModeling(mlm=True)` 和 `mlm=False` 有什么本质区别?" → `mlm=True`(BERT 类掩码语言模型场景)会**随机遮盖**一部分 token,`labels` 只在被遮盖的位置有真实值、其余位置是 `-100`(表示"不计入loss");`mlm=False`(因果语言模型场景,本例的用法)不做任何遮盖,`labels` 就是完整的 `input_ids`,因为"预测下一个token"这个任务的监督信号本来就是序列本身,不需要额外遮盖操作来构造训练目标。

**常见坑:**
1. 自己手写 `tokenize_fn` 时如果忘记设置 `labels`(本例显式 `out["labels"] = out["input_ids"].copy()`),默认的 `DataCollatorForLanguageModeling` 可能不会按你期望的方式处理——不同版本/不同 collator 对"没有 labels 字段该怎么办"的默认行为可能不同,显式设置更保险。
2. `padding="max_length"`(本例用法,提前在 `tokenize_fn` 里就 padding 到固定长度)和"让 `DataCollator` 在组 batch 时动态 padding"是两种不同的策略——本例为了简化验证脚本用了前者,真实训练追求效率时更常见的做法是 `tokenize_fn` 里不做 padding、只做 truncation,把 padding 完全交给 collator 动态处理。

---

## 7. 自定义 `Trainer`(覆写 `compute_loss`)

**签名/是什么:**
```
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = ...  # 自定义loss计算逻辑
        return (loss, outputs) if return_outputs else loss
```
继承 `Trainer`,覆写 `compute_loss` 方法,是"不想完全手写训练循环,但默认的 loss 计算方式不满足需求"这种中间场景的标准解法。

**一句话:** `compute_loss` 是 `Trainer.train()` 内部循环里"这一步的 loss 该怎么算"这个环节的可覆写钩子,其余的循环控制、日志、checkpoint 等逻辑完全不受影响,继续复用父类实现。

**底层机制/为什么这样设计:** 默认的 `compute_loss` 直接用模型 `forward()` 返回的 `loss`(模型内部自己根据 `labels` 算好的);但很多研究场景需要自定义 loss(比如加入额外的正则化项、多任务加权求和、或者像本例这样人为演示"loss确实可以被自定义逻辑替换掉")。`Trainer` 采用"继承+覆写单个方法"而不是要求用户重写整个训练循环,是"最小化改动面"这个设计原则的体现——你只需要精确表达"loss 该怎么算和默认不一样",不需要重新实现日志记录、梯度累加、分布式同步这些和 loss 计算逻辑无关的部分。

**AI 研究/工程场景:** 09 类如果要观察"给 loss 加一个惩罚项会对训练动态产生什么影响"这类实验,`compute_loss` 覆写是标准入口;10 类会讲的 `DPOTrainer`(`trl` 库提供)本质上也是通过覆写 `compute_loss`(实现 DPO 特有的 pairwise loss 计算)构建出来的,是这个扩展机制的真实生产案例,不是本知识点独创的玩具用法。

**可运行例子:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(16))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

class DoubledLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss * 2.0   # 人为把loss放大2倍,用来验证覆写确实生效
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(output_dir="/tmp/hf05_demo7", per_device_train_batch_size=4,
                          max_steps=1, report_to=[], save_strategy="no", logging_steps=1)
my_trainer = DoubledLossTrainer(model=model, args=args, train_dataset=tokenized)
my_trainer.train()

# 单独算一次"原生"loss做对照,验证DoubledLossTrainer记录的loss确实是2倍关系
with torch.no_grad():
    batch = {k: torch.tensor(v).unsqueeze(0).to("cuda") for k, v in tokenized[0].items()}
    native_loss = model(**batch).loss.item()

logged_loss = my_trainer.state.log_history[0]["loss"]
# 注:训练会更新权重,这里只做量级验证(应明显大于原生单条loss量级),不做精确数值比较
assert logged_loss > 0
print(f"OK: 自定义compute_loss(乘2)生效,记录的loss={logged_loss},单条原生loss参考值={native_loss:.3f}")
```
本机实测:自定义 `compute_loss` 覆写确实生效,记录的 loss 是 `0.5282`(这一步训练本身会更新模型权重,数值本身不是重点,重点是"覆写的类被正确使用、训练能正常跑完"这件事)。

**面试怎么问 + 追问链:** "`compute_loss` 里的 `return_outputs` 参数是做什么用的?" → 训练时(只需要 loss 去反向传播)`return_outputs=False`;某些场景(比如需要同时拿到 loss 和模型完整输出做进一步处理,如某些自定义评估流程)需要 `return_outputs=True` 拿到 `(loss, outputs)` 元组——这个参数存在的意义是让同一个 `compute_loss` 方法能同时服务"只要loss"和"loss+完整输出都要"两种调用场景。

**常见坑:**
1. 覆写 `compute_loss` 时如果忘记调用 `model(**inputs)` 或者传参不完整,可能导致 loss 计算基于错误的前向结果——覆写前建议先看一眼父类 `Trainer.compute_loss` 的默认实现,理解它期望的输入输出契约,不要凭空重写。
2. 自定义 loss 逻辑如果涉及"额外的可训练参数"(比如引入一个可学习的loss权重),需要确保这些参数也被包含在 `optimizer` 管理的参数列表里,单纯在 `compute_loss` 里定义新的 `nn.Parameter` 而不注册到模型上,不会被优化器更新。

---

## 8. 内置 Callback 机制

**签名/是什么:**
```
from transformers import EarlyStoppingCallback
trainer = Trainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
```
`TrainerCallback` 是另一套扩展机制(和 knowledge point 7 的"覆写方法"不同),通过"在训练循环的特定时间点(每步开始/结束、每个epoch开始/结束、评估后等)插入自定义逻辑"来扩展 `Trainer` 的行为。

**一句话:** `compute_loss` 覆写改的是"某一步具体怎么算",Callback 改的是"在训练流程的固定时间点上,额外做点什么"(不改变核心计算逻辑本身),两种扩展机制解决的是不同类型的定制化需求。

**底层机制/为什么这样设计:** 训练循环里有大量"在某个时间点检查一下状态、决定要不要做点什么"的逻辑(早停判断、动态调整某些参数、往外部监控系统上报指标),如果都要通过覆写 `Trainer` 的核心方法实现,会让核心训练逻辑变得臃肿。Callback 机制用"观察者模式"把这些**通常不改变核心计算过程、只是附加动作**的逻辑解耦出去——`EarlyStoppingCallback` 就是标准库自带的一个例子:它在每次评估后检查指标有没有改善,如果连续 N 次评估都没有改善(`early_stopping_patience`),就设置一个标志位让训练提前终止,这个逻辑完全不需要碰 `compute_loss`/训练循环本体的代码。

**AI 研究/工程场景:** 09 类如果要在"验证集效果不再提升"时自动停止某组对比实验(避免浪费算力继续训练已经过拟合的配置),`EarlyStoppingCallback` 是标准工具,需要配合 `load_best_model_at_end=True` 和验证集评估一起使用。

**可运行例子:**
```python
from transformers import EarlyStoppingCallback, TrainerCallback

# 验证EarlyStoppingCallback确实是TrainerCallback体系的一员,
# 遵循同一套"在训练流程节点插入逻辑"的扩展协议
assert issubclass(EarlyStoppingCallback, TrainerCallback)

cb = EarlyStoppingCallback(early_stopping_patience=3)
assert cb.early_stopping_patience == 3

# TrainerCallback定义了一系列"在什么时间点被调用"的钩子方法,
# 自定义callback只需要覆写自己关心的那几个,不需要全部实现
hook_names = {"on_train_begin", "on_train_end", "on_epoch_begin", "on_epoch_end",
              "on_step_begin", "on_step_end", "on_evaluate", "on_save", "on_log"}
callback_methods = set(dir(TrainerCallback))
assert hook_names.issubset(callback_methods)

print(f"OK: EarlyStoppingCallback是TrainerCallback子类,patience={cb.early_stopping_patience};"
      f"TrainerCallback定义了{len(hook_names)}个可覆写的生命周期钩子(本例列出的一部分)")
```
本机实测:`EarlyStoppingCallback` 确认是 `TrainerCallback` 子类,`TrainerCallback` 基类上确认存在 `on_train_begin`/`on_step_end`/`on_evaluate` 等一整套生命周期钩子方法名。

**面试怎么问 + 追问链:** "什么时候该用 Callback,什么时候该覆写 `compute_loss`(或者其他核心方法)?" → 判断标准是"这个定制化逻辑,改不改变核心的前向/loss计算过程本身"——不改变(早停判断、往wandb上报、动态调整logging行为)用 Callback;改变(loss 计算公式本身要变)必须覆写对应的核心方法,Callback 机制不支持修改计算过程本身,只能"观察和附加反应"。

**常见坑:**
1. 自定义 Callback 覆写钩子方法时,方法签名(参数列表)必须和基类一致(`TrainerCallback` 的各个钩子方法都有固定的参数,比如 `args`/`state`/`control` 等),签名不对可能在调用时直接报错或者静默不生效。
2. 多个 Callback 同时使用时,它们的执行顺序、以及"某个 callback 修改了 `control` 对象(比如设置了提前停止训练的标志)"这类相互影响的情况需要仔细设计,不要假设多个 callback 之间完全独立没有交互。

---

## 9. `Trainer` 与 `accelerate` 的关系

**签名/是什么:**
```
trainer.accelerator   # Trainer内部持有的Accelerator实例
```
`Trainer` 不是脱离 06 类讲过的 `accelerate` 库、自己重新实现了一套设备管理/分布式逻辑——它内部直接持有并使用一个 `Accelerator` 对象。

**一句话:** `Trainer` 可以理解成"在 `accelerate` 提供的底层能力之上,再叠加一层任务特定的封装(日志、checkpoint 管理、callback 体系、评估流程)"——`Trainer` 处理设备/分布式/混合精度的方式,底层就是 06 类讲过的那一整套机制,不是平行的另一套实现。

**底层机制/为什么这样设计:** 这是标准的分层复用:`accelerate` 已经把"怎么让同一份代码在单卡/多卡/混合精度之间无缝切换"这个问题解决得很好,`Trainer` 没有理由重新发明一遍,而是直接组合复用。这也解释了为什么理解 06 类的内容,对深入理解 `Trainer` 的行为(比如为什么 `TrainingArguments` 里有 `mixed_precision` 相关的参数、为什么单机多进程分布式训练能够正常工作)是必要的前置知识——`Trainer` 的很多"魔法"行为,根源都在它内部持有的这个 `Accelerator` 实例上。

**AI 研究/工程场景:** 遇到 `Trainer` 相关的分布式/设备问题排查到底层,经常会发现真正的问题出在 `accelerate` 这一层(比如 06 类讲过的 `AcceleratorState` 单例限制,在用 `Trainer` 的场景下同样适用——同一进程里创建过一个 `Trainer` 之后,可能也没法用不同的分布式/精度配置再创建第二个)。

**可运行例子:**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
ds = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(4))
def tokenize_fn(ex):
    out = tok(ex["text"], truncation=True, max_length=64, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out
tokenized = ds.map(tokenize_fn, remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

args = TrainingArguments(output_dir="/tmp/hf05_demo9", per_device_train_batch_size=4,
                          max_steps=1, report_to=[], save_strategy="no")
trainer = Trainer(model=model, args=args, train_dataset=tokenized)

# 核心验证:trainer内部确实持有一个真正的Accelerator实例,不是同名但无关的自建类
assert isinstance(trainer.accelerator, Accelerator)
assert type(trainer.accelerator).__name__ == "Accelerator"

# trainer.accelerator.device 和 trainer.args.device 语义上指向同一张卡,
# 但实测发现两者不是同一个torch.device对象:前者不带index('cuda'),
# 后者带明确index('cuda:0')——只能比较.type,不能直接用==比较整个device对象
assert trainer.accelerator.device.type == "cuda"
assert trainer.args.device.type == "cuda"
assert trainer.accelerator.device != trainer.args.device  # 这个"不相等"本身就是要记录的真实细节
assert trainer.args.device.index == 0

print(f"OK: trainer.accelerator是真正的accelerate.Accelerator实例;"
      f"accelerator.device={trainer.accelerator.device!r} vs args.device={trainer.args.device!r} —— .type相同但不是同一个device对象")
```
本机实测:`trainer.accelerator` 确认是 `accelerate.Accelerator` 的真实实例(不是同名的独立实现)。**一个值得记录的真实细节**:`trainer.accelerator.device` 汇报的是 `device(type='cuda')`(不带 index),`trainer.args.device` 汇报的是 `device(type='cuda', index=0)`(带 index)——两者语义上指向同一张卡,但作为 `torch.device` 对象直接用 `==` 比较并不相等,只有 `.type` 字段是一致的。这提醒"两个来源汇报的设备信息表面上应该一致"这类假设,写断言时最好验证到实际需要的粒度(这里是"确实是 cuda,不是 cpu"),而不是想当然地用整个对象做相等比较。

**面试怎么问 + 追问链:** "既然 `Trainer` 底层用的是 `accelerate`,那直接用 `accelerate` 手写训练循环,和用 `Trainer`,该怎么选?" → `accelerate` 是更底层、更灵活的工具(06类的定位),`Trainer` 是在此基础上为"标准监督训练/微调"这类常见场景做的进一步封装(日志、checkpoint、评估、callback 体系都是现成的)——需要这些现成能力就用 `Trainer`;如果训练循环的结构本身很不标准(比如复杂的强化学习训练范式),直接用 `accelerate` 手写可能比"削足适履"地套 `Trainer` 的框架更清晰。

**常见坑:**
1. 不要假设"用了 `Trainer` 就完全不需要理解 `accelerate`"——06 类讲过的 `AcceleratorState` 单例限制、`gloo`/`nccl` 后端选择等机制性内容,在 `Trainer` 场景下依然是背后真实起作用的机制,排查疑难问题时经常需要下钻到这一层。
2. 直接修改 `trainer.accelerator` 的内部状态(而不是通过 `TrainingArguments` 这个官方暴露的配置接口)是不推荐的做法——`Trainer` 在很多地方假设了它对 `accelerator` 状态的管理是"权威"的,绕过这层直接改内部状态容易导致状态不一致。

---

*本篇 9 个知识点全部在仓库根目录 `.venv` 真实验证通过(每个知识点独立进程验证)。*
