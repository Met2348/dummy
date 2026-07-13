# LLM 推理服务深挖 —— 路线图与进度表

> 目标:约 65-80 个 LLM 推理服务(Inference & Serving)知识点,由浅入深,深度对标 [torch-deep-dive/](../torch-deep-dive/00-roadmap.md)/[peft-deep-dive/](../peft-deep-dive/00-roadmap.md)(面试二三四面深度,不是"这个函数怎么调")。
> 定位:仓库"专题精读系列"第 5 条,直接对应 `learning/` 里一组结构完整的课程模块——学长自己称之为"Module 5《用大模型》"的 7 个专题:`inference-engine-core`/`sglang-radixattention`/`speculative-decoding`/`quantization-deploy`/`distributed-inference`/`production-serving`/`serving-graduation`。

## 和 `learning/` Module 5《用大模型》7 个专题的关系(差异化声明,必须先读)

**规模取舍(必须先说清楚)**:7 个源模块合计约 60 篇 lecture、7000+ 行 `src` 代码,不是逐 lecture 展开成 60+ 个孤立小节——本系列按"一个可独立验证的机制/设计"为颗粒度合并同类项(比如 `inference-engine-core` 里 CUDA Graphs 和 Attention Backends 两篇概念性 lecture 合并成 1 个知识点),目标是 65-80 个知识点,不是七八十篇罗列。`serving-graduation` 一个源模块因为自身性质分裂成两个 for_real_dummy 文件(07 常规知识点 + 08 capstone 叙事体),这是源材料自己"L01-L11 常规专题 + L12-L14 毕业顶点(R1-tiny mock 部署 + 五线综合回顾 Module1/3/4/5 + 5 mock checkpoint 端到端对比)"两段式结构决定的,不是本系列额外发明的拆分——08 号文件参照 [dsa-deep-dive/19-mock-interview-capstone.md](../dsa-deep-dive/19-mock-interview-capstone.md)/[statistics-deep-dive/21-mock-interview-capstone.md](../statistics-deep-dive/21-mock-interview-capstone.md) 的叙事体格式,不套用七步模板,进度表里记"1篇"不计入知识点总数。

已排除 `learning/cluster-networking/`——虽然名字看似相关,但它自称属于完全不同的"Module 8《系统与Infra》"系列(GPU架构→CUDA→kernel→cluster-networking→存储→训练编排→infra-graduation),前 3 站已被 [kernel-gpu-deep-dive/](../kernel-gpu-deep-dive/00-roadmap.md) 覆盖,cluster-networking 的 capstone 场景是训练时梯度 all-reduce,不是推理服务,以后应该接续 kernel-gpu-deep-dive 而不是塞进本系列。

每个知识点依然"从最笨的想法讲起"(比如 01 号文件先讲 naive KV cache 为什么在生产是灾难,再讲 PagedAttention 怎么解决这个问题),额外两块"底层机制/为什么这样设计"+"面试怎么问+追问链",和 torch-deep-dive/huggingface-deep-dive/peft-deep-dive 等系列完全一致。

## 环境声明

主线验证环境:仓库根目录 `.venv`(Windows 原生,Python 3.13,torch 2.11.0+cu128、transformers、fastapi、triton-windows 已装)。**7 个源模块的 `requirements.txt` 列出了 vllm/sglang/triton/auto-gptq/autoawq/bitsandbytes/ray,但已逐文件 grep 核实,`src/*.py` 里这些库几乎零真实 import**——可跑可测的部分全部是诚实标注的纯 Python/torch 算法复现(不是真起一个服务),29+ 条 `runbook.yaml` 命令全部 `gpu: false`。因此本系列 95%+ 内容(01-05、07、08 全部 + 06 除最后一点外)在根目录 `.venv` 即可完整验证,不需要额外安装重型框架。

**唯一例外**:06 号文件最后一个知识点(真实部署 bonus 案例)。这个知识点需要 WSL2 环境,和上面的"纯算法复现"性质完全不同,是全系列唯一一处真实起服务、真实发请求的案例:

1. WSL2 发行版:`Ubuntu`(24.04.1 LTS,Python 3.12.3 默认)。本机 `wsl -l -v` 另有 `RockyLinux`(`rhcsa-bash-deep-dive` 系列在用,互不干扰)和两个卡在 "Installing" 状态的 `Ubuntu-18.04`/`Ubuntu-20.04`(历史遗留失败注册,未使用,未清理)。
2. GPU 直通已确认:WSL2 内 `nvidia-smi` 可见 RTX 3080 Ti Laptop GPU,16384MiB(几乎全空闲),驱动 595.97,与 Windows 侧一致。
3. 独立 venv:`~/vllm-venv`(Python 3.12,`uv venv --python 3.12 --seed` 创建,**不与仓库另一套尚未使用的 tf-venv 混用**,呼应 [tensorflow-deep-dive/00-roadmap.md](../tensorflow-deep-dive/00-roadmap.md) 已声明的环境隔离原则),`uv pip install vllm --torch-backend=auto` 安装。
4. 这一处的真实结果(完全成功/部分成功/未验证跑通)会诚实写在该知识点自己的"环境声明"和"实测"段落里,不在这里预先断言——**撰写这一条知识点之前必须先确认波 0 的环境搭建结果,不能假设一定成功**。
5. 01 号文件知识点 11(mini-vLLM capstone)**选做**:如果上述 WSL2+vllm 环境搭建成功,可顺带跑一次 `inference-engine-core/src/vllm_compare.py` 本身的吞吐 benchmark(与 06 号 bonus 案例共享同一次环境安装成本),不强制,不影响该知识点主体判定完成。

## 知识点结构模板(七步,与 torch-deep-dive/huggingface-deep-dive/peft-deep-dive 完全一致)

1. **是什么** 2. **一句话** 3. **底层机制/为什么这样设计** 4. **AI 研究场景** 5. **可运行例子**(带 assert,真在 `.venv` 里跑过,唯一例外见上文环境声明第 4 点)6. **面试怎么问 + 追问链** 7. **常见坑**

七步是加粗行内标签(`**是什么:**` 这种格式),不是 markdown 标题。"可运行例子"下固定 1 个完整代码块,`sys.path.insert` 起手,密集 `assert`,不用 `print(f"OK...")`(那是"进阶深度追加"文件独有的约定,正文不用);代码块后另起段落 `**实测(`.venv` 真跑):**` 叙述真实数字。"面试怎么问+追问链"是 bullet list:`- **Q:** "..."—— 期望...` + `- **追问1:** "..."—— 期望...`(追问 2-3 视情况,不固定数目)。

## 进度表

| # | 分类 | 文件 | 知识点数(约) | 状态 |
|---|---|---|---|---|
| 01 | inference-engine-core(vLLM 骨架复刻) | [01-inference-engine-core.md](01-inference-engine-core.md) | 11 | ✅ 已完成(已验证,1254 行,全部 11 处可运行例子从最终 md 文件逐块提取、在干净 `.venv` 进程里重新跑通过;含四处发现——知识点 4 发现 `paged_attention_triton()` 全文没有一行 `import triton`,函数体是对 `paged_attention_torch` 的无条件委派,即便本机 `triton 3.7.1` 真实已装,装不装 triton 对这个函数的行为都没有任何影响;知识点 9 推翻了"Windows 上无法验证 CUDA Graph"的预设——裸 `torch.cuda.CUDAGraph` capture/replay 在本机(RTX 3080 Ti Laptop)真实生效,32 层/batch=1 达 21-23x、3 层/batch=8 达 16-18x 真实加速比,且过程中先用 `time.perf_counter()` 计时得到"replay 反而变慢"(0.84x)的误导性结论,换 `torch.cuda.Event` GPU 侧计时后才纠正,这段计时方法论教训被完整记录进正文;知识点 11 发现 `mini_vllm.py` 的 `prefix_cache` 字段被声明+import 但 `step()`/`add()`/`_make_table()`/`_can_admit()` 四个方法体内从未读写(接线未通电),以及 `vllm_compare.py` 模块 docstring 声称缺 vllm 时 `SystemExit(2)`,实测退出码是 `1`(传参是字符串不是整数);知识点 11 的 mini-vLLM 5 case 吞吐量本机实测比 README 记录值低 5-10 倍,但相对趋势(case2 远慢于其余四个)完全一致,坐实这套 benchmark 绝对数字对运行机器高度敏感、只有相对趋势可信) |
| 02 | sglang-radixattention(RadixAttention / Agent 推理) | [02-sglang-radixattention.md](02-sglang-radixattention.md) | 9 | ✅ 已完成(已验证,986 行,全部 9 处可运行例子在干净 `.venv` 进程里逐一重跑通过;含四处发现——知识点 2 用 01 号文件 `PrefixCache` 与本模块 `RadixTree` 并排实测,9-token 非对齐共享前缀在 `block_size=16` 下 block 哈希得 0、radix tree 精确得 9,量化坐实"树结构泛化"这一结论;知识点 3 发现 `RadixTree.root` 因赋值在 `__post_init__` 里、不是 dataclass 标注字段,导致两棵 token 内容完全不相干的树可被自动生成的 `__eq__` 判定"相等";知识点 5 发现 `grammar_fsm.py` 模块 docstring 宣称的 `\w`/`.`/`+`/`*`/`a|b` 五种语法一个未实现,且 lecture 反复强调的 token-level 查表函数 `compile_token_table` 经 AST 核实是全模块零调用点的死代码;知识点 7 发现 `frontend_lang.py::function` 装饰器未用 `functools.wraps`,被装饰函数(`react_loop`/`react_agent`)在 `__name__`/`__module__` 上"失忆",知识点 8-9 复用同一批函数时再次复现) |
| 03 | speculative-decoding(投机解码全家族) | [03-speculative-decoding.md](03-speculative-decoding.md) | 10 | ✅ 已完成(已验证,10/10 可运行例子干净进程重跑通过;发现 `MedusaHeads.draft()` 在 noise 足够大时会真实 `ZeroDivisionError` 崩溃——独立换 8 类扁平分布+6 头配置复现,noise=0.5 起就有概率触发,证实是 clamp-then-normalize 模式的结构性问题、不是某个分布形状的偶然;另发现 EAGLE 与 Medusa 在同噪声参数下 accept rate 几乎打平(0.486 vs 0.489),lecture 展示的"EAGLE 更强"印象来自两者默认噪声值不同,不是自回归结构本身在这份 mock 里带来的必然优势;L06 EAGLE-3 本仓库无对应源码,如实标注为概念性知识点,只用共享速度公式做方向性合理性检验) |
| 04 | quantization-deploy(推理期量化部署) | [04-quantization-deploy.md](04-quantization-deploy.md) | 11 | ✅ 已完成(已验证,11/11 可运行例子干净进程重跑通过;核心发现——`gptq_columnwise` 默认 `damp=0.01` 在校准数据列相关性较强/维度较高时数值不稳定,导致 GPTQ 补偿反而比朴素 RTN 更差(20 种子 18/20 复现),调大 `damp` 到 0.1 可稳定修复(10/10);同一根因在 capstone 自己的 toy 层构造上也被复现——"GPTQ<NF4"这条 README 宣称的排位关系在 seed 0-9 稳定成立(10/10)但 seed=99 反过来,两处发现互相印证;另发现 `fp8_demo.py` 的 E4M3 编码表实际最大值是 480 不是 lecture 声称的 448(未实现"最大指数最大尾数保留给NaN"这条真实硬件规范);以及模块自身 lecture 13 文档滞后于已重写的 capstone 源码和 README) |
| 05 | distributed-inference(分布式推理) | [05-distributed-inference.md](05-distributed-inference.md) | 9 | ✅ 已完成(已验证,9/9 可运行例子干净进程重跑通过;**核心发现**:`pp_demo.py::gpipe_bubble()` 公式只在 `n_stages=2` 时凑巧对,其余 15/18 组测试参数全部系统性低估真实 bubble(用 `schedule_naive()` 自己渲染的调度网格逐格计数验证地面真相,严谨公式 `(n_stages-1)/(n_micro+n_stages-1)` 全部吻合,`n_stages=6,micro=4` 时代码给 29.4% 而真相是 55.6%,几乎差一半);另发现 TP 列/行切分不是逐 bit 精确相等(~1e-6 浮点舍入,BLAS 分块方式随分片数变化);以及"命中率"指标在热门前缀基数小时对所有路由策略(含完全不看内容的 round_robin)都虚高,负载均衡度才是更诚实的策略差异化指标) |
| 06 | production-serving(生产级部署,含真实部署 bonus) | [06-production-serving.md](06-production-serving.md) | 11(+1 bonus 占位) | ✅ 常规 11 点已完成(已验证,11/11 可运行例子干净进程重跑通过;L06+L07 因均无对应源码且主题强关联,合并成 1 个知识点。**全系列最重要的发现**:`openai_api_server.py` 的 `/v1/chat/completions` 真实起服务后对**任何**请求(含 README 自己文档化的 curl 示例)统一返回 422——根因是 `from __future__ import annotations`(PEP 563 延迟注解求值)+ `Request` 类型仅在函数局部 `make_app()` 内 import 而非模块级,导致 FastAPI 运行时按 `__globals__` 解析字符串注解时找不到 `Request`,静默把该参数当成普通 query 参数处理;用完全独立、不复制原文件的最小 repro 单独复现同一故障模式,验证"仅需把 import 挪到模块级"即可修复;确认 Capstone(知识点 11)因直接 `from openai_api_server import app` 复用同一对象而原样遗传此 bug,并在不改源文件的前提下用其纯函数重新接线出一个可用版本。**知识点 12(真实部署 bonus)仍是占位**——WSL2 环境/vllm 0.25.0/GPU 直通均已就绪,但 `vllm serve` 因镜像缺 FFmpeg(vllm 无条件 `import torchcodec`)阻塞,已请用户手动 `sudo apt-get install -y ffmpeg`;2026-07-14 复查确认仍未安装,按计划既定预案不等待,先提交常规 11 点,bonus 待环境就绪后单独补写) |
| 07 | serving-graduation 常规专题 | [07-serving-graduation-topics.md](07-serving-graduation-topics.md) | 9 | ✅ 已完成(已验证,9/9 可运行例子干净进程重跑通过;实际合并方式与预案略有调整——L01+L03 合并(两种"agent 场景缓存"),L02(Thinking Budget)内容够厚独立成点,L09+L10 合并(冷启动+容错同属可靠性工程);全模块 4 个知识点(L03/L04/L09/L10)无独立 src,如实标注并分别复用系列内已验证真实代码(01 号文件 `NaiveKvPool`、06 号文件 `cost_calc.cost_for_workload`)或 Python 真实 stdlib(`functools.lru_cache`)做说明,唯一一处从零手写参照实现是 L09+L10 的 circuit breaker/backoff 状态机(标准 CS 确定性算法,非模拟 ML 效果)。**核心发现**:`thinking_budget.py::generate_with_budget()` 触发强制关闭(force-close)后直接 `break` 跳出循环,完整放弃了本该继续吐出的真实 answer 部分——用 3 组完全不同的参数(budget=8/15/3,不同长度 think、不同 answer 内容)独立复现,结论一致;另用 `NaiveKvPool` 验证 GQA 架构(8 KV heads)比无 GQA 假设(32 KV heads)省 4 倍显存,证实 lecture 插图性的"100k×32层×256KB≈800GB"是刻意夸张的教学量级,不对应任何真实架构) |
| 08 | serving-graduation 毕业顶点(capstone,叙事体) | [08-serving-graduation-capstone.md](08-serving-graduation-capstone.md) | 1 篇(3幕) | ⏳ 待撰写 |

**预计合计:约 65-80 个知识点,7 篇正文 + 1 篇 capstone(不单独计入知识点数)。**"进阶深度追加"本次不做,留到以后单独一轮,和其余 11 条系列的做法一致。

## 明细(源码路径,撰写时逐一核实文件路径/行号仍然准确)

### 01 inference-engine-core(源:`learning/inference-engine-core/src/*.py`,11 文件/1313 行,13 lectures/692 行)
1. LLM 推理服务全图——prefill/decode 两阶段特性差异(L01)
2. Naive KV Cache 及碎片问题(`naive_kv.py`,L02)
3. PagedAttention 原理(`paged_kv.py`,L03)—— OS 虚拟内存分页类比,block table 间接寻址
4. PagedAttention 的 Triton kernel(`paged_attention_triton.py`,L04)—— 与 [kernel-gpu-deep-dive/03](../kernel-gpu-deep-dive/03-kernel-design-triton-cutlass.md) 交叉引用,只讲分页在 kernel 层怎么落地,不重讲 Triton 语言基础
5. Continuous Batching(`continuous_batching.py`,L05)—— Orca,对比 static batching 的 GPU 空转问题
6. Chunked Prefill(`chunked_prefill.py`,L06)—— 长 prompt 和 decode 请求混批
7. Prefix Caching(`prefix_cache.py`,L07)—— 埋伏笔到 02 号文件的 RadixAttention
8. 调度策略(`scheduling_policies.py`,L08)—— FCFS/优先级/抢占
9. CUDA Graphs + Attention Backends(L09+L10 合并)—— 概念性,如实标注 Windows 无法编译真 CUDA Graph
10. Sampling 引擎(`sampling.py`,L11)—— temperature/top-p/top-k
11. Capstone:mini-vLLM 复刻(`mini_vllm.py`,L12+L13,5 case)——**选做**:若波0环境搭建成功,可加真 vllm 跑 `vllm_compare.py` 同 5 case 对照

### 02 sglang-radixattention(源:`learning/sglang-radixattention/src/*.py`,9 文件/929 行,11 lectures/565 行)
1. SGLang 全图与 vLLM 设计哲学差异(L01,呼应 L10)
2. RadixAttention 原理(L02)—— prefix caching 的树结构泛化,承接 01 号文件知识点 7
3. Radix Tree 实现细节(`radix_tree.py`,L03)
4. Constrained Decoding(`constrained_sampler.py`,L04)
5. Grammar FSM(`grammar_fsm.py`,L05)
6. Jump-Forward Decoding(`jump_forward.py`,L06)
7. Frontend DSL(`frontend_lang.py`,L07)
8. Agent 模式 5 种(`agent_patterns.py`,L08)—— ReAct/ToT/自洽等
9. Capstone:32 并发 Agent Server(`agent_server.py`,L11,radix 命中率 91.7%)+ vs vLLM 5 场景横评(L09+L10 并入)

### 03 speculative-decoding(源:`learning/speculative-decoding/src/*.py`,11 文件/757 行,12 lectures/542 行)
**特别纪律**:11 个 src 文件里只有 `capstone_eagle3.py`/`speculative_original_minimal.py` 两个有 `__main__`,其余 9 个纯库模块的"可运行例子"必须走 `sys.path.insert` 后直接 import 类/函数调用断言的路径,不能假设 `python xxx.py` 直接有输出。
1. 投机解码直觉(L01)—— 猜测+验证为什么不改变输出分布
2. Classic Speculative Decoding(`classic_spec_decode.py`,L02)
3. Medusa(`medusa_heads.py`,L03)
4. EAGLE(`eagle_minimal.py`,L04)
5. EAGLE-2(`eagle2.py`,L05)—— 动态草稿树
6. EAGLE-3(L06,呼应 `capstone_eagle3.py`)
7. Lookahead Decoding(`lookahead.py`,L07)
8. Self-Speculative Decoding(`self_spec.py`,L08)
9. Tree Attention + Dynamic Tree(`tree_attention.py`,L09+L10 合并)
10. 评测方法论(`spec_eval.py`,L11)+ Capstone(`capstone_eagle3.py`,L12,4method×5task 合成对照,如实标注 synthetic)

### 04 quantization-deploy(源:`learning/quantization-deploy/src/*.py`,10 文件/678 行,13 lectures/615 行)
1. 量化全图(L01)—— 精度/速度/显存三角权衡
2. int8 基础(`int8_basics.py`,L02)—— per-tensor/per-channel/per-group
3. GPTQ(`gptq_minimal.py`+`gptq_original_minimal.py`,L03)—— Hessian 近似逐列量化误差补偿
4. AWQ(`awq_minimal.py`,L04)—— activation-aware scaling
5. SmoothQuant(`smooth_quant.py`,L05)
6. LLM.int8()(L06)—— 离群值分解
7. FP8 格式(`fp8_demo.py`,L07)
8. FP8 训练(L08)—— 如实标注本模块只讲推理侧应用
9. W4A16/W4A8(`bnb_int4.py`,L09+L10 合并)—— 与 [peft-deep-dive/02-quantized-lora.md](../peft-deep-dive/02-quantized-lora.md) 的 NF4 是两个独立实现、不同生命周期阶段(训练时量化 vs 部署时量化),已核实无实质重复
10. KV Cache 量化(`kv_quant.py`,L11)
11. 评测方法论(`quant_eval.py`,L12)+ Capstone:量化动物园(`capstone_quant_zoo.py`,L13,6 种量化器真跑,真实重建 MSE/压缩比/显存)

### 05 distributed-inference(源:`learning/distributed-inference/src/*.py`,8 文件/1051 行,12 lectures/537 行)
1. 分布式推理全图(L01)—— 和分布式训练的关键差异(无反向传播/梯度 all-reduce)
2. Tensor Parallel + Megatron 风格(`tp_demo.py`,L02+L03 合并)
3. Pipeline Parallel + 1F1B 调度(`pp_demo.py`,L04+L05 合并)
4. Expert Parallel + All-to-All(`ep_demo.py`,L06+L07 合并)—— MoE 专属
5. Disaggregated Prefill/Decode(`disaggregated_mock.py`,L08)—— DistServe/Mooncake,和 01 号文件知识点 5"合并批处理"形成"合并 vs 分离"两种哲学对照
6. KV Cache 跨节点传输(`kv_transfer_mock.py`,L09)
7. Prefix-Aware Routing(`routing_policies.py`,L10)—— 和 02 号文件 RadixAttention 呼应("单机树" vs "多副本路由")
8. 多节点部署实践(L11)
9. Capstone:Disaggregated 3 配置对比(`capstone_disagg.py`,L12,colocate/disagg同节点/disagg跨节点,TTFT/TPOT/tok/s/wall time,真算不是硬编码)

### 06 production-serving(源:`learning/production-serving/src/*`,6 文件+2 模板/789 行,12 lectures/683 行;**含真实部署 bonus**)
1. 生产部署全图 + Clipper 历史回顾(L01+`clipper_original_minimal.py`)—— 2017 NSDI 系统,现代 LLM serving 栈的"史前"参照系
2. TensorRT-LLM 概览(L02)
3. TRT-LLM build 实战(`trtllm_build.py`,L03)—— Windows 无 trtllm,如实标注 mock 配置
4. Triton Inference Server(`triton_model_repo/`,L04)
5. Ollama(`ollama_modelfile/`,L05)
6. llama.cpp + GGUF + LM Studio(L06+L07 合并)—— 均无对应源码,合并为"端侧部署引擎/CLI/GUI 三层关系"1 个知识点
7. OpenAI 兼容 API 规范(`openai_api_server.py`,L08)—— 协议层纯函数与 FastAPI `app` 解耦;**真实起服务后 `/v1/chat/completions` 对任何请求统一 422**(根因:`from __future__ import annotations` + 函数局部 `Request` import,详见知识点正文)
8. FastAPI + SSE 流式包装(`streaming_sse.py`,L09)
9. 生产监控 Prometheus(`metrics_prometheus.py`,L10)
10. 成本工程(`cost_calc.py`,L11)
11. Capstone:生产栈整合(L12,mock,复用 `openai_api_server.py`)—— 知识点 7 的 422 bug 原样遗传到此处
12. **【真实部署 bonus,见环境声明;占位待补】**:WSL2 + 真实 vllm serve `Qwen/Qwen2.5-0.5B-Instruct`(`--quantization bitsandbytes`)+ 真实 OpenAI 兼容 API 请求,和知识点 7 的 mock 协议逐项对照(`/v1/models` 真模型名 vs mock 的 `"mock-7b"`;`usage.prompt_tokens` 真 tokenizer vs mock 的 `len(content.split())` 估算;真实 TTFT/TPOT;量化前后显存对比)。阻塞于 WSL2 镜像缺 FFmpeg,已请用户手动安装,2026-07-14 复查仍未就绪

### 07 serving-graduation 常规专题(源:`learning/serving-graduation/lectures/01-11.md`;实际合并方式:L01+L03、L09+L10,L02 独立成点,共 9 点)
1. Agent 场景两条缓存线:Prefix Cache + Reasoning Cache(L01+L03 合并)—— `agent_inference_demo.py` 真实函数 + `functools.lru_cache` 演示 reasoning cache 设计哲学
2. Thinking Budget(L02,独立成点,内容够厚不与 L01/L03 合并)—— `thinking_budget.py`,发现 force-close 丢答案的真实 bug
3. 长上下文推理服务(L04,无 src)—— 交叉引用 05 号文件 KV 传输机制 + [long-context-deep-dive](../long-context-deep-dive/00-roadmap.md) 的 RoPE 外推,复用 01 号文件 `NaiveKvPool` 验证 GQA 显存量级
4. 多模型路由(L05,`multi_model_router.py`)
5. Batch vs Online 推理(L06,无 src)—— 复用 06 号文件 `cost_calc.cost_for_workload` 演示 50% 折扣
6. VLM 推理服务(L07,`vlm_serve.py`,mock)
7. Embedding 服务(L08,`embedding_serve.py`,cosine() 数学性质独立复验)
8. 冷启动 + 容错(L09+L10 合并,均无 src)—— 手写 circuit breaker/backoff 参照实现(全系列唯一一处从零手写)
9. 服务工程 5 原则(L11,`serving_scorecard.py`)—— 呼应 DistServe goodput

### 08 serving-graduation 毕业顶点(叙事体,不拆知识点,源:L12+L13+L14)
- 幕一:Capstone-1 R1-tiny mock 部署(`r1_tiny_deploy/serve.py`)—— 如实标注这是手写 token 序列生成(`MockR1Model.stream()`)的叙事框架,不是真加载
- 幕二:五线综合(L13,回顾 Module1/3/4/5 共 25 个 topic)—— 主动链接回本仓库已完成的 [peft-deep-dive](../peft-deep-dive/00-roadmap.md)(对应Module1)和 [alignment-algorithms-deep-dive](../alignment-algorithms-deep-dive/00-roadmap.md)(对应Module4 DPO部分);Module4 的 reasoning-r1 和 Module3 的 phi_tiny 本仓库暂无对应精读系列,如实标注"这里只作为 mock checkpoint 消费,不深入其训练机制"
- 幕三:Capstone-2 五 mock checkpoint 端到端对比(`graduation_e2e/`,L14)—— 如实标注 5 个 checkpoint 是预烤好的字面量数据,不是真推理
- 复盘小结

## 撰写与验证纪律

- 每个知识点的可运行例子必须在仓库根目录 `.venv` 真实跑通(唯一例外:06 号文件知识点 13,需要 WSL2 真实 vllm 环境,标注见该知识点自身和上文环境声明)。
- `notebooks/` 目录下的 `.ipynb` 是空壳(270-630 字节,1 个标题 cell),不作为撰写素材来源,一律以 `lectures/*.md` + `src/*.py` 为准。
- 源模块 `requirements.txt` 列出的 vllm/sglang/triton/auto-gptq/autoawq/bitsandbytes/ray 在 `src/*.py` 里几乎未被实际 import,撰写"是什么"/"可运行例子"时以 grep 实测的真实依赖为准,不能假设 requirements.txt 等于真实依赖。
- 03 号文件特别注意:11 个 src 脚本只有 2 个有 `__main__`,其余 9 个纯库模块的可运行例子必须走 `sys.path.insert` 后直接 import 类/函数调用断言的路径。
- 涉及"真跑不是查表"的亮点(如 04 号 `capstone_quant_zoo.py`、05 号 `capstone_disagg.py`)撰写时应展开强调,这是源材料自己在开发过程中修正过的诚实纪律。
- 涉及纯叙事/mock 的部分(如 08 号 `r1_tiny_deploy`/`graduation_e2e`)如实标注,不冒充真实推理。
- 每篇写完后:全文代码块在干净 Python 进程里重新跑一遍,1-3 处最重要结论换参数/方法独立复现一遍,不能只信撰写 agent 的报告。

---
*创建:2026-07-13*
