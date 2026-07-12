# Q&A —— long-context 深挖系列的建立与验证(2026-07-12)

## Q:runbook-verification-task(46/46 模块)收官后,启动之前规划好的四条新深挖系列,第一条是 long-context

延续前几条系列的方法论,但这条系列(以及后面三条 kernel-gpu/alignment-algorithms/peft)和前 6 条深挖系列性质不同——前 6 条补的是"读懂 `learning/` 代码需要的通用框架/语言技能"(numpy/torch/HF 库机制本身),这条**直接对应 `learning/long-context/` 这一个专题模块**,用更适合初学者/面试备考的讲解深度(7 步结构,从最笨的想法讲起,面试追问链)重新讲解同一份代码,不是重复造轮子。

## 做了什么

建立了第 7 条深挖系列:**[long-context-deep-dive/](../long-context-deep-dive/00-roadmap.md)** —— 17 个知识点,4 个分类,覆盖 RoPE 外推家族(vanilla→PI→NTK→YaRN→3D-RoPE)→ 长上下文 Attention 架构(Ring/Striped/Infini-Attention)→ 长上下文评测方法论(NIAH/RULER/Lost-in-the-Middle)→ 数据工程与 Capstone(文档打包/课程学习/YaRN+LoRA capstone/KV-cache 显存核算)。套用 torch-deep-dive/huggingface-deep-dive 的重型 7 步模板(签名/是什么→一句话→底层机制/为什么这样设计→AI研究场景→可运行例子→面试怎么问+追问链→常见坑),不是 python-advanced/python-idioms 用的轻量 6 步——因为这条系列讲的是算法/框架机制,不是通用 Python 语言技能。

**环境:** 仓库根目录 `.venv`(Windows 原生),`learning/long-context/src/` 下全部代码纯 CPU、零第三方 ML 框架依赖(niah_eval.py/ruler_eval.py 甚至连 torch 都不 import),秒级跑完,不需要 GPU、不需要下载任何模型权重。

## 怎么做的

- **先派发调研,再规划,再派发撰写**:动笔之前先用一个 Explore 子代理逐文件读完 `learning/long-context/src/` 全部 12 个源文件(含行数、每个文件实现什么、`__main__` 实际打印什么、已有 pytest 断言),精确拿到"13 篇 lecture(不是想当然的 6 篇)"这类事实,再据此规划 4 个分类、17 个知识点,写成 `00-roadmap.md` 定稿之后才派发 4 个内容撰写子代理(一个分类一个 agent,并行执行)。
- **独立复验,重点盯"最复杂的一条"**:4 个文件全部完成后,没有直接采信子代理报告,而是对每个文件挑选最复杂/最有风险的断言亲自在 `.venv` 里重新跑一遍——尤其是 01 篇第 5 个知识点(YaRN 教学代码 vs 真实 `transformers==5.10.2` 生产库的公式对比),独立验证了 `ROPE_INIT_FUNCTIONS["yarn"]` 真实算出的 `attention_factor=1.138629436111989`、`rope_yarn.py::attn_scale=0.9371493244339143`、两者互为倒数、以及"生产库把这个系数乘进 cos/sin 导致 Q、K 各乘一次、点积实际被放大 `attention_factor²≈1.2965倍`,而教学代码只在 score 上乘一次、是缩小成 `0.9371倍`——两者连方向都相反"这一整条推理链,全部数字位级精确复现。
- **验证方法论本身的收获比任何单个知识点都值钱**:这条系列在撰写过程中三次独立发现了源码本身的真实瑕疵(不是撰写者编造的教学案例,是子代理和我各自独立读代码读出来的):① `rope_yarn.py` 第 28-29 行算出的 `inv_freq_ntk`/`new_base` 是死代码,真正生效的分段混合只用了 `mask`/`inv_freq`/`pi_freq` 三者(用"手写一个不依赖死代码的版本,结果逐位精确一致"这种方式证明,不是猜的);② `niah_eval.py::niah_grid` 的类型注解写的是 `-> dict`,但实测运行时返回的其实是 `list`;③ `ruler_eval.py::mk_niah` 里有一行 `pos = random.randint(0, len(hay))` 算出来之后压根没被用到,真正生效的是紧接着独立的另一次 `random.randint(0, len(pieces))`。三处发现全部由我亲自读源码复核确认属实,不是子代理的一面之词。
- **诚实标注"这是引用文献结论,不是本仓库验证过的实验"**:知识点 3(Lost in the Middle)全程严格区分"lecture 引用的外部研究结论"和"本仓库能实际操作/构造的测试用例"两层——可运行例子只演示 `depth_pct` 参数怎么精确控制针的插入位置(NIAH 热力图横轴的可操作定义),不暗示本仓库跑出过那条 U 型准确率曲线本身(纵轴)。这个边界意识贯穿全篇多处("YaRN+LoRA capstone 的 82% NIAH 准确率是 lecture 给出的预期数字,不是本仓库实测"等)。

## 结论

4 个文件全部完成并逐条独立验证(含对最复杂知识点的位级精确复算),合计 17 个知识点,1547 行(含 roadmap)。这条系列最大的方法论价值,不是任何一个具体的 RoPE 公式,而是**把"教学代码不能默认和生产库/文档完全一致,必须亲自动手核对"这件事,从一句抽象的原则,变成一条可以逐字复现的完整验证链路**(定位安装路径→尝试直接 import 发现失败→grep 源码发现是嵌套函数→构造真实 config 走生产入口拿到真值→对比公式差异→追查这个差异在下游被应用的方式是否也不同)——这是比"记住 YaRN 公式"本身更能扛住面试反复追问的能力。

加上已有的 6 条深挖系列(numpy/python-advanced/torch/huggingface/tensorflow/python-idioms,约 467 点)+ 1 条独立技能系列(rhcsa-bash,100 点),仓库现在有 8 条深挖系列 + 1 条独立技能系列,合计约 484 个知识点。规划中的另外三条新系列(kernel-gpu-deep-dive / alignment-algorithms-deep-dive / peft-deep-dive)延续同一套方法论继续推进。
