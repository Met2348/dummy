# 12 · 序列化与部署基础(Serialization and Deployment)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 训练完一个 TF/Keras 模型,怎么把它存下来、换一台机器加载回来、交给 TF Serving/TFLite/ONNX Runtime 这些部署侧的消费者去跑——这一批和 [torch-deep-dive/10-serialization-and-deployment.md](../torch-deep-dive/10-serialization-and-deployment.md) 是完全同一个主题,但落地到 TF 这边会明显"碎"很多:PyTorch 基本上"`state_dict()` + `torch.save`"就能覆盖 90% 场景,TF 这边并存着 SavedModel、HDF5、新版 `.keras`、`tf.train.Checkpoint` 四套格式,外加 Keras 2/3 的路线分裂——这种碎片化本身就是历史遗留(Keras 从独立项目被并入 TF,又在 TF 2.16 起被拆回多后端 Keras 3)的产物,是本类目真正的看点,不是我在制造复杂度。

**环境声明(呼应 [00-roadmap.md](00-roadmap.md) 第 0 节,这里只重复和本篇直接相关的一句):** 本系列显式安装了 `tf_keras` 并设置 `TF_USE_LEGACY_KERAS=1`,让 `tf.keras` 解析回经典实现——第 8 节会现场验证这个设置具体在"存/取模型"这个环节上意味着什么,不是空口白话。

**"AI 研究/工程场景"段落的诚实声明(全系列统一,这里不重复展开):** 仓库里没有可引用的 TF/Keras 真实代码,以下场景是根据真实训练/部署流程重构的,不是仓库文件引用。

**验证方法论:** 本文所有代码在 WSL2 `~/tf-venv`(TensorFlow 2.21.0,RTX 3080 Ti Laptop GPU)下用 `source ~/tf-venv/bin/activate && python ...` 实际跑通;涉及"环境变量/Keras 版本切换才会触发"的行为,用子进程现场构造不同环境实测,不转述文档;ONNX 一节涉及的 `tf2onnx` 包本机未安装,如实说明,不假装跑通。

**本篇统一结构(同前几批):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场验证,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. SavedModel 格式 —— TF 的"官方标准"格式:计算图 + 权重 + 签名,不只是权重

**是什么:**
```
model.save("some_dir")                    # legacy tf.keras:路径不带扩展名 -> SavedModel 目录
tf.saved_model.save(obj, "some_dir")       # 框架级 API,obj 可以是任意 tf.Module,不一定是 Keras 模型
loaded = tf.saved_model.load("some_dir")   # 框架级加载,不依赖 Keras、不依赖原始 Python 类
```

**一句话:** SavedModel 是一个目录(不是单文件),里面除了权重(`variables/`)之外,还打包了通过 `tf.function` 追踪出来的**真实计算图**(`saved_model.pb`,protobuf 格式的 `ConcreteFunction` 集合)和具名的**签名**(下面第 5 节详解)——这意味着加载它不需要重新执行你写的 Python 前向代码,图本身已经是可独立执行的产物,这是它和"只存权重/只存配置"的 H5、`.keras` 最根本的区别。

**底层机制/为什么这样设计:**

`tf.saved_model.save` 会遍历传入对象的 `Trackable` 依赖图(第 4 节详解这套机制),把每个 `tf.Variable` 的值写进 `variables/`,同时把对象上被 `@tf.function` 修饰的方法**具体化(concretize)**成一份或多份 `ConcreteFunction`——即针对某个具体输入 shape/dtype 追踪出来的、不再包含 Python 控制流的计算图,序列化进 `saved_model.pb`。这一点和 H5/`.keras` 纯粹靠 `get_config()`/`from_config()` 记录"如何用 Python 重新构造这个对象"完全是两条路：**SavedModel 存的是"已经算出来的图长什么样",H5/`.keras` 存的是"怎么用代码重新构造这个对象"**。这也是为什么 `tf.saved_model.load()` 这个框架级加载入口完全不需要原始的 Keras 模型类定义——它拿到的直接是图,不需要重新跑一遍你的 `__init__`/`call`。

如果保存的对象是 Keras 模型,`model.save()` 在写 SavedModel 的同时还会额外塞一份 `keras_metadata.pb`,记录 `get_config()` 那套 Python 端重建信息——这是为了让 `tf.keras.models.load_model()` 之后仍能拿回一个"看起来和原来一样"的 Keras 模型对象(而不只是一个能调用的裸函数),两条元信息在同一个 SavedModel 里共存,互不冲突,按你用哪个 API 加载来决定读哪一份。

**AI 研究/工程场景:** 把模型交给非 Python 技术栈(TF Serving、C++ 推理服务、Java 后端)时,SavedModel 是唯一支持这些场景的格式,因为消费方根本不需要、也没有你的 Keras 类定义——它们只认图和签名;TFLite/TF.js 转换器(第 6 节)也统一以 SavedModel 作为标准输入格式。

**可运行例子:**

```python
import os, tempfile
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu", name="dense1"),
    tf.keras.layers.Dense(3, name="dense2"),
])
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))
y_ref = model(x).numpy()

sm_dir = os.path.join(tmpdir, "sm_model")
model.save(sm_dir)   # legacy tf.keras:无扩展名 -> SavedModel 目录

contents = sorted(os.listdir(sm_dir))
print("SavedModel 目录内容:", contents)
assert "saved_model.pb" in contents          # 计算图(protobuf)
assert "variables" in contents               # 权重
assert "keras_metadata.pb" in contents       # Keras专属:供 load_model() 做完整对象重建
assert sorted(os.listdir(os.path.join(sm_dir, "variables"))) == [
    "variables.data-00000-of-00001", "variables.index",
]

# 框架级加载:完全不需要 Keras、不需要这份 Python 代码里的 Sequential 定义
raw = tf.saved_model.load(sm_dir)
assert "serving_default" in raw.signatures     # Keras 的 save() 自动生成了默认签名(05节详解)
sig = raw.signatures["serving_default"]
input_name = list(sig.structured_input_signature[1].keys())[0]
out = sig(**{input_name: x})
out_key = list(out.keys())[0]
assert np.allclose(out[out_key].numpy(), y_ref, atol=1e-6)

# 用 Keras 高层 API 加载,数值同样一致(读的是同一份 SavedModel 里的 keras_metadata.pb)
loaded_keras = tf.keras.models.load_model(sm_dir)
assert np.allclose(loaded_keras(x).numpy(), y_ref, atol=1e-6)
print("raw tf.saved_model.load 与 tf.keras.models.load_model 结果都和原模型一致")
```

第二个例子验证"完全不依赖原始类"这个说法到底有多彻底——用一个**没有注册**的自定义 Layer:

```python
import os, tempfile
import numpy as np
import tensorflow as tf

class UndecoratedDouble(tf.keras.layers.Layer):
    """故意不用 @tf.keras.utils.register_keras_serializable() 装饰"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units)

    def call(self, x):
        return self.dense(x) * 2.0

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.dense.units})
        return cfg

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), UndecoratedDouble(3, name="custom1")])
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))
y_ref = model(x).numpy()
sm_path = os.path.join(tmpdir, "sm_custom")
model.save(sm_path)

# raw tf.saved_model.load:全程不需要知道 UndecoratedDouble 这个类的存在,直接吃内嵌的计算图
raw = tf.saved_model.load(sm_path)
sig = raw.signatures["serving_default"]
input_name = list(sig.structured_input_signature[1].keys())[0]
y_raw = sig(**{input_name: x})
out_key = list(y_raw.keys())[0]
assert np.allclose(y_raw[out_key].numpy(), y_ref, atol=1e-6)
print("raw tf.saved_model.load:全程不需要 UndecoratedDouble 这个 Python 类,数值正确")

# tf.keras.models.load_model 走 Keras 重建路径,即使类没注册也能成功——
# 但"复活"出来的是一个同名的动态代理类,不是原来那个类对象,isinstance 检查会失败
loaded = tf.keras.models.load_model(sm_path)
revived_layer = loaded.get_layer("custom1")
assert np.allclose(loaded(x).numpy(), y_ref, atol=1e-6)          # 数值依然正确(用的是内嵌 trace 图)
assert isinstance(revived_layer, UndecoratedDouble) is False       # 但不是原来那个类!
assert type(revived_layer).__name__ == "UndecoratedDouble"          # 类名相同,只是"复活"出来的代理类
print("revived_layer 实际类型:", type(revived_layer))
print("isinstance(revived_layer, UndecoratedDouble):", isinstance(revived_layer, UndecoratedDouble))
```

**面试怎么问 + 追问链:**
- **Q:** "SavedModel 和只存权重的格式,本质区别是什么?"—— 期望答"SavedModel 里打包了追踪出来的真实计算图(ConcreteFunction),不只是权重数值,所以加载不需要原始 Python 类代码就能拿到一个可执行的对象"。
- **追问 1(核心,区分度高):** "如果我保存 SavedModel 时用了一个自定义 Layer,加载时这个类的 Python 定义找不到了(比如换了一台机器,代码库没同步),会发生什么?"—— 期望能区分两条路径:走 `tf.saved_model.load()` 框架级加载完全不受影响(不需要这个类);走 `tf.keras.models.load_model()` 的 Keras 重建路径,即使类没注册也能"复活"出一个同名代理类、数值正确,但这个对象不再是 `isinstance` 意义上的原始类型——这是上面第二个例子现场验证的结论,很多人只知道"SavedModel 更鲁棒",答不出这个鲁棒性是有代价、有例外情况的。
- **追问 2(联系 torch10):** "这和 PyTorch `torch.save(model)` 整模型 pickle 需要原始类可 import,是同一个问题吗?"—— 期望能对比:表面相似(都涉及"能不能脱离原始类代码把模型救回来"),但 SavedModel 的 `tf.saved_model.load()` 路径**从设计上就不需要**原始类(存的是图,不是"如何构造这个类"的指令);pickle 存的恰恰是"这是哪个类、用什么数据重建它"这份指令本身,反序列化时必须能 `import` 到——这是两种本质不同的序列化哲学(图 vs 指令重放),TF 这边只有 Keras 重建路径才会部分触及类似的问题,框架级路径完全绕开了它。
- **追问 3:** "`keras_metadata.pb` 是干什么用的,删掉它 SavedModel 还能用吗?"—— 期望能答出"它只是给 `tf.keras.models.load_model()` 用来重建 Keras 对象外壳的元信息,`tf.saved_model.load()` + 手动调用 `.signatures[...]` 这条路径根本不读它,所以缺了它图本身依然完整可用,只是拿不回一个'像 Keras 模型'的对象"。

**常见坑:** 把 SavedModel 当成一个文件到处传("把 `xxx.pb` 发我一下")——它是一整个目录(`assets/`、`variables/`、若干 `.pb`),必须整体打包/传输,漏掉 `variables/` 目录会导致图能加载但权重全部是初始化状态,而且**不一定报错**(变量会用图里记录的初始值填充),这是比直接报错更危险的静默错误模式。

---

## 2. HDF5(`.h5`)格式 —— 历史上 Keras 的标准格式,只存权重 + 架构,不含计算图

**是什么:**
```
model.save("model.h5")                       # 显式 .h5 扩展名
loaded = tf.keras.models.load_model("model.h5")
```

**一句话:** `.h5` 是单文件格式(基于 HDF5 这种通用科学计算分层数据格式),内部只有两类东西——权重数值(按层级路径组织的 dataset)和一份**用 `get_config()` 生成的 JSON 架构描述**(存在文件的顶层 attribute 里)——**没有**第 1 节讲的追踪计算图,`load_model()` 拿到 JSON 之后是真的重新跑一遍 `from_config()` 把 Keras 对象"用代码重新构造"出来,不是直接读一张现成的图。

**底层机制/为什么这样设计:**

`.h5` 是 Keras 还是独立于 TF 的项目(多后端支持 Theano/TensorFlow/CNTK)时代就定下的格式——那个年代 Keras 本身不知道"计算图追踪"这种 TF 专属的概念,模型的可复现性完全依赖"权重 + 一份能重新 `import` 并 `from_config()` 出同构对象的架构描述"这条路,这也解释了为什么它先天不含签名、不含 SavedModel 那种可独立执行的图。这个设计直接决定了它的能力边界:任何 `get_config()`/`from_config()` 没有完整覆盖到的自定义逻辑(比如自定义训练细节、非标准 Python 控制流),`.h5` 都无法保真存下来,而 SavedModel 靠追踪实际执行路径反而能存下来。

**AI 研究/工程场景:** 大量存量的教程、Kaggle notebook、早期开源项目 checkpoint 仍然是 `.h5`,读旧代码库、复现别人几年前的实验时绕不开这个格式;它单文件、体积小、`h5py` 可以脱离 TF/Keras 直接用通用工具打开检查内容,做"这个 checkpoint 里到底存了什么"这类排查时比目录形态的 SavedModel 更直观。

**可运行例子:**

```python
import os, tempfile
import numpy as np
import tensorflow as tf
import h5py

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu", name="dense1"),
    tf.keras.layers.Dense(3, name="dense2"),
])
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))
y_ref = model(x).numpy()

h5_path = os.path.join(tmpdir, "model.h5")
model.save(h5_path)

with h5py.File(h5_path, "r") as f:
    top_keys = list(f.keys())
    attr_keys = list(f.attrs.keys())
    print("h5 顶层 keys:", top_keys)
    print("h5 attrs keys:", attr_keys)
    assert top_keys == ["model_weights"]              # 只有权重这一棵树,没有计算图/签名
    assert "model_config" in attr_keys                 # 架构JSON存在attribute里,不是普通dataset
    import json
    config = json.loads(f.attrs["model_config"])
    assert config["class_name"] == "Sequential"
    print("model_config 顶层 class_name:", config["class_name"])

loaded = tf.keras.models.load_model(h5_path)
assert np.allclose(loaded(x).numpy(), y_ref, atol=1e-6)
print("load_model(.h5) 数值和原模型一致")
# 和第1节对照:.h5 没有对应的"raw / 框架级"加载入口(SavedModel 有 tf.saved_model.load 这条路,
# .h5 没有等价物)——它天生就是 Keras 专属格式,这一点在 h5py 打开后看到的内容里已经体现:
# 顶层只有 model_weights 一棵树,没有任何计算图/签名相关的结构。
```

**面试怎么问 + 追问链:**
- **Q:** "`.h5` 和 SavedModel 存的内容有什么本质区别?"—— 期望答"`.h5` 只有权重 + 架构 JSON(`get_config()`产物),没有追踪出来的计算图,`load_model()` 靠重新执行 `from_config()` 构造对象,不是直接读图"。
- **追问 1:** "这意味着 `.h5` 在'自定义层能不能被正确保存'这件事上,和 SavedModel 相比有什么劣势?"—— 期望提到"`.h5`/`.keras` 这类纯 config 驱动的格式,自定义 Layer 必须能被 `get_config`/`from_config` 完整描述且类可解析(呼应第 3 节的 `register_keras_serializable`),SavedModel 因为存的是已经跑出来的图,即使类解析失败也能在框架级拿到一个能算数的对象(第 1 节验证过)"。
- **追问 2:** "`.h5` 里存的架构 JSON 具体是调用了模型的哪个方法生成的?"—— 期望直接答上 `get_config()`,并且能联系到"这就是为什么自定义 Layer 必须正确实现 `get_config()`/`from_config()` 这一对方法,它们才是 `.h5`/`.keras` 序列化真正依赖的契约"。

**常见坑:** 以为 `.h5` 存了优化器状态就能无缝"接着训练"——`model.save()` 默认确实会把优化器状态一起存进去(`model_weights` 之外还有 `optimizer_weights`),但如果加载时模型结构、优化器类型或参数顺序有任何变化,优化器状态的恢复比权重恢复脆弱得多,实践中"仅用于推理"或"精确复现训练环境"之外的场景,更推荐第 4 节的 `tf.train.Checkpoint` 或干脆重新初始化优化器。

---

## 3. 新版 `.keras` 格式 —— Keras 3 主推的新标准,zip 包,定位介于前两者之间

**是什么:**
```
model.save("model.keras")   # 扩展名必须是 .keras
```

**一句话:** `.keras` 本质是一个 **zip 压缩包**,内部拆成三个独立文件——`metadata.json`(版本信息)、`config.json`(架构,`get_config()` 产物,和 `.h5` 的 `model_config` attribute 是同一套信息只是换了容器)、`model.weights.h5`(权重,单独一个 H5 子文件)——是 `.h5` 单文件方案在 Keras 3 时代的"重新打包升级版",定位仍然是"Python/Keras 端的完整重建描述",不含 SavedModel 那种追踪计算图,不是 SavedModel 的替代品,是 `.h5` 的替代品。

**底层机制/为什么这样设计:**

把架构 JSON、权重、元信息拆成 zip 里的独立文件而不是像 `.h5` 那样全塞进一个 HDF5 树,换来两个直接好处:一是各部分可以独立读取/校验(比如只想看 `metadata.json` 确认版本,不需要拉起整个 HDF5 解析器);二是权重单独用 `model.weights.h5` 存放,格式上和 `model.save_weights()` 产出的纯权重文件同源,复用同一套读写代码路径。更重要的是这是 Keras 3"多后端"架构下的产物——同一份 `.keras` 文件设计上希望能被 TensorFlow/JAX/PyTorch 三种后端的 Keras 3 加载(只要各后端都认识 `config.json` 里描述的层类型),这是它和历史上强绑定 TF 的 `.h5`/SavedModel 在定位上最大的不同:`.h5`/SavedModel 是"TF 生态内部格式",`.keras` 从设计目标上是"Keras 概念本身的格式",只是当前主要还是在 TF 后端上用。

**AI 研究/工程场景:** Keras 官方文档和新版教程已经把 `.keras` 定为"新代码默认应该用的格式",`.h5` 标注为"legacy,仍支持但不再推荐";如果你在写新的训练脚本,`model.save("xxx.keras")` 应该是默认选择,只有在明确需要给 TF Serving/TFLite/C++ 这类"部署侧"消费者时才应该用 SavedModel(或第 8 节会讲到的 `model.export()`)。

**可运行例子:**

```python
import os, json, zipfile, tempfile
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu", name="dense1"),
    tf.keras.layers.Dense(3, name="dense2"),
])
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))
y_ref = model(x).numpy()

keras_path = os.path.join(tmpdir, "model.keras")
model.save(keras_path)

assert zipfile.is_zipfile(keras_path)          # 本质就是个 zip
with zipfile.ZipFile(keras_path) as zf:
    names = zf.namelist()
    print(".keras zip 内容:", names)
    assert set(names) == {"metadata.json", "config.json", "model.weights.h5"}
    metadata = json.loads(zf.read("metadata.json"))
    config = json.loads(zf.read("config.json"))
    print("metadata.json:", metadata)
    print("config.json 顶层 keys:", list(config.keys()))
    assert "keras_version" in metadata
    assert config["class_name"] == "Sequential"

loaded = tf.keras.models.load_model(keras_path)
assert np.allclose(loaded(x).numpy(), y_ref, atol=1e-6)
print(".keras 加载结果和原模型一致")

# 和 .h5/config.json 是同一套 get_config() 信息 —— 自定义层同样要求类可解析,机制和第1节完全一致
class UndecoratedDouble(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units)
    def call(self, x):
        return self.dense(x) * 2.0
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.dense.units})
        return cfg

model2 = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), UndecoratedDouble(3, name="c2")])
path2 = os.path.join(tmpdir, "custom.keras")
model2.save(path2)

try:
    tf.keras.models.load_model(path2)
    assert False, "预期应该报错"
except TypeError as e:
    # 真实触发的报错文本
    assert "Cannot deserialize object of type" in str(e)
    assert "register_keras_serializable" in str(e)
    print("\n未注册自定义类,加载 .keras 时真实触发的报错:")
    print(str(e)[:300])

y_ref2 = model2(x).numpy()
loaded2 = tf.keras.models.load_model(path2, custom_objects={"UndecoratedDouble": UndecoratedDouble})
assert np.allclose(loaded2(x).numpy(), y_ref2, atol=1e-6)
print("传 custom_objects= 显式映射后加载成功")
```

**面试怎么问 + 追问链:**
- **Q:** "`.keras` 和 `.h5` 是什么关系,和 SavedModel 又是什么关系?"—— 期望答"`.keras` 是 `.h5` 的继任者,都是纯粹的'Python 端重建描述'(架构 JSON + 权重),不含计算图;和 SavedModel 不是取代关系,是并存关系——一个面向'在 Python/Keras 里继续用',一个面向'部署给非 Python 消费者'"。
- **追问 1(区分度高,直接对照第1节结论):** "同样是自定义 Layer 没注册,`.keras` 报错和 SavedModel 走 `load_model()` 那条路径,行为一样吗?"—— 期望能精确区分:SavedModel 因为内嵌了追踪出来的图,`load_model()` 依然能"复活"出一个可用的代理对象(第1节验证过);`.keras`/`.h5` 完全没有图这个后备手段,只能靠 `get_config`/`from_config`,类解析不了就是硬报错(`TypeError: Cannot deserialize object of type ...`),没有中间地带——这个对比直接体现"有没有内嵌计算图"这个根本设计差异的实际后果。
- **追问 2:** "如果我一个模型里用了很多自定义 Layer,每次加载都要手动传一长串 `custom_objects=`,有没有更省事的办法?"—— 期望答出 `@tf.keras.utils.register_keras_serializable()` 装饰器——在类定义时注册进 Keras 的全局序列化表,加载时不需要手动传 `custom_objects`,只要这次进程里执行过这行装饰器代码(即模块被 import 过)即可。

**常见坑:** 文件扩展名写错(比如手滑写成 `.Keras` 或者忘记扩展名直接 `model.save("model")`)——legacy tf.keras 环境下无扩展名会退化成 SavedModel(第1节的行为),这经常不是你想要的结果,而且不会有任何警告提示你"你可能选错格式了";Keras 3 环境下则相反,不带受支持扩展名会直接报错拒绝保存(第 8 节现场验证)——两种环境对"该报错还是该猜"这件事的哲学是相反的,不能凭经验假设行为一致。

---

## 4. `tf.train.Checkpoint` vs 整模型保存 —— 对象图匹配 vs 扁平 key 匹配,机制和 torch state_dict 完全不同

**是什么:**
```
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step_var)  # 可以挂任意 Trackable 对象
save_path = ckpt.save(prefix)             # 写入 xxx.index + xxx.data-00000-of-00001
status = ckpt.restore(save_path)          # 只把数值restore进你已经构造好的对象图
status.assert_consumed()                  # 显式检查:是否完全匹配
```

**一句话:** `tf.train.Checkpoint` 是比第 1-3 节都更底层的存取原语——它不知道"架构"这个概念,只是沿着你传入对象的属性引用(`self.dense1`、`self.optimizer` 这类 Python 属性)递归遍历出一张 **`Trackable` 依赖图**,把每个叶子 `tf.Variable` 的值和它在这张图里的"路径"绑定后写进磁盘;恢复时同样靠这张路径去匹配,**不匹配的部分默认静默跳过,不报错**,这和 PyTorch `state_dict()` 基于扁平字符串 key 匹配（torch10 第 1、2 节)是完全不同的两套机制,只是解决的问题在直觉上相似。

**底层机制/为什么这样设计:**

TF1 时代的 `tf.train.Saver` 是"基于变量名字符串"存取的(和 torch `state_dict()` 的扁平 key 思路更接近),但 TF1 的变量名是由当时的 `variable_scope`/`name_scope` 拼接出来的,代码稍微重构一下作用域名字就变了、checkpoint 就废了,这是 TF1 用户的一个长期痛点。TF2 引入 `tf.train.Checkpoint` 时换了一套完全不同的思路——**不看变量叫什么名字,看这个变量在你的 Python 对象图里挂在哪条属性路径下**(`Trackable` 基类会自动记录"我被赋值给了谁的哪个属性名",`tf.Module`/`Layer`/`Optimizer` 都继承了这套机制)。这带来一个直接后果:只要你恢复时构造出的对象图**结构**(属性名、嵌套关系)和保存时一致,变量数值就能对上号,不要求你手写任何字符串 key 映射;但代价是这套匹配完全不做"名字模糊匹配"或"形状推断",一旦你重命名了某个属性(比如 `self.backbone` 改成 `self.encoder`),路径就断了,机制上和 torch `state_dict()` 里"改名导致 key 对不上"是同一类根因(标识符变了),但发生的层面不同——一个是 Python 属性名,一个是 dotted string key。

另一个和 torch10 第 4 节直接相关的差异点:`tf.train.Checkpoint` 序列化的是"沿着 `Trackable` 图能走到的变量值",天然不涉及"这是哪个 Python 类、需要 `import` 什么模块"这类信息——它不重建架构,只灌数值进你已经用代码构造好的对象——这意味着它没有 pickle 那种"反序列化可能执行任意代码"的攻击面,安全性质上更接近 torch 的 `state_dict()`(纯数据),而不是 `torch.save(model)`(整个对象连同类信息一起 pickle)。

**AI 研究/工程场景:** 训练循环里用同一个 `Checkpoint` 对象打包 `model` + `optimizer` + 一个记录训练步数的普通 `tf.Variable`,每隔 N 步 `ckpt.save()` 一次——重启训练时不仅权重、优化器动量能完整恢复,连"训练到第几步了"这种任意 Python 状态也一并救回来,这是它比"只存权重"的 `.h5`/`.keras` 更适合"断点续训"场景的原因,`tf.keras.callbacks.ModelCheckpoint` 内部按需也会走到这套机制。

**可运行例子:**

```python
import os, tempfile
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()

class Counter(tf.Module):
    def __init__(self):
        super().__init__()
        self.value = tf.Variable(0, dtype=tf.int64)

model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3, name="dense1")])
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
step_counter = Counter()
step_counter.value.assign(7)

ckpt = tf.train.Checkpoint(model=model, optimizer=opt, step=step_counter)
prefix = os.path.join(tmpdir, "ckpt", "ckpt")
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))
y_ref = model(x).numpy()

save_path = ckpt.save(prefix)   # 自动创建父目录,不需要提前 os.makedirs
files = sorted(os.listdir(os.path.dirname(prefix)))
print("checkpoint 文件:", files)
assert files == ["checkpoint", "ckpt-1.data-00000-of-00001", "ckpt-1.index"]

# .index 是二进制 protobuf,不是可读文本(用非文本判定,避免依赖具体字节序列)
with open(save_path + ".index", "rb") as f:
    head = f.read(32)
assert any(b > 127 or (b < 32 and b not in (9, 10, 13)) for b in head)
print(".index 文件头部含二进制控制字节,不是文本格式")

# --- 结构完全匹配的全新对象图:恢复成功,assert_consumed() 不报错 ---
model2 = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3, name="dense1")])
opt2 = tf.keras.optimizers.Adam(learning_rate=0.01)
step_counter2 = Counter()
ckpt2 = tf.train.Checkpoint(model=model2, optimizer=opt2, step=step_counter2)
status = ckpt2.restore(save_path)
status.assert_consumed()
assert np.allclose(model2(x).numpy(), y_ref, atol=1e-6)
assert step_counter2.value.numpy() == 7          # 任意Python状态(不只是模型权重)也恢复了
print("结构匹配:模型预测 + 训练步数计数器,全部正确恢复")

# --- 属性改名:静默不匹配,不报错,除非你主动检查 ---
model3 = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3, name="dense1")])
ckpt3 = tf.train.Checkpoint(renamed_model=model3)    # 属性名从 model 改成 renamed_model
status3 = ckpt3.restore(save_path)                    # restore() 本身不报错
y3_before_check = model3(x).numpy()
assert not np.allclose(y3_before_check, y_ref, atol=1e-6)   # 权重其实根本没被加载,还是随机初始化
try:
    status3.assert_consumed()
    assert False, "预期应该报错"
except AssertionError as e:
    assert "not bound to checkpointed values" in str(e)
    print("\n属性改名后 assert_consumed() 真实触发的报错(节选):")
    print(str(e)[:250])
```

第二段验证"不检查 status 会怎样"——TF 并不是完全沉默,而是在 status 对象被垃圾回收时**延迟**打一条警告日志,很容易被忽略:

```python
import os, sys, tempfile, gc
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3)])
opt = tf.keras.optimizers.Adam()
ckpt = tf.train.Checkpoint(model=model, optimizer=opt)
save_path = ckpt.save(os.path.join(tmpdir, "ckpt"))

model2 = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3)])
ckpt2 = tf.train.Checkpoint(model=model2)     # 故意只track model,不track optimizer
status = ckpt2.restore(save_path)             # 没有调用 assert_consumed(),也没有 expect_partial()

# 这条警告走的是"对象被真正回收时"才触发的延迟路径,不是Python logging模块的常规handler能可靠拦到的
# (它绑定在status/checkpoint对象的终结器上,只有引用计数真正归零才会同步触发)——
# 用OS级别重定向fd 2(stderr)配合彻底删除所有引用,如实复现这个时机
stderr_fd = sys.stderr.fileno()
saved_fd = os.dup(stderr_fd)
tmp_err = tempfile.TemporaryFile(mode="w+b")
sys.stderr.flush()
os.dup2(tmp_err.fileno(), stderr_fd)
try:
    del status
    del ckpt2
    del model2
    gc.collect()
    gc.collect()
finally:
    sys.stderr.flush()
    os.dup2(saved_fd, stderr_fd)
    os.close(saved_fd)

tmp_err.seek(0)
captured = tmp_err.read().decode(errors="replace")
tmp_err.close()
assert "expect_partial" in captured
print("对象真正被回收时才真实触发的延迟警告(节选):")
print(captured[:400])

# 用 expect_partial() 主动声明"我知道这是部分恢复",不再依赖GC时机,权重依然正确加载
model3 = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3)])
ckpt3 = tf.train.Checkpoint(model=model3)
status3 = ckpt3.restore(save_path).expect_partial()
x = tf.zeros((1, 4))
model3(x)  # build
print("expect_partial() 主动声明部分恢复:不再需要依赖对象回收时机才能确认没有被警告")
```

最后对照 Keras 高层的 `save_weights()`——它是 `Checkpoint` 机制的一层封装,不是另一套独立实现:

```python
import os, tempfile
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3)])

# 无扩展名:legacy tf.keras 下走的正是 tf.train.Checkpoint 格式,文件形态完全一致
sw_dir = os.path.join(tmpdir, "sw")
os.makedirs(sw_dir, exist_ok=True)
model.save_weights(os.path.join(sw_dir, "weights"))
assert sorted(os.listdir(sw_dir)) == ["checkpoint", "weights.data-00000-of-00001", "weights.index"]
print("save_weights(无扩展名) 产出的文件和 tf.train.Checkpoint 完全同构:", sorted(os.listdir(sw_dir)))

# 显式 .weights.h5 扩展名:走 H5 权重专用格式(第2节 H5 机制的"仅权重"子集)
h5_path = os.path.join(tmpdir, "weights.weights.h5")
model.save_weights(h5_path)
assert os.path.exists(h5_path)
print("save_weights('.weights.h5') 走的是 H5 权重格式,同一个高层 API 背后是两套不同底层机制")
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.train.Checkpoint` 存取模型,靠什么东西对齐权重和变量?"—— 期望答"沿 Python 属性引用递归遍历出的对象依赖图(`Trackable`),不是变量名字符串",并能对比 TF1 `Saver` 基于名字容易被作用域重构破坏的历史背景。
- **追问 1(直接对照 torch,区分度高):** "这和 PyTorch `state_dict()` 用扁平字符串 key、`strict=False` 精确字符串匹配的机制,谁更鲁棒?"—— 期望能给出有分寸的答案:不是单纯谁更好,而是匹配失败的**触发条件不同**——TF 属性改名和 torch 参数改名都会导致对应不上,这一点上两者其实同构;但**默认反馈方式不同**:torch `strict=True`(默认)是改名会立刻抛异常,TF `Checkpoint.restore()` 默认完全不吭声,只有你显式调用 `assert_consumed()` 才会暴露问题,不调用的话只在 status 对象被垃圾回收时才追加一条容易被日志噪音淹没的延迟警告——**默认哲学一个是"错了就喊",一个是"错了不说,除非你问"**,这个差异本身就是一个值得在面试里主动提出来的观察。
- **追问 2(联系 torch10 第 4 节安全性话题):** "`tf.train.Checkpoint` 存的东西,有没有 PyTorch pickle 整模型那种反序列化安全风险?"—— 期望答"没有同等级别的风险,因为它只序列化 `Trackable` 图上的变量数值,不记录'这是哪个 Python 类、如何用代码重建它'这类信息,反序列化不涉及动态 import/重建任意对象——这一点上它的安全性质更接近 torch 的 `state_dict()`,而不是 `torch.save(model)`"。
- **追问 3(工程向):** "`model.save_weights()` 和直接用 `tf.train.Checkpoint(model=model)` 有什么实际差别?"—— 期望能答出"`save_weights` 本质是 `Checkpoint` 的一层便捷封装,无扩展名时产出的文件格式完全一致;区别在于 `Checkpoint` 能同时打包 optimizer、自定义计数器等任意对象,`save_weights` 默认只处理模型自身这一个对象,想要打包更多东西还是得自己构造 `tf.train.Checkpoint`"。

**常见坑:** 恢复 checkpoint 后不检查 `assert_consumed()`/`assert_existing_objects_matched()`,只凭"程序没报错"就认为权重加载成功——上面第一个例子已经验证:属性改名之后 `restore()` 调用本身完全不报错,模型实际用的还是随机初始化权重,唯一的信号是一条容易被淹没在其他日志里的延迟警告,这类问题往往要等到"loss 一开始就明显不对、或者精度低得离谱"才会被排查出来,而且很容易被误判成别的原因(数据问题、学习率问题)。

---

## 5. `tf.saved_model.save` 的 signature 机制 —— concrete function 导出、具名 `serving_default`、多签名并存

**是什么:**
```
concrete_fn = obj.some_tf_function.get_concrete_function()   # 针对具体输入shape/dtype的图
tf.saved_model.save(obj, path, signatures={"serving_default": concrete_fn, "other_name": ...})
loaded.signatures["serving_default"](x=tensor)   # 调用约定:keyword-only,返回dict
```

**一句话:** "签名"是 SavedModel 对外暴露的**具名入口**——一个 SavedModel 内部可能包含很多 `ConcreteFunction`,但只有被塞进 `signatures=` 字典的那些才会被注册成外部消费者(TF Serving、`saved_model_cli`、跨语言调用方)能按名字发现和调用的入口,调用约定是**只认关键字参数**、返回值永远是一个按输出 tensor 名字组织的 `dict`,这是它和"直接 `model(x)` 调用"最大的语法差异。

**底层机制/为什么这样设计:**

`@tf.function` 修饰的方法在被具体的输入调用过至少一次(或者显式调用 `.get_concrete_function()`)之后,才会生成一份固定了输入 shape/dtype 的 `ConcreteFunction`——这是"图"这个概念第一次变成一个可以被单独拿出来、单独序列化的对象(呼应第 1 节"SavedModel 存的是已经追踪出来的图"这句话,这里就是那张图的来源)。`tf.saved_model.save` 的 `signatures=` 参数本质是"告诉序列化器,这些 `ConcreteFunction` 需要被特殊标记、赋予一个外部可查询的名字,并且要把输入输出的名字固定下来写进 `saved_model.pb`"——这些名字后续会变成调用约定的一部分:TF Serving 收到的 gRPC/REST 请求需要按名字把 tensor 塞进对应输入槽,这要求调用接口必须是"具名关键字"形式而不是"位置参数"形式(位置参数没有名字信息,没法对应到协议里按名字组织的请求),这也是它和"直接调用一个 Python 函数"在设计目标上根本不同的地方——签名从一开始就是为**跨进程、跨语言的具名调用协议**设计的,不是为 Python 内部方便调用设计的。

**AI 研究/工程场景:** 同一个模型给不同下游暴露不同"面孔"是签名机制真实存在的理由——比如给在线服务暴露一个只返回 top-1 类别的 `serving_default`,同时给需要做进一步分析/校准的下游暴露一个返回完整概率分布的 `probabilities` 签名,两者共享同一份权重和大部分计算,不需要导出两份模型;`saved_model_cli show --dir xxx --all` 这类命令行工具能罗列出一个 SavedModel 到底暴露了哪些签名、每个签名的输入输出名字和 shape 是什么,是排查"部署时到底该传什么参数名"最直接的手段。

**可运行例子:**

```python
import os, tempfile
import numpy as np
import tensorflow as tf

class LinearModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.normal([4, 3], seed=0))
        self.b = tf.Variable(tf.zeros([3]))

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32, name="x")])
    def serve_logits(self, x):
        return {"logits": self(x)}

    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32, name="x")])
    def serve_probs(self, x):
        return {"probabilities": tf.nn.softmax(self(x))}

    @tf.function(input_signature=[
        tf.TensorSpec([None, 4], tf.float32, name="x"),
        tf.TensorSpec([], tf.float32, name="scale"),
    ])
    def serve_scaled(self, x, scale):
        return {"scaled_logits": self(x) * scale}

mod = LinearModule()
x = tf.constant(np.random.RandomState(0).randn(2, 4).astype("float32"))

logits_cf = mod.serve_logits.get_concrete_function()
probs_cf = mod.serve_probs.get_concrete_function()
scaled_cf = mod.serve_scaled.get_concrete_function()

tmpdir = tempfile.mkdtemp()
sm_path = os.path.join(tmpdir, "multi_sig")
tf.saved_model.save(mod, sm_path, signatures={
    "serving_default": logits_cf,
    "probabilities": probs_cf,
    "scaled": scaled_cf,
})

loaded = tf.saved_model.load(sm_path)
assert set(loaded.signatures.keys()) == {"serving_default", "probabilities", "scaled"}

out_logits = loaded.signatures["serving_default"](x=x)
out_probs = loaded.signatures["probabilities"](x=x)
assert np.allclose(out_logits["logits"].numpy(), mod(x).numpy(), atol=1e-6)
assert np.allclose(tf.reduce_sum(out_probs["probabilities"], axis=1).numpy(), 1.0, atol=1e-5)
print("两个具名签名分别返回 logits 和 probabilities,和 eager 调用结果一致")

# 单输入签名:碰巧传一个位置参数也能work(实现细节,不代表这是承诺的用法)
single_pos_result = loaded.signatures["serving_default"](x)
assert np.allclose(single_pos_result["logits"].numpy(), mod(x).numpy(), atol=1e-6)
print("单输入签名:位置参数碰巧也能work,但不要依赖这一点")

# 多输入签名:位置参数被严格拒绝,真实报错精确到"too many positional arguments"
try:
    loaded.signatures["scaled"](x, tf.constant(2.0))
    assert False, "预期应该报错"
except TypeError as e:
    assert "too many positional arguments" in str(e)
    print("\n多输入签名下位置调用真实触发的报错(节选):")
    print(str(e)[:200])

# 关键字名字拼错:真实报错精确到"missing a required keyword-only argument"
try:
    loaded.signatures["serving_default"](wrong_kw=x)
    assert False, "预期应该报错"
except TypeError as e:
    assert "missing a required keyword-only argument" in str(e)
    print("\n关键字名拼错真实触发的报错(节选):")
    print(str(e)[:200])

# 没有显式传 signatures= 的裸 tf.Module,不会像 Keras 模型那样自动生成 serving_default
sm_nosig = os.path.join(tmpdir, "nosig")
tf.saved_model.save(mod, sm_nosig)
loaded_nosig = tf.saved_model.load(sm_nosig)
assert dict(loaded_nosig.signatures) == {}
print("裸 tf.Module 不传 signatures=:加载后 .signatures 是空字典,不像 Keras 模型自动给一个 serving_default")
```

**面试怎么问 + 追问链:**
- **Q:** "SavedModel 的'签名'是什么,为什么调用签名要用关键字参数而不是直接像调用函数一样传位置参数?"—— 期望答"签名是给跨进程/跨语言消费者用的具名入口,调用协议(TF Serving 的 gRPC/REST 请求)本身就是按名字组织输入 tensor 的,关键字调用约定是这套协议在 Python 端的直接体现"。
- **追问 1(区分度高):** "为什么第 1 节里 Keras 模型 `model.save()` 出来的 SavedModel,不用你自己传 `signatures=` 就自动有 `serving_default`,这里 `tf.Module` 却没有?"—— 期望能答出"自动生成 `serving_default` 是 Keras 的 `save()` 在背后帮你多做的一步(它知道模型的 `call`/输入 shape,能自动构造一个默认签名),这不是 `tf.saved_model.save` 这个更底层 API 的默认行为——直接用 `tf.saved_model.save` 存一个普通 `tf.Module`,没有任何签名是免费得到的,必须显式传 `signatures=` 才有"——这是上面代码最后一段专门验证的点,很多人默认"SavedModel 就该有 serving_default",其实那是 Keras 加的糖。
- **追问 2(工程向):** "如果模型需要同时支持'返回原始 logits'和'返回校准后概率'两种下游,应该导出几个 SavedModel?"—— 期望答"一个,用多个具名签名(`signatures={"serving_default": ..., "probabilities": ...}`)分别指向不同的 `ConcreteFunction`,不需要重复导出整份模型——权重只有一份,不同签名只是暴露了不同的计算路径出口"。
- **追问 3:** "`get_concrete_function()` 和直接把 `@tf.function` 修饰的方法传给 `signatures=` 有什么区别?"—— 开放题,合理方向:如果这个方法是多态的(能接受多种 shape/dtype,没有固定 `input_signature`),直接传裸的 `tf.function` 对象给 `signatures=` 在语义上是不明确的(该用哪次 trace 的图?),`get_concrete_function()` 强制你显式选定一份具体的、shape 已经固定的图,这是签名机制"必须有明确输入契约"这个设计要求决定的。

**常见坑:** 假设"只要用了 `tf.saved_model.save`,就自动有 `serving_default` 可以调"——上面已验证,这只在 Keras 模型的 `model.save()` 场景成立,直接用框架级 `tf.saved_model.save` 存一个 `tf.Module` 而不传 `signatures=`,得到的是一个 `.signatures` 为空字典的 SavedModel,拿去给 TF Serving 部署会在服务启动或第一次请求时才暴露"找不到 `serving_default`"这类错误,而不是在导出阶段就被发现。

---

## 6. TFLite 转换简介 —— 移动端/边缘设备部署,转换时做的优化与量化

**是什么:**
```
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)   # 或 from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]                      # 开启默认优化(含权重量化)
tflite_model = converter.convert()
```

**一句话:** TFLite 转换器把 SavedModel/Keras 模型的计算图整体转换成一种为移动端/嵌入式设备设计的、体积更小、依赖更少、支持算子集更精简的 FlatBuffer 格式(`.tflite`),`optimizations` 开关打开后默认做的是**动态范围量化**(权重从 float32 压缩成 int8,推理时激活值仍按 float 动态计算),不需要额外提供校准数据就能拿到明显的体积缩减。

**底层机制/为什么这样设计:**

移动端/边缘设备的核心约束是内存、算力、功耗都远小于服务器——TFLite 转换器要做的不只是"换个文件格式",而是要在保证数值可用的前提下尽量压缩体积、减少运行时依赖(TFLite 的算子实现是 TF 完整算子集的一个精简子集,不需要链接整个 TensorFlow 运行时)。动态范围量化能做到"不需要代表性数据就能量化"的原因是它只量化权重(训练完就固定不变的数值,离线统计一次每层权重的数值范围就够了),激活值(依赖运行时输入,分布无法离线预知)仍按原始精度动态计算——这是它跟"全整数量化"最大的区别:全整数量化连激活值也要量化成 int8,由于激活值分布依赖真实输入,必须额外提供一个 `representative_dataset` 生成器(用几百个有代表性的真实样本跑一遍模型,统计每层激活值的实际数值范围来标定量化参数),本文没有现场跑这一步,但机制上就是"多给转换器一份数据来源用于校准"。转换时输入 shape 会被固定下来(未知的 batch 维度默认按 1 处理),这是 FlatBuffer 格式本身对静态内存布局的要求决定的,想在推理时喂不同 batch size,需要显式调用 `resize_tensor_input()` 重新分配。

**AI 研究/工程场景:** 把一个在服务器上训练好的模型部署到手机 App(比如端上的关键词检测、图像预处理)或者嵌入式设备(比如摄像头里跑轻量检测模型)时,TFLite 几乎是 TF 生态里的默认选择;动态范围量化通常作为"不需要精度验证成本就能白捡的体积缩减"被默认打开,全整数量化则用在对延迟/功耗要求更极端、且有条件做完整精度回归测试的场景。

**可运行例子:**

```python
import os, tempfile, warnings
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(4),
])
x = np.random.RandomState(0).randn(5, 20).astype("float32")
y_ref = model(x).numpy()

sm_dir = os.path.join(tmpdir, "sm")
model.save(sm_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
tflite_plain = converter.convert()
plain_path = os.path.join(tmpdir, "plain.tflite")
with open(plain_path, "wb") as f:
    f.write(tflite_plain)

converter2 = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]   # 动态范围量化,不需要 representative_dataset
tflite_quant = converter2.convert()
quant_path = os.path.join(tmpdir, "quant.tflite")
with open(quant_path, "wb") as f:
    f.write(tflite_quant)

plain_size = os.path.getsize(plain_path)
quant_size = os.path.getsize(quant_path)
print(f"未量化 tflite 大小: {plain_size} bytes")
print(f"动态范围量化后大小: {quant_size} bytes  (比例 {quant_size/plain_size:.2%})")
assert quant_size < plain_size * 0.6   # 权重量化到int8,体积明显缩小

# tf.lite.Interpreter 目前仍可用,但已标注废弃,真实触发这条迁移警告
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    interp = tf.lite.Interpreter(model_path=plain_path)
    messages = [str(item.message) for item in w]
assert any("LiteRT" in m or "ai_edge_litert" in m for m in messages), messages
print("\n真实触发的迁移警告(节选):")
print(next(m for m in messages if "LiteRT" in m or "ai_edge_litert" in m)[:200])

def run_tflite(path, x_np):
    interp = tf.lite.Interpreter(model_path=path)
    inp = interp.get_input_details()[0]
    # SavedModel->TFLite 转换会把 batch 维度默认按1固定;喂不同batch前必须显式resize
    interp.resize_tensor_input(inp["index"], x_np.shape)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    return interp.get_tensor(out["index"])

y_plain = run_tflite(plain_path, x)
assert np.allclose(y_ref, y_plain, atol=1e-4)
print("\n未量化 tflite 推理结果和原 Keras 模型几乎完全一致")

y_quant = run_tflite(quant_path, x)
max_diff = np.abs(y_ref - y_quant).max()
print("量化后推理最大绝对误差:", max_diff)
assert max_diff < 1.0   # 有真实但有界的数值漂移,不是精确相等
print("量化引入了真实但有界的数值误差,不是精确匹配(符合预期,不是bug)")
```

**面试怎么问 + 追问链:**
- **Q:** "TFLite 转换默认做了哪些优化?量化是必须的吗?"—— 期望答"格式转换本身(FlatBuffer,精简算子集)和量化是两件独立的事,量化通过 `converter.optimizations` 显式开启,默认走的动态范围量化只压缩权重不需要额外数据,全整数量化需要代表性数据集校准激活值范围"。
- **追问 1(区分度高):** "动态范围量化和全整数量化,为什么一个不需要数据、一个需要?"—— 期望能讲清楚"权重训练完就固定了,离线统计一次数值范围就够;激活值依赖运行时输入,分布无法离线预知,必须喂真实/近似分布的数据实际跑一遍才能统计出合理的量化范围"这条因果链,而不是死记"哪个需要 representative_dataset"。
- **追问 2(工程向):** "转换好的 `.tflite` 模型,推理时能不能随便换 batch size?"—— 期望答"不能直接换,输入 shape 在转换时已经固定(未知维度默认按1处理),想用不同 batch 必须先 `resize_tensor_input()` 再重新 `allocate_tensors()`"——上面例子已经验证这一步是必需的,不做会直接抛 `ValueError: Cannot set tensor: Dimension mismatch`。
- **追问 3(时效性,考察是否关注生态变化):** "你现在用的 `tf.lite.Interpreter` 这套 API 还会长期维护吗?"—— 期望能提到"官方已经在往 `ai_edge_litert` 这个独立包(对应 LiteRT 品牌重塑)迁移,`tf.lite.Interpreter` 已经标注废弃、计划在后续版本删除,新项目应该关注这个迁移方向,不是只知道 `tf.lite` 这一个入口"——这是上面代码真实触发的警告,不是道听途说。

**常见坑:** 忽略"输入 shape 在转换时被固定"这件事,直接拿训练时任意 batch size 的输入去喂转换后的 `.tflite` 模型,得到 `ValueError: Cannot set tensor: Dimension mismatch. Got N but expected 1 for dimension 0 of input 0` ——这个报错文本本身已经说得很明确,但因为它发生在推理阶段而不是转换阶段,容易被误以为是"模型转换坏了",实际上只是忘了调用 `resize_tensor_input()`。

---

## 7. ONNX 互操作 —— TF 这边怎么转出去,双向转换在实践中会踩的算子不支持问题

**是什么:**
```bash
# tf2onnx 的典型用法(命令行),本机环境未安装该包,以下命令未实际执行,只作为文档记录
python -m tf2onnx.convert --saved-model /path/to/saved_model --output model.onnx --opset 18
```

**一句话:** TF 官方并不自带 ONNX 导出能力(不像 PyTorch 内建 `torch.onnx.export`,参见 [torch-deep-dive 第10篇第5节](../torch-deep-dive/10-serialization-and-deployment.md)),TF→ONNX 方向依赖社区维护的独立包 **`tf2onnx`**(以 SavedModel 为标准输入),反方向 ONNX→TF 目前实践中更多用另一个独立工具 **`onnx2tf`**——本机 WSL2 `~/tf-venv` 环境没有安装这些包,以下如实展示这一点,不假装跑通完整转换。

**底层机制/为什么这样设计:**

`tf2onnx` 的转换逻辑是"遍历 SavedModel 里的计算图,把每个 TF 算子按照一张内部维护的映射表翻译成等价的 ONNX 算子(可能一对一,也可能一个 TF 算子拆成好几个 ONNX 算子的组合)"——这套映射表覆盖不了 TF 全部算子集,遇到没有映射规则的算子(常见于比较新的层类型、TF 特有的字符串/资源类型操作、或者某些控制流结构),转换会直接报错并明确指出是哪个算子不支持,不会像早期一些转换工具那样静默跳过或错误替换。反方向的 ONNX→TF 转换还有一个 TF 圈子里非常经典的痛点:**数据排布约定不一致**——ONNX(继承自 PyTorch 等框架的惯例)默认用 NCHW(通道在前),TensorFlow 原生用 NHWC(通道在后),早期的 `onnx-tf`(onnx-tensorflow)工具在两种排布之间转换时,会在几乎每个卷积/池化算子前后插入 `Transpose` 节点来做布局转换,层数一多整张图会被这些 `Transpose` 节点"爆炸式"膨胀,严重拖慢推理速度——这也是后来 `onnx2tf` 这个社区工具存在的直接原因,它针对性地优化了这条转换路径,避免朴素插入 `Transpose` 的问题。

**AI 研究/工程场景:** 把 TF 训练的模型交给用 TensorRT/ONNX Runtime 做统一推理服务的团队时,ONNX 是最常见的中间交换格式(和 torch10 第 5 节的场景是同一个动机,只是这里是从 TF 出发);需要注意 `tf2onnx` 目前在其官方仓库上明确标注着"寻找新维护者"(截至 2026 年的现状,仍可正常使用,但更新节奏、对最新 TF/opset 的跟进速度值得关注,做生产依赖前建议自己验证一遍目标模型能否完整转换,不要假设"能装上就一定能转"。

**可运行例子:**

```python
# 如实记录本机环境现状:tf2onnx 未安装,以下是真实触发的 ModuleNotFoundError
try:
    import tf2onnx  # noqa: F401
    HAS_TF2ONNX = True
except ModuleNotFoundError as e:
    HAS_TF2ONNX = False
    real_error = str(e)

assert HAS_TF2ONNX is False
assert "tf2onnx" in real_error
print("本机 WSL2 tf-venv 环境未安装 tf2onnx,真实触发的报错:")
print("ModuleNotFoundError:", real_error)
print("\n若要转换,标准做法是: pip install tf2onnx 之后用")
print("  python -m tf2onnx.convert --saved-model <dir> --output model.onnx --opset <N>")
print("或 Python API: tf2onnx.convert.from_saved_model(...)")
```

**面试怎么问 + 追问链:**
- **Q:** "TF 模型怎么转成 ONNX?这个转换总能成功吗?"—— 期望答"依赖社区包 `tf2onnx`,不是 TF 自带能力;不能保证总成功,算子映射表覆盖不到的算子会直接报错终止,不会静默产出一个错误的图"。
- **追问 1(联系 torch10):** "PyTorch 内建 `torch.onnx.export`,TF 却要依赖第三方包,这个差异说明了什么?"—— 期望能给出"两个生态对 ONNX 的定位投入不同"这类观察,并且能对比失败模式:torch10 第 5 节验证过 `torch.onnx.export` 遇到数据依赖控制流会抛 `GuardOnDataDependentSymNode` 主动终止;`tf2onnx` 遇到映射表覆盖不到的算子同样会报错终止而不是静默转错——两边在"宁可失败也不要产出错误模型"这个原则上是一致的,只是失败的具体触发条件不同(一个是控制流,一个是算子映射缺失)。
- **追问 2(区分度高,考察是否了解生态现状而非死记文档):** "ONNX 转回 TensorFlow,和 TF 转出 ONNX,能用同一个工具吗?"—— 期望能答出"不能,是两个不同方向的独立工具链:TF→ONNX 用 `tf2onnx`,ONNX→TF 目前社区更活跃维护的是 `onnx2tf`(不是早期的 `onnx-tf`)——`onnx2tf` 存在的一大动机就是解决 ONNX 的 NCHW 和 TF 原生 NHWC 数据排布不一致导致朴素转换插入大量 `Transpose` 节点、拖慢推理的问题"。
- **追问 3(工程向):** "如果转换过程中报出'不支持的算子',除了等社区更新映射表,还有什么办法?"—— 开放题,合理方向:检查是否有等价写法能绕开这个算子(比如换一个功能等价但映射表支持的算子组合)、把不支持的子图单独拆出来自定义实现、或者确认目标 ONNX opset 版本是否覆盖了这个算子(有些算子只在较新 opset 里才有 ONNX 对应项,升级 `--opset` 参数可能就解决了)。

**常见坑:** 想当然地认为"两个框架都支持 ONNX,所以互转是无损、总能成功的"——实际上 ONNX 只是一个中间表示标准,具体转换成功与否完全取决于两边转换工具各自的算子覆盖率和实现质量,这一节验证的"包都没装"这个最朴素的现状本身就是常见坑的第一层:很多团队在真正需要用到 ONNX 之前根本没有把转换链路在自己的模型上跑通过一次,等到部署前最后一步才发现某个自定义层/较新算子没有映射规则,这类问题应该在项目早期就用目标模型跑一次转换验证,而不是假设"这是标准格式,肯定没问题"。

---

## 8. Keras 3 多后端与 legacy tf.keras 的选择 —— 面试怎么答"你怎么处理 Keras 2 到 3 的迁移"

**是什么:**
```bash
# ~/tf-venv/bin/activate 里设置的环境变量,决定 tf.keras 解析到哪个实现
export TF_USE_LEGACY_KERAS=1
```

**一句话:** 自 TF 2.16 起,`pip install tensorflow` 默认让 `tf.keras` 指向独立的多后端 **Keras 3**(可以跑在 TF/JAX/PyTorch 之上);想让 `tf.keras` 解析回历史上大家更熟悉的经典实现,必须**同时**满足两个条件——显式装上 `tf_keras` 这个包、**并且**设置 `TF_USE_LEGACY_KERAS=1` 环境变量——只装包不设环境变量,`tf.keras` 依然会解析到 Keras 3,这是本节第一个现场验证会打破的直觉误区。

**底层机制/为什么这样设计:**

Keras 3 是一次真正的代码库拆分,不是老 `tf.keras` 打了个新版本号——它是一个独立发布、能同时支持三种后端框架的包(顶层模块路径是 `keras.src.*`),而历史上内嵌在 TF 里的经典实现被原样抽成了一个单独的兼容包 `tf_keras`(模块路径是 `tf_keras.src.*`),两者是**两份不同的代码**,不是同一份代码的两个版本号。`tf.keras` 这个大家用惯的名字,现在只是一个"转发入口",转发去哪一份代码由环境变量决定;`tf_keras` 包只是让"转发去经典实现"这个选项**存在**,不设置环境变量的话,TF 默认转发到新的 Keras 3——这个"包在但默认不生效"的设计,是很多人第一次遇到 Keras 2/3 混乱现象的根源。这种分裂直接体现在序列化格式的兼容性上:`.keras`/`.h5` 这类纯"config + 权重"格式,因为 Keras 3 在设计时特意保留了对经典层类型名字/`get_config` 结构的兼容读取能力,两边基本能互相读;但 **SavedModel 格式的高层加载入口 `tf.keras.models.load_model()` 在 Keras 3 里被直接砍掉了**——SavedModel 曾经严重依赖经典 Keras 内部实现细节(`keras_metadata.pb` 的具体结构)来做对象重建,Keras 3 换了全新的内部结构,索性不再支持这条路径,官方给出的替代方案是专门为此新增的 `keras.layers.TFSMLayer`(只做推理,不重建完整 Keras 对象)。

**AI 研究/工程场景:** 接手一个几年前用经典 `tf.keras` 训练、以 SavedModel 格式存档的项目,在一台新装的、默认 Keras 3 环境的机器上尝试 `tf.keras.models.load_model()` 直接失败,是这几年 TF 社区里一个真实高频的报错来源;下面的例子会用两个独立进程(一个模拟"legacy 环境保存",一个模拟"Keras 3 默认环境加载")现场复现这个场景,而不是空口描述。

**可运行例子:**

先验证解析规则本身——包在不在不是决定性因素,环境变量才是:

```python
import os
import tensorflow as tf

# 本进程继承自 ~/tf-venv/bin/activate,TF_USE_LEGACY_KERAS=1 已经生效
assert os.environ.get("TF_USE_LEGACY_KERAS") == "1"
assert tf.keras.Model.__module__.startswith("tf_keras"), tf.keras.Model.__module__
print("环境变量=1 时,tf.keras.Model.__module__ =", tf.keras.Model.__module__)

import subprocess, sys
# 同一个 tf_keras 包依然装着,只是子进程里去掉环境变量 -> 验证"包在不代表默认生效"
env = dict(os.environ)
env.pop("TF_USE_LEGACY_KERAS", None)
code = "import tensorflow as tf; print(tf.keras.Model.__module__)"
result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
resolved_module = result.stdout.strip().splitlines()[-1]
assert resolved_module.startswith("keras."), resolved_module
assert not resolved_module.startswith("tf_keras")
print("去掉环境变量后(tf_keras包依然装着),tf.keras.Model.__module__ =", resolved_module)
print("结论:包只是让'能解析回legacy'这个选项存在,环境变量才是真正的开关")
```

再验证真正的痛点——用 legacy 环境保存,用 Keras 3 默认环境加载,现场对比 `.keras`/SavedModel 的实际行为差异(用子进程模拟两种独立环境,不依赖手动切换):

```python
import os, sys, subprocess, tempfile
import numpy as np
import tensorflow as tf

assert tf.keras.Model.__module__.startswith("tf_keras")   # 本进程:legacy tf.keras(保存方)

tmpdir = tempfile.mkdtemp()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(3, name="dense1"),
])
x = np.random.RandomState(0).randn(2, 4).astype("float32")
y_ref = model(x).numpy()
np.save(os.path.join(tmpdir, "x.npy"), x)
np.save(os.path.join(tmpdir, "y_ref.npy"), y_ref)
model.save(os.path.join(tmpdir, "m.keras"))     # 新格式
model.save(os.path.join(tmpdir, "m_sm"))        # SavedModel(legacy下无扩展名的默认行为)
print("已用 legacy tf.keras 存好 .keras 和 SavedModel 两种格式")

child_code = """
import tensorflow as tf, numpy as np, os
assert not tf.keras.Model.__module__.startswith("tf_keras"), tf.keras.Model.__module__  # 加载方: Keras 3 默认
x = np.load(os.path.join(r"{tmp}", "x.npy"))
y_ref = np.load(os.path.join(r"{tmp}", "y_ref.npy"))

# .keras 跨版本能读:Keras 3 保留了对经典config结构的兼容
m1 = tf.keras.models.load_model(os.path.join(r"{tmp}", "m.keras"))
assert np.allclose(m1(x).numpy(), y_ref, atol=1e-6)
print("KERAS_FMT_OK")

# SavedModel 走 tf.keras.models.load_model() 在 Keras 3 下被直接拒绝
try:
    tf.keras.models.load_model(os.path.join(r"{tmp}", "m_sm"))
    print("SAVEDMODEL_LOADMODEL_UNEXPECTEDLY_OK")
except ValueError as e:
    print("SAVEDMODEL_LOADMODEL_FAILED::" + str(e))

# 官方给的替代方案:TFSMLayer,只做推理,不重建完整Keras对象
tfsm = tf.keras.layers.TFSMLayer(os.path.join(r"{tmp}", "m_sm"), call_endpoint="serving_default")
out = tfsm(x)
key = list(out.keys())[0]
assert np.allclose(out[key].numpy(), y_ref, atol=1e-6)
print("TFSMLAYER_OK")
""".format(tmp=tmpdir)

env = dict(os.environ)
env.pop("TF_USE_LEGACY_KERAS", None)   # 模拟"没装/没启用legacy keras"的默认Keras 3环境
result = subprocess.run([sys.executable, "-c", child_code], env=env,
                         capture_output=True, text=True, timeout=100)
print("\n--- Keras 3 默认环境(子进程)的实际输出 ---")
print(result.stdout)

assert "KERAS_FMT_OK" in result.stdout
assert "SAVEDMODEL_LOADMODEL_FAILED::" in result.stdout
assert "TFSMLayer" in result.stdout          # 报错信息本身就指向了官方推荐的替代方案
assert "TFSMLAYER_OK" in result.stdout
print("\n结论: .keras/.h5 跨Keras版本基本兼容; SavedModel 的 load_model() 不兼容,")
print("但换成 TFSMLayer 或 tf.saved_model.load()+signatures 依然能拿到正确结果")
```

最后验证 Keras 3 默认环境下"存"这一侧也变严格了,呼应第 3 节结尾提到的"两种环境对该不该报错的哲学相反":

```python
import subprocess, sys, os, tempfile

code = """
import tensorflow as tf, tempfile, os
assert not tf.keras.Model.__module__.startswith("tf_keras")
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3)])
tmp = tempfile.mkdtemp()
try:
    model.save(os.path.join(tmp, "no_ext_model"))
    print("UNEXPECTEDLY_OK")
except ValueError as e:
    print("NO_EXT_SAVE_FAILED::" + str(e))

# 官方指向的替代方案:model.export() 专门产出推理用 SavedModel
export_dir = os.path.join(tmp, "exported")
model.export(export_dir)
print("EXPORT_CONTENTS::" + ",".join(sorted(os.listdir(export_dir))))
"""
env = dict(os.environ)
env.pop("TF_USE_LEGACY_KERAS", None)
result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
print(result.stdout)

assert "NO_EXT_SAVE_FAILED::" in result.stdout
assert "Invalid filepath extension" in result.stdout
assert "model.export" in result.stdout        # 报错文本里直接指路
assert "EXPORT_CONTENTS::" in result.stdout
print("Keras 3 默认环境下 model.save(无扩展名) 直接报错拒绝,")
print("legacy tf.keras 下同样的调用会静默退化成 SavedModel(第3节验证过)——行为相反,不能凭经验假设")
```

**面试怎么问 + 追问链:**
- **Q:** "你怎么处理 Keras 2 到 Keras 3 的迁移/兼容问题?"—— 期望候选人不要只回答"装 `tf_keras` 包就行了",而是分层回答:(1) 解析规则上,包 + 环境变量 `TF_USE_LEGACY_KERAS=1` 两者都要,只装包不生效;(2) 格式兼容性上,`.keras`/`.h5` 基本能跨版本互读,SavedModel 的 `load_model()` 高层入口在 Keras 3 里被砍掉了,需要换成 `TFSMLayer` 或者退回到 `tf.saved_model.load()` 这层框架级 API;(3) 存的一侧行为也变了,Keras 3 对不带受支持扩展名的 `save()` 直接报错,不再是 legacy 那种"静默退化成 SavedModel"。
- **追问 1(区分度极高):** "如果同事给你发来一个用旧代码训练、存成 SavedModel 的模型,你的环境是默认 Keras 3,`tf.keras.models.load_model()` 报错了,你怎么办,不能重新训练?"—— 期望能现场想到上面验证过的两条路:要么用 `tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")` 包一层当推理层用;要么更底层地 `tf.saved_model.load(path)` 拿到 `.signatures['serving_default']` 直接手动调用——两条路都不需要重新训练,只是拿不回一个完整可继续训练的 Keras 对象,仅用于推理是够用的。
- **追问 2(考察是否真的理解报错本身):** "`tf.keras.models.load_model()` 加载 legacy SavedModel 在 Keras 3 下具体报的是什么错,报错信息有没有用?"—— 期望能提到"报错信息本身其实相当友好,直接点名'legacy SavedModel format is not supported by load_model() in Keras 3'并且给出了 `TFSMLayer` 的具体调用示例,这不是一个需要查文档才能解决的报错,读报错原文就够了"——这是一个考察"你是不是真的会读错误信息"的追问,很多人遇到报错第一反应是搜索引擎而不是完整读一遍报错文本。
- **追问 3(工程向,前瞻性):** "如果你现在要开始一个新项目,应该主动选 legacy tf.keras 还是 Keras 3?"—— 开放题但有明确倾向:面试题库和大量存量教程还停留在经典 tf.keras 心智模型(这也是本系列显式选择 legacy 的原因,呼应 [00-roadmap.md](00-roadmap.md) 的环境声明),但新项目长期看应该倾向 Keras 3(多后端能力、官方主线维护方向),两边都要认识、能叫出对方的名字和边界,是这个问题真正想考察的成熟度,而不是选边站。

**常见坑:** 只知道"装 `tf_keras` 就能解决 Keras 3 兼容问题",忽略环境变量这一步——上面第一个例子已经验证,`tf_keras` 包装着但不设 `TF_USE_LEGACY_KERAS=1` 的情况下 `tf.keras` 依然解析到 Keras 3,如果这时候还带着"我已经在用 legacy tf.keras 了"的错误假设去调试一个实际发生在 Keras 3 上的行为差异,会在错误的方向上排查很久;另一个常见坑是遇到 SavedModel 加载失败就直接认定"这个 checkpoint 坏了/不兼容,只能重新训练",而没有意识到 `TFSMLayer`/`tf.saved_model.load()` 这类推理侧的救援路径完全够用。

---

## 小结:这一批 8 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | SavedModel 格式 | 目录形态,内含追踪出的计算图+权重+签名;`tf.saved_model.load()` 完全不依赖原始 Python 类,`load_model()` 的 Keras 重建路径即使类未注册也能"复活"出同名代理对象(数值正确但 isinstance 失败) |
| 2 | HDF5(`.h5`)格式 | 单文件,只有权重 + `get_config()` 产出的架构 JSON,不含计算图,没有脱离 Keras 的框架级加载入口 |
| 3 | 新版 `.keras` 格式 | zip 包(metadata.json+config.json+model.weights.h5),是 `.h5` 的继任者而非 SavedModel 的替代品;自定义层未注册时报错精确、无中间地带 |
| 4 | `tf.train.Checkpoint` vs 整模型保存 | 按 Python 对象图路径匹配,不是扁平字符串 key;默认静默不匹配(仅延迟警告),需主动 `assert_consumed()`,与 torch `strict=True` 默认告警的哲学相反;不涉及类重建,无 pickle 级安全风险 |
| 5 | `tf.saved_model.save` 签名机制 | `signatures=` 注册具名入口,调用约定关键字参数专用、返回 dict;`serving_default` 是 Keras `save()` 加的糖,裸 `tf.Module` 不传 `signatures=` 什么签名都没有 |
| 6 | TFLite 转换 | 动态范围量化只压权重不需要数据,全整数量化需要 `representative_dataset` 校准激活值;输入 shape 转换时固定,需 `resize_tensor_input()`;`tf.lite.Interpreter` 已标注废弃,迁移方向是 `ai_edge_litert`(LiteRT) |
| 7 | ONNX 互操作 | TF→ONNX 靠社区包 `tf2onnx`(非官方内建),ONNX→TF 现在更多用 `onnx2tf`;NCHW/NHWC 布局不一致是双向转换的经典痛点;本机环境未安装,如实记录 |
| 8 | Keras 3 多后端 vs legacy tf.keras | 包 + 环境变量缺一不可;`.keras`/`.h5` 跨版本基本兼容,SavedModel 的 `load_model()` 高层入口在 Keras 3 被砍掉,救援路径是 `TFSMLayer`/`tf.saved_model.load()`/`model.export()` |

下一批:[13-debugging-and-common-errors.md](13-debugging-and-common-errors.md) —— 调试与常见报错精解(全系列收尾)。

---

*更新:2026-07-09*
