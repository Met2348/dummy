# be-07 · 设计模式与工程实践

对应代码：[`src/backend_qa/qbank_design_patterns.py`](../src/backend_qa/qbank_design_patterns.py)（18 题）+ 代码验证 [`src/backend_coding/design_patterns_demo.py`](../src/backend_coding/design_patterns_demo.py)。

## 这块在国内大厂后端面试里的分量

设计模式在面试里几乎不会被单独口头问"背一下 GoF 23 种模式"，它真正的杀伤力在**"手撕代码"环节**：面试官会直接让你现场写一个线程安全的单例、写一个能扩展的日志/支付渠道工厂、或者用装饰器给一个类加缓存能力——你不仅要写出能跑的代码，还要在写的过程中被追问"为什么这里要加锁"、"为什么不用继承"。SOLID 原则、Code Review、CI/CD 这些则更多出现在"你们团队怎么做代码质量保障"这类偏工程素养的开放问题里，考察的不是背诵而是有没有真正踩过坑。整体来说，这块内容"手写正确"比"口头背对"更重要，建议对照 `design_patterns_demo.py` 把 5 个模式的最小实现自己重敲一遍，而不是只读答案。

## 追问链深挖 1：单例模式的线程安全演进

**Q1（面试官）：写一个单例模式。**

**A1**：最直接的写法是懒汉式（lazy initialization）——第一次调用 `getInstance()` 时才创建实例：

```java
public class Singleton {
    private static Singleton instance;
    private Singleton() {}
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}
```

**Q2（追问）：这个写法在多线程下有什么问题？**

**A2**：`if (instance == null)` 判空和 `instance = new Singleton()` 创建不是一个原子操作。假设线程 A 和线程 B 几乎同时调用 `getInstance()`，都读到 `instance == null`，于是都会各自执行 `new Singleton()`，最终产生两个不同的实例，破坏了单例"全局唯一"的语义。如果构造函数里还有副作用（比如打开一个文件句柄、注册到某个全局注册表），这个副作用还会被执行两次。最直接的修复是给整个方法加 `synchronized`：

```java
public static synchronized Singleton getInstance() {
    if (instance == null) {
        instance = new Singleton();
    }
    return instance;
}
```

**Q3（追问）：这样修复之后还有什么问题？**

**A3**：正确性没问题了，但性能不好——`getInstance()` 在实例创建完成之后，99% 的调用其实只是"读一下已经存在的 instance"，根本不需要加锁，但整体加锁之后每次调用都要竞争同一把锁，高并发场景下会成为性能瓶颈。于是有了双重检查锁（Double-Checked Locking, DCL）：只在实例还没创建时才进入同步块，创建完成之后走的都是无锁的第一次判空：

```java
public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {}
    public static Singleton getInstance() {
        if (instance == null) {                 // 第一次判空：性能优化，避免每次都抢锁
            synchronized (Singleton.class) {
                if (instance == null) {          // 第二次判空：正确性保证
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

**Q4（追问）：为什么第二次判空是必要的？为什么 `instance` 必须加 `volatile`？**

**A4**：第二次判空是因为可能有多个线程同时通过了第一次判空、在 `synchronized` 外面排队等锁；第一个进入的线程创建完实例、释放锁之后，如果不再判一次空，后面排队的线程会不知道实例已经建好，重复再创建一次——第二次判空就是为了防止这种重复创建。而 `volatile` 是整个 DCL 里最容易被问倒的细节：`instance = new Singleton()` 表面上是一行代码，但 JVM 层面实际拆成三步——(1) 分配一块内存；(2) 在这块内存上执行构造函数完成初始化；(3) 把这块内存的地址赋值给 `instance` 引用。如果没有 `volatile`，编译器和 CPU 出于优化目的可能对第 (2)、(3) 步做指令重排序，变成"先赋值引用、再初始化"。这样一来，线程 A 还没构造完对象、只是把地址赋给了 `instance`，此时线程 B 进来检查 `instance == null` 已经是 `false`（因为地址已经被赋值），于是直接返回并使用这个"半成品"对象，可能触发字段全是默认值导致的空指针或者逻辑错误。`volatile` 的作用是在读写这个变量的位置插入内存屏障，禁止这种重排序，同时保证一个线程对 `instance` 的写入能立刻对其他线程可见，从根本上避免"看到非空引用但对象没初始化完成"的问题。

这条追问链完整对应 `qbank_design_patterns.py` 里的 `be-dp-01` → `be-dp-02` → `be-dp-03`，再往下一层还可以追问"除了 DCL，还有没有更简洁的写法"，答案是静态内部类（利用 JVM 类加载机制天然的线程安全，不需要 `volatile`）和枚举单例（`be-dp-04`，还能顺带防住反射攻击），可参考 `design_patterns_demo.py` 里 `Singleton` 类用 `threading.Lock` 写的 Python 版 DCL 实现和对应的并发测试。

## 追问链深挖 2：工厂模式三兄弟怎么取舍

**Q1（面试官）：简单工厂、工厂方法、抽象工厂三个模式有什么区别？**

**A1**：简单工厂（Simple Factory）是最朴素的写法——一个工厂类内部用 `if-else` 根据传入参数创建不同的产品，调用方不需要知道具体类名。它甚至不算 GoF 23 种模式之一，只是一种常见的编码习惯。

**Q2（追问）：简单工厂有什么明显缺点？**

**A2**：每新增一种产品，都要回去修改工厂内部的 `if-else` 分支，这违反了开闭原则（对扩展开放、对修改关闭）——理想情况下新增功能应该是"加代码"而不是"改代码"，改已经上线、测试过的代码天然有引入回归 bug 的风险。产品种类一多，这个工厂类的分支会变得又长又难维护。

**Q3（追问）：工厂方法模式是怎么解决这个问题的？它和抽象工厂又有什么区别？**

**A3**：工厂方法把"创建产品"这件事从一个工厂类的 `if-else`，拆成一个抽象工厂接口 + 每种产品各自一个具体工厂子类。新增产品时只需要新增一个产品类和对应的工厂子类，完全不用碰已有代码，符合开闭原则，代价是类的数量成倍增加。而抽象工厂解决的是另一个维度的问题：工厂方法一个工厂只生产"一种"产品，抽象工厂一个工厂能生产"一整族"相互配套的产品——比如一个 `MacFactory` 同时生产风格一致的 `MacButton` 和 `MacCheckbox`，一个 `WinFactory` 同时生产 `WinButton` 和 `WinCheckbox`，保证同一工厂产出的产品是配套使用的。抽象工厂新增一整个产品族（比如新增 `LinuxFactory`）很容易，但如果要新增一种产品等级（比如所有工厂都要加一个 `ScrollBar`），就得修改所有已有的具体工厂类——这时候反而不满足开闭原则了，取舍要看"你的产品是单一维度扩展，还是需要保证多个产品配套"。

对应 `be-dp-05` → `be-dp-06` → `be-dp-07`，代码验证见 `design_patterns_demo.py` 里 `ShapeFactory.create()` 的写法（简化的工厂方法风格，用注册表代替了纯 `if-else`）。

## 其余题目点行

- **观察者模式**：一对多依赖关系，主题 `notify()` 广播给所有已订阅的观察者，把"状态变化方"和"响应方"解耦。完整答案见 `qbank_design_patterns.py`（`be-dp-08`）。
- **装饰器模式**：不改原始类代码，用层层包装动态叠加职责，避免继承按功能组合数指数爆炸出子类。完整答案见 `qbank_design_patterns.py`（`be-dp-09`）。
- **策略模式**：把可互换的算法各自封装成策略类，上下文运行时切换策略，消除大段 `if-else`。完整答案见 `qbank_design_patterns.py`（`be-dp-10`）。
- **代理模式**：静态代理手写、代码量随接口线性增长；动态代理运行时反射生成，JDK 动态代理要求实现接口，CGLIB 靠继承生成子类。完整答案见 `qbank_design_patterns.py`（`be-dp-11`）。
- **SOLID 五原则**：单一职责（一个类一个变化原因）、开闭原则（对扩展开放对修改关闭）、里氏替换（子类可透明替换父类且行为不变）、接口隔离（不强迫依赖用不到的接口方法）、依赖倒置（高层低层都依赖抽象，常靠依赖注入落地）。完整答案见 `qbank_design_patterns.py`（`be-dp-12` ~ `be-dp-16`）。
- **Code Review 关注点**：正确性、可读性、设计合理性、测试覆盖、安全与性能，风格问题交给自动化 lint。完整答案见 `qbank_design_patterns.py`（`be-dp-17`）。
- **CI/CD**：持续集成解决"集成冲突发现太晚"，持续交付自动出可发布制品但人工审批上线，持续部署连上线也自动化。完整答案见 `qbank_design_patterns.py`（`be-dp-18`）。

## 易错点 / 常见误区

1. **以为加了 `synchronized` 就万事大吉**：整体加锁只解决正确性问题，面试官通常会继续追问性能，答不出 DCL 和 `volatile` 就等于只答对了一半。
2. **把 DCL 里的 `volatile` 说成"只是为了可见性"**：可见性只是一半答案，另一半、也是更容易被追问倒的点是"禁止指令重排序、防止拿到半成品对象"，两者都要说到。
3. **混淆工厂方法和抽象工厂**："一个工厂生产一种产品"还是"一个工厂生产一族产品"是两者最核心的区别，很多人只记得"抽象工厂更复杂"这种模糊印象，一追问产品族的概念就答不上来。
4. **把装饰器模式和代理模式说成一回事**：两者结构确实很像（都是持有内部对象再包一层），但装饰器强调"动态叠加职责、可以多层任意组合"，代理强调"控制访问、通常关注点是权限/缓存/远程调用这类横切逻辑"，且代理很多时候是编译期/生成期确定好的单层包装，不强调多层任意叠加。
5. **SOLID 只会背名字，说不出一句话定义**：面试官问 SOLID 通常不会要求你展开讲设计模式怎么支撑它，但连"开闭原则是什么"都说不出一句话定义会显得没有真正理解过，建议把 `be-dp-12` ~ `be-dp-16` 五张卡片当作最低门槛先过一遍。
6. **CI 和 CD 说反**：CI 是"集成"阶段解决冲突发现太晚的问题，CD（交付/部署）是"发布"阶段的自动化程度问题，持续交付和持续部署的区别在于"最后一步上线是否需要人工审批"，容易被追问混淆。
