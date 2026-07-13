"""扩散模型深水区第3个文件——Flow Matching与一致性模型深水追问链(15个DeepPoint)。

覆盖:Lipman et al. 2022(《Flow Matching for Generative Modeling》,ICLR 2023)提出的Flow
Matching核心思想——不像扩散模型那样通过随机加噪的前向SDE隐式定义生成路径,而是直接design一条
概率路径(probability path)并回归它对应的向量场;Conditional Flow Matching怎么把边缘化积分
intractable的问题绕开、训练目标怎么写、和denoising score matching那套"转向条件版本"技巧的
相似性;Liu et al. 2022(《Flow Straight and Fast》,ICLR 2023 Spotlight)提出的Rectified Flow
和Flow Matching几乎同时独立提出的直线路径思想、以及reflow这个用模型自己生成的配对反复重训、
拉直路径的自举操作背后的边缘保持定理和凸序传输代价不增结论;Flow Matching和扩散模型SDE/ODE
框架的关系——高斯路径是Flow Matching允许路径集合里的一个特例、这带来的非高斯路径设计灵活性;
Flow Matching为什么训练更稳、采样更快,以及Stable Diffusion 3(Esser et al. 2024)转向这套
框架背后的实际工程动机;Song et al. 2023(《Consistency Models》,ICML 2023)提出的一致性模型
——self-consistency约束、边界条件与c_skip/c_out参数化、consistency distillation(蒸馏已有
扩散模型)和consistency training(从头训练不依赖teacher)两条训练路线的具体机制差异、以及单步
和2-4步采样的质量-速度权衡。

边界:直线插值/reflow的具体PyTorch实现代码留给flow-matching-sota/src/flow_matching.py这个
代码实现向track,这里只聚焦面试追问角度的机制推导、和扩散模型SDE/ODE理论的关系、以及真实论文
数字,不重复它的代码细节。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, grade_chain  # noqa: E402

CAT = "扩散模型深水三:Flow Matching与一致性模型深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-diff-flow-01",
        cat=CAT,
        trigger="Flow Matching据说和扩散模型不一样,不是通过随机加噪的前向过程隐式定义生成路径,而是直接指定一条'概率路径'去训练,这个'直接指定'具体是什么意思?",
        chain=(
            ("这条概率路径在数学上到底是什么样的对象?",
             "Lipman et al. 2022定义了一族随时间t在0到1之间连续变化的概率密度p_t(x),t=0对应一个简单的先验分布比如标准正态,t=1对应真实数据分布,和扩散模型不同的是,这条路径不是由某个前向加噪SDE反推出来的边缘分布,而是设计者可以自由选定的对象,只需要满足两端的边界条件。",
             ("一族随时间t在0到1之间连续变化的概率密度p_t(x)", "设计者可以自由选定的对象")),
            ("训练网络具体要拟合的是哪个量,这个量和score function有什么不同?",
             "网络要拟合的是这条路径对应的向量场u_t(x),这个向量场满足连续性方程,描述的是概率质量沿时间t流动的瞬时速度方向,而score是对数密度关于x的梯度,两者是完全不同种类的量,只是在扩散模型的特定路径选择下两者之间存在一个已知的线性变换关系。",
             ("向量场u_t(x)", "满足连续性方程")),
            ("既然路径是设计者自由选的,这个向量场是不是可以直接闭式写出来,不需要训练?",
             "对边缘路径p_t(x)本身而言,u_t(x)一般没有闭式解,因为p_t(x)是对所有可能数据点x1的条件路径做边缘化积分得到的,这个边缘化积分和扩散模型里对score做边缘化一样是intractable的,Flow Matching真正的技巧在于转向训练一个以单个数据点x1为条件的conditional vector field,这个条件版本才有闭式解。",
             ("这个边缘化积分和扩散模型里对score做边缘化一样是intractable的", "以单个数据点x1为条件的conditional vector field")),
            ("这个'边缘化积分intractable、转向条件版本回归'的技巧,是不是似曾相识?",
             "确实和Vincent 2011提出的denoising score matching思路是同一类技巧,那里也是把对边缘分布score的直接估计换成对单个数据点条件下加噪核的score做回归,Flow Matching把同样的思路从score推广到了更一般的向量场上,这也是它能兼容非高斯路径设计的原因之一。",
             ("同一类技巧", "把同样的思路从score推广到了更一般的向量场上")),
        ),
        pitfall="很多人以为Flow Matching只是'扩散模型换了个名字',说不出它和扩散模型的根本区别"
                "在于概率路径是否可以脱离某个具体前向SDE而被自由设计。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-02",
        cat=CAT,
        trigger="Conditional Flow Matching的具体训练目标怎么写,给定一对(x0,x1)是怎么算出确定的回归目标的?",
        chain=(
            ("最常见的条件路径长什么样,训练目标里的回归对象具体是什么?",
             "最常见的选择是在噪声点x0和数据点x1之间做直线插值,x_t等于(1-t)乘x0加t乘x1,这条直线的速度是常数x1减x0,训练目标就是让网络u_θ(x_t,t)去回归这个常数速度x1减x0,损失是两者的均方误差在t和(x0,x1)上的期望。",
             ("在噪声点x0和数据点x1之间做直线插值", "让网络u_θ(x_t,t)去回归这个常数速度x1减x0")),
            ("这个目标为什么说比score matching的目标'更直接'?",
             "因为对给定的一对(x0,x1)和t,回归目标x1减x0是一个确定的向量,不需要像score matching那样对真实数据分布做边缘化积分才能得到监督信号,每个训练样本自己就带着一个精确已知的目标,不需要额外近似。",
             ("回归目标x1减x0是一个确定的向量", "每个训练样本自己就带着一个精确已知的目标")),
            ("既然每对样本的目标都是精确的,那网络学到的边缘向量场为什么还是对的,会不会只是在死记硬背每一对配对?",
             "Lipman et al. 2022证明了这个conditional flow matching损失关于θ的梯度,和难以计算的边缘flow matching损失关于θ的梯度完全相同,所以最小化这个每对样本都精确已知的条件损失,等价于在优化那个原本无法直接计算的边缘向量场目标,不是死记硬背单独一对。",
             ("这个conditional flow matching损失关于θ的梯度", "等价于在优化那个原本无法直接计算的边缘向量场目标")),
        ),
        pitfall="很多人只会说'CFM就是回归x1减x0这个速度',说不出这个每对样本精确目标为什么能"
                "让网络学到正确的边缘向量场,漏掉梯度等价这个定理性支撑。",
        real_world_link="learning/flow-matching-sota/src/flow_matching.py",
    ),
    DeepPoint(
        id="dp-diff-flow-03",
        cat=CAT,
        trigger="有人说Flow Matching在训练目标上比score matching'更简单',这个'更简单'具体简单在哪里,是不是只是数学记法上好看一点?",
        chain=(
            ("具体对比一下,score matching的监督信号来自哪里,为什么处理起来更麻烦?",
             "score matching里的监督信号本质上是加噪核诱导出的denoising score,虽然对单个数据点也有闭式解,但网络最终要拟合的边缘score本身量级随噪声标准差σ剧烈变化,训练时还要额外处理噪声schedule的选择、不同σ下损失的加权,这些都是围绕如何让score这个量在不同噪声水平下数值稳定而引入的额外设计。",
             ("边缘score本身量级随噪声标准差σ剧烈变化", "如何让score这个量在不同噪声水平下数值稳定而引入的额外设计")),
            ("Flow Matching在这方面具体省掉了什么?",
             "因为训练目标直接是一条设计好的路径在时刻t的速度,只要路径本身选得足够规整比如直线插值,这个速度沿t的量级变化很小甚至恒定,不需要单独设计一套类似噪声schedule的机制去平衡不同t下的损失量级,也不需要选择方差参数化。",
             ("这个速度沿t的量级变化很小甚至恒定", "也不需要选择方差参数化")),
            ("这种'量级更稳定'的说法有没有代价,是不是意味着Flow Matching什么路径都行、不用调参了?",
             "并不是,路径的具体形状比如直线插值还是曲线插值、噪声端的方差取多大,仍然会显著影响训练难度和采样步数,只是这些选择被搬到了路径设计这一个更透明的环节,而不是像score matching那样分散在噪声schedule、方差参数化、损失加权这几处互相耦合的设计里。",
             ("被搬到了路径设计这一个更透明的环节", "分散在噪声schedule、方差参数化、损失加权这几处互相耦合的设计里")),
        ),
        pitfall="很多人以为Flow Matching'更简单'等于'不需要调参',说不出它只是把设计自由度"
                "集中到了路径形状这一处,而不是彻底消除了设计难度。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-04",
        cat=CAT,
        trigger="Flow Matching和扩散模型的probability flow ODE到底是什么关系,是两套独立的理论还是一个包含另一个?",
        chain=(
            ("先说结论性的关系是什么?",
             "Flow Matching可以被看成probability flow ODE框架的一个更一般化的版本,扩散模型的概率路径,也就是VP-SDE、VE-SDE对应的那些边缘分布,只是Flow Matching允许的高斯概率路径家族里的一个特例。",
             ("Flow Matching可以被看成probability flow ODE框架的一个更一般化的版本", "只是Flow Matching允许的高斯概率路径家族里的一个特例")),
            ("具体来说,扩散模型的路径是哪一种高斯路径,Flow Matching又多允许了什么?",
             "扩散模型的路径由前向加噪SDE的解析边缘分布决定,均值和方差都要满足信号衰减加噪声累积这套约束,而Flow Matching里的条件高斯路径,均值和方差可以是任意随t变化的函数,只要满足t=0是噪声、t=1是数据这两个边界条件,比如直线插值对应的均值线性变化、方差单调收缩到0,这在扩散模型的SDE推导里是得不到的一种路径形状。",
             ("均值和方差都要满足信号衰减加噪声累积这套约束", "在扩散模型的SDE推导里是得不到的一种路径形状")),
            ("除了高斯路径内部的形状自由,Flow Matching理论上还允许更极端的灵活性吗?",
             "允许,论文的框架本身并不要求路径必须是高斯的,后续工作把这套框架推广到黎曼流形、离散状态空间上的非高斯概率路径,这些路径不再有解析的均值方差描述,但依然可以用同一套conditional vector field回归的思路训练,这是扩散模型的SDE框架里没有对应物的。",
             ("推广到黎曼流形、离散状态空间上的非高斯概率路径", "扩散模型的SDE框架里没有对应物")),
        ),
        pitfall="很多人以为Flow Matching和扩散SDE是两套并列的理论,说不出前者是后者的严格"
                "推广、高斯路径只是被允许的路径集合里的一个特例。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-05",
        cat=CAT,
        trigger="Rectified Flow和Flow Matching讲的好像是同一件事,这两篇论文到底是什么关系,是谁抄了谁?",
        chain=(
            ("两者的核心思想具体是怎么重合的?",
             "Liu et al. 2022的Rectified Flow和Lipman et al. 2022的Flow Matching几乎同时独立提出,都是用噪声点和数据点之间的直线插值来定义一条确定性的ODE路径,训练一个网络去回归这条直线的速度,两篇论文的训练目标在数学形式上高度一致。",
             ("Liu et al. 2022的Rectified Flow和Lipman et al. 2022的Flow Matching几乎同时独立提出", "训练一个网络去回归这条直线的速度")),
            ("既然核心训练目标几乎一样,两篇论文各自的独特贡献是什么?",
             "Flow Matching那篇的贡献偏理论统一,提出了适用于任意条件概率路径的一般化框架,并证明了条件损失和边缘损失梯度相同这个定理,而Rectified Flow那篇更聚焦直线路径这一个具体选择,额外提出了reflow这个可以反复迭代、把路径越拉越直的自举操作。",
             ("提出了适用于任意条件概率路径的一般化框架", "额外提出了reflow这个可以反复迭代、把路径越拉越直的自举操作")),
            ("这种'两组人几乎同时独立发现同一个核心思想'的情况,给你什么启示?",
             "说明直线插值加速度回归这套思路在当时已经是这个方向上足够自然的下一步,几乎同期还有stochastic interpolants这类工作从随机过程插值的角度得到了类似结论,面试里被问到这类问题时应该讲清楚是哪个具体子贡献对应哪篇论文,而不是笼统地说'都一样'。",
             ("stochastic interpolants这类工作从随机过程插值的角度得到了类似结论", "而不是笼统地说'都一样'")),
        ),
        pitfall="很多人会把Rectified Flow和Flow Matching混为一谈,说不出reflow是Rectified Flow"
                "这篇论文独有的贡献、Flow Matching的独有贡献是任意条件路径的一般化框架。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-06",
        cat=CAT,
        trigger="Rectified Flow里的reflow操作具体是怎么把路径'拉直'的,为什么反复做这个操作路径会越来越直?",
        chain=(
            ("reflow具体的操作步骤是什么?",
             "先用随机配对的噪声点x0和数据点x1训练出第一版速度场,再用这个训好的模型把一批新的x0沿着学到的ODE轨迹积分,生成对应的终点x1',这样就得到了一批不再是随机配对、而是模型自己流出来的(x0,x1')配对,再拿这些新配对重新训练一个网络,这就是一次reflow。",
             ("再用这个训好的模型把一批新的x0沿着学到的ODE轨迹积分", "拿这些新配对重新训练一个网络,这就是一次reflow")),
            ("为什么用模型自己生成的配对重训,路径就会变直,这背后有没有理论保证?",
             "论文证明了reflow这个操作会把任意一个初始耦合变成一个新的确定性耦合,并且这个新耦合对任意凸的传输代价函数,传输代价都不会比原来的耦合更大,这个'对所有凸代价函数同时不增'的性质保证了新配对之间的直线互相交叉的情况变少,连接它们的直线路径因此更直。",
             ("对任意凸的传输代价函数,传输代价都不会比原来的耦合更大", "连接它们的直线路径因此更直")),
            ("这个重训过程有没有边界条件,也就是说reflow之后,模型是不是还在噪声分布和数据分布这两个端点之间转换?",
             "有边界条件,论文的定理同时证明了reflow前后x0和x1各自的边缘分布保持不变,也就是说这个操作只改变了噪声和数据之间具体怎么配对连线,不改变两端各自的分布本身,所以可以放心递归地做下去,得到1-rectified flow、2-rectified flow这样不断拉直的序列。",
             ("reflow前后x0和x1各自的边缘分布保持不变", "得到1-rectified flow、2-rectified flow这样不断拉直的序列")),
            ("反复reflow到最后,实际论文里报告的效果具体是什么样的数字?",
             "论文和官方实现仓库报告,经过若干轮reflow并配合蒸馏之后,2-rectified flow在CIFAR-10上做到一步生成就能达到FID 4.85、recall 0.51,这个水平已经明显超过同期其他快速采样的扩散模型和GAN基线。",
             ("2-rectified flow在CIFAR-10上做到一步生成就能达到FID 4.85、recall 0.51", "已经明显超过同期其他快速采样的扩散模型和GAN基线")),
        ),
        pitfall="很多人以为reflow只是'多训练几轮'这种工程重复,说不出它背后有边缘分布保持"
                "不变、传输代价对所有凸函数同时不增这两个具体的理论保证。",
        real_world_link="learning/flow-matching-sota/src/flow_matching.py",
    ),
    DeepPoint(
        id="dp-diff-flow-07",
        cat=CAT,
        trigger="既然reflow靠迭代拉直路径来减少采样步数,那是不是意味着reflow次数越多、路径就必然会收敛到最优传输意义下的最短路径?",
        chain=(
            ("reflow保证的'传输代价不增'是不是就等于'最优传输'?",
             "不完全等于,reflow保证的是新耦合相比旧耦合,对每一个凸传输代价函数同时不增,这是一个比单一固定代价函数下的最优传输更强也更弱的性质,它不针对某一个具体代价函数去找全局最优解,而是同时改善一整族凸代价,所以reflow本身并不保证收敛到某个具体代价比如二次代价下的Monge最优传输映射。",
             ("对每一个凸传输代价函数同时不增", "并不保证收敛到某个具体代价比如二次代价下的Monge最优传输映射")),
            ("那在什么条件下reflow得到的耦合才会等于真正的最优传输映射?",
             "只有在一维情形下,reflow得到的确定性耦合被证明就是唯一的单调最优传输耦合,在更高维的一般情形下,reflow只保证单调改善任意初始耦合,并不天然具备最优传输映射所要求的那种梯度场或者说potential函数结构,除非额外施加这类结构性约束。",
             ("reflow得到的确定性耦合被证明就是唯一的单调最优传输耦合", "并不天然具备最优传输映射所要求的那种梯度场或者说potential函数结构")),
            ("这个区分在实际训练里会带来什么后果?",
             "后果是reflow次数增多确实能持续拉直路径、减少采样步数,但不能简单假设'reflow到收敛就自动是最优传输',如果任务真的需要严格的最优传输映射,需要在reflow之外额外引入针对特定代价函数的约束或者专门的单目标最优传输方法。",
             ("不能简单假设'reflow到收敛就自动是最优传输'", "需要在reflow之外额外引入针对特定代价函数的约束或者专门的单目标最优传输方法")),
        ),
        pitfall="很多人把'reflow拉直路径'和'reflow等价于求解最优传输'划等号,说不出两者"
                "只在一维情形下重合、高维情形下reflow只保证凸代价不增而非最优传输。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-08",
        cat=CAT,
        trigger="为什么Flow Matching在训练稳定性和采样速度上通常被认为优于经典DDPM,这个'通常更好'具体是靠什么机制实现的?",
        chain=(
            ("先说训练稳定性这一侧,具体机制是什么?",
             "DDPM的训练目标需要在一整套预先设定的噪声schedule和方差参数化选择之下工作,不同t的损失量级、信噪比差异很大,需要额外的加权策略去平衡,而Flow Matching如果选用直线插值这类路径,回归目标在整个t区间上量级稳定,不需要再单独设计这套加权和参数化去补救。",
             ("需要额外的加权策略去平衡", "不需要再单独设计这套加权和参数化去补救")),
            ("采样速度这一侧呢,机制上具体在哪里体现?",
             "采样本质是数值求解一条从噪声到数据的ODE或者SDE轨迹,轨迹越弯曲,数值求解器为了不'切弯'跟丢真实路径就需要越多步,DDPM对应的VP-SDE路径形状由固定的信号衰减和噪声累积规律决定,通常比较弯曲,而Flow Matching允许直接设计接近直线的路径,数值积分自然需要更少步数就能保持精度。",
             ("轨迹越弯曲,数值求解器为了不'切弯'跟丢真实路径就需要越多步", "数值积分自然需要更少步数就能保持精度")),
            ("这是不是说明Flow Matching在理论上就一定比DDPM好,DDPM已经过时了?",
             "不能这么说,DDPM对应的高斯路径只是Flow Matching允许路径集合里的一种选择,两者的优劣本质上是路径形状选择的优劣,而不是两套训练框架本身谁更先进,DDPM之所以显得更弯,是因为它的路径形状是由信号衰减和噪声累积这套物理直觉决定的,而不是专门为了压低轨迹曲率而设计的。",
             ("而不是两套训练框架本身谁更先进", "而不是专门为了压低轨迹曲率而设计的")),
        ),
        pitfall="很多人只会背'flow matching更快更稳'这个结论,说不出这个优势的根源是路径形状"
                "可以被专门设计成低曲率,而DDPM的路径形状是被信号衰减规律决定、并非为降低曲率而设计。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-09",
        cat=CAT,
        trigger="2024到2025年像Stable Diffusion 3这样的前沿系统转向flow matching,实际的工程动机是什么,只是为了理论上更漂亮吗?",
        chain=(
            ("Esser et al. 2024这篇论文具体做了什么工作?",
             "这篇论文把rectified flow形式的训练目标和多种既有的扩散模型公式、损失加权方式做了一次大规模的系统对比,并提出了一套新的timestep采样schedule,让训练时更多地采样对感知质量更关键的中间噪声水平,而不是均匀采样所有t。",
             ("和多种既有的扩散模型公式、损失加权方式做了一次大规模的系统对比", "让训练时更多地采样对感知质量更关键的中间噪声水平")),
            ("这套新的timestep schedule解决的是什么问题?",
             "均匀采样t会让训练把大量算力花在对最终样本质量影响很小的高噪声或者低噪声区间上,通过把timestep的采样分布往对感知质量更重要的中间区间偏移,可以在同样的训练算力预算下让最终生成质量更好,这个改进和路径本身是不是直线是两个独立的贡献,前者是加权策略,后者是路径形状。",
             ("把timestep的采样分布往对感知质量更重要的中间区间偏移", "前者是加权策略,后者是路径形状")),
            ("所以转向flow matching在工程上的实际动机,应该怎么概括?",
             "动机是在大规模训练和高分辨率生成这个具体场景下,直线路径带来的采样步数下降和这套针对性的timestep重加权带来的训练效率提升可以叠加,论文的大规模对比证明了这套组合在同等算力下优于传统扩散公式,而不是单纯因为flow matching的数学形式更简洁。",
             ("直线路径带来的采样步数下降和这套针对性的timestep重加权带来的训练效率提升可以叠加", "而不是单纯因为flow matching的数学形式更简洁")),
        ),
        pitfall="很多人以为SD3选flow matching纯粹是'跟风前沿理论',说不出Esser et al. 2024"
                "真正验证的是大规模系统对比加新timestep schedule这套具体的工程组合,而不是路径形状本身。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-10",
        cat=CAT,
        trigger="Consistency Models里说的self-consistency约束具体是什么,和前面讲的probability flow ODE有什么关系?",
        chain=(
            ("self-consistency这个约束具体怎么表述?",
             "Song et al. 2023定义了一个consistency function,输入是probability flow ODE轨迹上任意一点x_t和它对应的时间t,要求这个函数对同一条轨迹上任意两个不同时刻t和t'给出的输出必须相等,也就是f_θ(x_t,t)等于f_θ(x_t',t'),这个约束把'同一条轨迹上的点都应该被映射到同一个终点'变成了一个可以直接写进损失函数的等式。",
             ("要求这个函数对同一条轨迹上任意两个不同时刻t和t'给出的输出必须相等", "同一条轨迹上的点都应该被映射到同一个终点")),
            ("这个函数具体要映射到轨迹的哪个终点,起点t趋近0的地方有什么特殊要求?",
             "轨迹的终点被定义在时间趋近0也就是接近干净数据的一端,论文额外要求一个边界条件,当t等于最小时间ε时consistency function必须满足f_θ(x_ε,ε)等于x_ε本身,也就是在几乎无噪声的点上,函数必须退化成恒等映射。",
             ("当t等于最小时间ε时consistency function必须满足f_θ(x_ε,ε)等于x_ε本身", "函数必须退化成恒等映射")),
            ("这个边界条件在网络架构上具体是怎么被强制满足的?",
             "网络不直接输出f_θ,而是用一个skip connection式的参数化,f_θ(x,t)等于c_skip(t)乘x加c_out(t)乘一个自由网络F_θ(x,t)的输出,其中c_skip(ε)取1、c_out(ε)取0,这样在t等于ε处网络的输出自动就等于x,边界条件不需要额外的损失项去逼近,而是由架构直接保证。",
             ("c_skip(ε)取1、c_out(ε)取0", "边界条件不需要额外的损失项去逼近,而是由架构直接保证")),
        ),
        pitfall="很多人只会说'一致性模型能一步生成',说不出self-consistency这个约束的精确"
                "数学形式、以及边界条件是靠c_skip/c_out这种参数化在架构层面强制满足的。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-11",
        cat=CAT,
        trigger="Consistency Distillation具体是怎么把一个已经训好的扩散模型蒸馏成一致性模型的?",
        chain=(
            ("蒸馏过程里,已经训好的扩散模型扮演什么角色,具体怎么用它生成训练信号?",
             "已经训好的扩散模型被当作teacher,用来估计probability flow ODE的漂移方向,训练时先在轨迹上采样一个时刻t加1对应的点,再用teacher的score配合一个数值ODE求解器,往前走一步得到时刻t对应的相邻点,这一步只需要teacher、不需要真实数据的标签。",
             ("用来估计probability flow ODE的漂移方向", "这一步只需要teacher、不需要真实数据的标签")),
            ("有了这一对相邻时刻的点,consistency distillation的损失具体怎么算?",
             "让学生网络在这对相邻点上的输出保持一致,也就是最小化f_θ在t加1时刻的点上的输出,和f_θ负在t时刻的点上的输出之间的距离,其中θ负是θ的滑动平均,这样目标网络更新更平滑,训练更稳定。",
             ("最小化f_θ在t加1时刻的点上的输出", "其中θ负是θ的滑动平均,这样目标网络更新更平滑")),
            ("论文里报告的这套best配置和最终效果具体是什么样的?",
             "论文实验里consistency distillation效果最好的配置是用LPIPS作为距离度量、用Heun二阶方法做ODE求解器、离散化步数N取18,在这套配置下consistency distillation在CIFAR-10上做到FID 3.55、在ImageNet 64x64上做到FID 6.20,当时是蒸馏类方法里少步生成的新纪录。",
             ("consistency distillation效果最好的配置是用LPIPS作为距离度量、用Heun二阶方法做ODE求解器、离散化步数N取18", "在CIFAR-10上做到FID 3.55、在ImageNet 64x64上做到FID 6.20")),
        ),
        pitfall="很多人以为consistency distillation只是'普通的知识蒸馏套个新名字',说不出"
                "它依赖teacher的score配合数值ODE求解器生成相邻点、损失是让EMA目标网络在相邻点上输出一致这个具体机制。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-12",
        cat=CAT,
        trigger="Consistency Training和Consistency Distillation最本质的区别是什么,前者是不是完全不需要扩散模型这个概念了?",
        chain=(
            ("Consistency Training具体是怎么在没有teacher的情况下获得训练信号的?",
             "Consistency Training不依赖任何预训练的扩散模型,而是直接用前向加噪过程本身构造相邻噪声水平的点对,用一个对真实score的无偏估计量去替代teacher原本提供的ODE求解方向,整个训练自成一体,不需要事先跑一个独立的扩散模型训练阶段。",
             ("用一个对真实score的无偏估计量去替代teacher原本提供的ODE求解方向", "不需要事先跑一个独立的扩散模型训练阶段")),
            ("既然不依赖teacher提供的ODE求解轨迹,Consistency Training在离散化步数和EMA上具体怎么处理?",
             "论文没有把离散化步数N和EMA衰减率固定成常数,而是让它们随训练迭代次数按一个递增的schedule调整,训练早期N较小、后期N逐渐变大,这个自适应的schedule被证明能让Consistency Training收敛得明显更快。",
             ("让它们随训练迭代次数按一个递增的schedule调整", "这个自适应的schedule被证明能让Consistency Training收敛得明显更快")),
            ("既然不依赖Heun这类高阶ODE求解器,Consistency Training在求解器选择上是不是也有讲究?",
             "论文特别指出Consistency Training不需要像Consistency Distillation那样依赖Heun二阶方法,因为它的损失函数本身不依赖某一条具体的PF ODE数值轨迹,而Consistency Distillation的损失是直接建立在teacher给出的那条特定数值轨迹之上的,对求解器精度更敏感。",
             ("因为它的损失函数本身不依赖某一条具体的PF ODE数值轨迹", "对求解器精度更敏感")),
        ),
        pitfall="很多人以为Consistency Training就是'去掉teacher之外别的都一样',说不出它在"
                "离散化步数schedule、求解器依赖程度上和Consistency Distillation有实质区别。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-13",
        cat=CAT,
        trigger="一致性模型既然能一步生成,为什么论文里还会讨论2到4步这种多步采样,一步不是最快吗?",
        chain=(
            ("多步采样具体是怎么在一致性模型上做的?",
             "多步采样在一步生成的结果上先人为地重新加回一定量的噪声,把样本推回到轨迹上更靠后的一个时刻,再用同一个consistency function对这个新的带噪点重新映射回终点,如此交替进行若干次,每一次交替都相当于对上一步的结果做一次精修。",
             ("先人为地重新加回一定量的噪声", "每一次交替都相当于对上一步的结果做一次精修")),
            ("这种加噪再映射的交替,为什么能提升质量而不是纯粹浪费计算?",
             "因为一步生成完全依赖consistency function在单个前向传播里对整条轨迹做的近似,近似总会有误差,而多次交替相当于给模型多次机会去修正上一轮映射里残留的偏差,用增加少量额外的网络前向次数换取更接近teacher真实分布的样本质量。",
             ("用增加少量额外的网络前向次数换取更接近teacher真实分布的样本质量", "近似总会有误差")),
            ("这样看,一致性模型的采样开销和DDPM比处于什么量级,这个权衡具体怎么描述?",
             "一致性模型的采样步数通常在1到4步这个量级,而原始DDPM动辄需要几十到上千步的迭代去噪,一致性模型是用单次前向传播里更强的自洽性约束换来的步数量级下降,代价是这套self-consistency约束本身的训练比标准扩散模型的训练更复杂。",
             ("原始DDPM动辄需要几十到上千步的迭代去噪", "代价是这套self-consistency约束本身的训练比标准扩散模型的训练更复杂")),
        ),
        pitfall="很多人以为一致性模型'既然能一步生成,步数越少必然越好',说不出多步采样是"
                "用加噪-重映射的交替去修正单步近似误差、以此换取更高质量这个具体权衡。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-14",
        cat=CAT,
        trigger="后续的Improved Techniques这篇论文对consistency training做了什么改进,是不是彻底不需要蒸馏和teacher了?",
        chain=(
            ("这篇后续工作具体在原始Consistency Training基础上改了什么?",
             "这篇后续工作进一步调整了离散化步数N和EMA衰减率随训练进程变化的具体schedule形式,让consistency training在完全不依赖任何预训练扩散模型、不做蒸馏的前提下,单独训练也能达到和当时最好的蒸馏方法相当甚至更好的效果。",
             ("单独训练也能达到和当时最好的蒸馏方法相当甚至更好的效果", "调整了离散化步数N和EMA衰减率随训练进程变化的具体schedule形式")),
            ("具体的效果数字体现在哪里?",
             "论文报告在两步生成的设置下,改进后的consistency training在CIFAR-10上做到FID 2.24、在ImageNet 64x64上做到FID 2.77,这两个数字都超过了原始consistency distillation论文里报告的一步和两步结果。",
             ("改进后的consistency training在CIFAR-10上做到FID 2.24、在ImageNet 64x64上做到FID 2.77", "这两个数字都超过了原始consistency distillation论文里报告的一步和两步结果")),
            ("这是不是意味着蒸馏这条路线已经没有价值了,以后都应该直接用consistency training?",
             "不能这么下结论,这篇改进工作证明的是consistency training这一条独立训练路线的天花板可以被推得更高,但蒸馏路线在有现成高质量扩散teacher可用的场景下依然更省训练成本,两条路线各自适用的场景不同,不存在一条路线整体淘汰另一条的结论。",
             ("这一条独立训练路线的天花板可以被推得更高", "两条路线各自适用的场景不同,不存在一条路线整体淘汰另一条的结论")),
        ),
        pitfall="很多人看到改进后的consistency training效果反超原始蒸馏结果,就直接得出"
                "'蒸馏路线过时了'的结论,说不出两条路线在训练成本和适用场景上依然有各自的取舍。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-diff-flow-15",
        cat=CAT,
        trigger="如果把DDIM、rectified flow、consistency models这三种少步采样方法放在一起比较,你会怎么给面试官讲清楚它们的思路差异,而不是笼统地说'都是加速采样'?",
        chain=(
            ("先说这三者共享的理论基础是什么?",
             "三者都建立在probability flow ODE这同一个确定性轨迹的概念之上,一个训练好的扩散模型或者速度场都对应着一条从噪声到数据的确定性ODE曲线,三种方法的差别在于怎么处理这条曲线,而不是各自发明了三套互不相关的理论。",
             ("三者都建立在probability flow ODE这同一个确定性轨迹的概念之上", "三种方法的差别在于怎么处理这条曲线")),
            ("那具体每一种方法各自的处理思路是什么?",
             "DDIM是把已有的VP-SDE对应的probability flow ODE换一种关于噪声尺度的离散化方式来跳步,曲线的形状本身没有变;rectified flow是用reflow反复重训、直接改变曲线本身让它变直,从而少步积分也能保持精度;consistency models则完全绕开数值积分曲线这件事,直接训练一个映射从曲线上任意一点跳到终点。",
             ("直接改变曲线本身让它变直,从而少步积分也能保持精度", "直接训练一个映射从曲线上任意一点跳到终点")),
            ("这样区分之后,面试官追问'这三者能不能组合使用',你会怎么答?",
             "可以组合,比如rectified flow产出的更直的路径可以再被consistency distillation当作teacher轨迹去蒸馏,因为路径越直,teacher的numerical ODE求解器给出的相邻点误差越小,consistency distillation的训练信号也就越准,这也是InstaFlow这类后续工作实际采用的组合思路。",
             ("因为路径越直,teacher的numerical ODE求解器给出的相邻点误差越小", "这也是InstaFlow这类后续工作实际采用的组合思路")),
            ("最后如果被问到'这三者有没有共同的局限',你会怎么收尾?",
             "三者都是在缓解采样步数这个工程瓶颈,但都没有解决扩散和flow matching模型训练阶段本身的统计难题,比如高维数据下向量场或者score的估计误差、以及数据分布本身的多模式覆盖问题,少步采样能力和生成质量的理论上限依然由底层训练目标的估计精度决定。",
             ("但都没有解决扩散和flow matching模型训练阶段本身的统计难题", "少步采样能力和生成质量的理论上限依然由底层训练目标的估计精度决定")),
        ),
        pitfall="很多人把DDIM、rectified flow、consistency models笼统地归为'加速采样的"
                "trick合集',说不出三者分别是'换离散化方式''改变曲线本身''绕开曲线直接学映射'"
                "这三种不同的处理思路,也说不出它们可以组合使用。",
        real_world_link="",
    ),
]


def _self_test() -> None:
    assert 13 <= len(BANK) <= 17, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复id"
    assert all(i.startswith("dp-diff-flow-") for i in ids), "id前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的条目"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的条目"
    assert all(dp.trigger for dp in BANK), "存在缺失trigger的条目"
    triggers = [dp.trigger for dp in BANK]
    assert len(triggers) == len(set(triggers)), "存在重复trigger"
    for dp in BANK:
        answers = [ref for (_q, ref, _k) in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 采分关键词未能在参考答案里全部命中: {scores}"
    print(f"[PASS] dp_diffusion_flow_matching_consistency: {len(BANK)}个DeepPoint 自洽性检查通过")


if __name__ == "__main__":
    _self_test()
