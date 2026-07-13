# Diffusion Mastery 使用说明

## 为什么这是老手三专题队列的收官之作

老手在完成 `onsite-mastery` 的四轮通关(792点)之后,追加要求 Foundation Model、Federated Learning、Diffusion 三个"enormous"级别专题各做100+点、难度"浅到深到社招级别"。这是队列第三个,也是最后一个,组织方式沿用前两个专题(`foundation-model-mastery`/`federated-learning-mastery`)已验证的"难度分层"轴。

## 三层怎么练

1. **Tier 1(浅,`tier1_shallow/`)**:先建立"扩散模型是什么、DDPM具体怎么训练、怎么采样出图"这套基础认知,再理解"怎么让生成结果听指令"(DDIM加速采样、Classifier-Free Guidance)。这一层的核心是把DDPM的前向/反向过程数学机制吃透,不要因为"看起来是数学"就跳过——面试追问链的第二三层经常会问到"为什么预测噪声而不是预测均值""为什么cosine schedule比linear好"这类需要理解机制而非死记结论的问题。
2. **Tier 2(深,`tier2_deep/`)**:机制深水层,三个文件分别对应扩散模型理论演化的三条主线——Score-based/SDE统一框架(把DDPM和NCSN统一起来的理论视角)、Latent Diffusion与DiT(工业界实际部署的架构选择)、Flow Matching与一致性模型(2024-2025年最活跃的采样加速前沿)。这三个文件之间有清晰的历史脉络:先有DDPM/NCSN各自发展,后被SDE框架统一,DiT是U-Net的替代方案,Flow Matching/一致性模型是对"采样太慢"这个DDPM遗留问题的两条不同解法。练习时建议留意这条脉络,而不是把三个文件当成互相独立的知识点。
3. **Tier 3(社招级别,`tier3_social_hire/`)**:无标准答案的资深判断题。第一个文件考"这套技术在生产环境里怎么真实落地"(采样方法选型、蒸馏投入、版权/内容安全这些真实存在但没有唯一正确答案的决策),第二个文件考2026年最前沿的"扩散模型+LLM/机器人/视频/多模态"交叉判断——这些方向本身还在快速演化,练的不是背答案,是"面对一个新兴、尚未成熟的技术方向,该怎么诚实评估它的成熟度、不夸大也不过度保守"这个更高阶的判断力,和 `foundation-model-mastery`/`federated-learning-mastery` 的 Tier 3 是同一种训练目标。

## 一个容易被忽视的陷阱

Tier 3 的两个文件反复强调"不要因为某个架构最近很火就无脑套用"——这是扩散模型这个领域2024-2026年特别容易踩的坑:扩散语言模型、Diffusion Policy、统一多模态架构都是真实存在的前沿方向,但每一个都有具体的适用边界和局限,练习时要留意每个 ScenarioPoint 的 `trap` 字段,那里指出的正是"看到一个亮眼demo/benchmark数字就得出过度乐观结论"这个最容易犯的错误。
