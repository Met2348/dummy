# Diffusion Mastery —— 扩散模型专题(浅→深→社招级别)

> 定位:老手要求 Foundation Model、Federated Learning、Diffusion 三个"enormous"级别专题各做100+点(浅到深到社招级别),这是队列第三个、也是最后一个。

## 组织轴:难度分层(与 `foundation-model-mastery`/`federated-learning-mastery` 一致)

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | DDPM前向反向过程核心机制、采样与引导(DDIM/CFG)基础 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | Score-based/SDE统一框架、Latent Diffusion与DiT架构、Flow Matching与一致性模型 | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 生产部署判断、Diffusion for LLM/多模态前沿判断 | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,练法建议按 Tier 1→2→3 顺序推进。

## 与仓库已有 diffusion 相关学术 notebook portfolio 的边界

仓库里已有 `diffusion-foundations/`、`diffusion-language-models/`、`dit-latent-diffusion/`、`flow-matching-sota/` 这几个目录——它们是"实现深度"的学术notebook portfolio(可跑的PyTorch代码+讲义),而本track是"面试问答"的DeepPoint/ScenarioPoint题库,两者性质不同、不是重复内容。部分DeepPoint的`real_world_link`字段引用了这些portfolio目录下真实存在的具体文件(比如`learning/dit-latent-diffusion/src/dit.py`),已逐一核实存在。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_diffusion_forward_reverse_ddpm.py` | DDPM前向马尔可夫链、重参数化技巧、反向过程与噪声预测、变分下界简化、VAE/GAN对比、noise schedule(linear/cosine) | 18 |
| `tier1_shallow/dp_diffusion_sampling_guidance_basics.py` | DDPM采样步数问题、DDIM确定性跳步采样、Classifier Guidance、Classifier-Free Guidance、guidance scale权衡 | 17 |
| `tier2_deep/dp_diffusion_score_sde.py` | Score-based生成模型、denoising score matching、annealed Langevin dynamics、VE/VP-SDE统一框架、probability flow ODE、predictor-corrector采样 | 15 |
| `tier2_deep/dp_diffusion_latent_dit.py` | LDM压缩空间设计、VAE自编码器、U-Net跳跃连接、DiT patchify架构、adaLN-Zero条件注入、Gflops-FID scaling关系 | 15 |
| `tier2_deep/dp_diffusion_flow_matching_consistency.py` | Flow Matching核心思想、Conditional Flow Matching、Rectified Flow与reflow、一致性模型(consistency distillation/training) | 15 |
| `tier3_social_hire/sc_diffusion_production_deployment_judgment.py` | 采样方法选型判断、蒸馏时机判断、guidance scale产品化、版权合规判断、内容安全多层防御、水印溯源诚实边界 | 14 |
| `tier3_social_hire/sc_diffusion_llm_multimodal_frontier_judgment.py` | 扩散语言模型场景判断、Diffusion Policy机器人控制、视频生成时间一致性、统一多模态架构判断、2026年交叉领域成熟度诚实评估 | 14 |

与已有八子包(全局id/trigger程序化验证零冲突)合计1115点。

## 数据结构:复用已验证的 DeepPoint + ScenarioPoint

```python
import sys
sys.path.insert(0, "learning/diffusion-mastery/src")
from diffusion_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/diffusion-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部来自各 agent 自主 WebSearch 验证的真实论文(DDPM/DDIM/Classifier-Free Guidance/Score-SDE/LDM/DiT/Flow Matching/Rectified Flow/Consistency Models)和真实工程案例(Getty Images v. Stability AI、LAION-5B CSAM争议、Mercury/Gemini Diffusion/Diffusion Policy等2026年真实进展)。写作过程中修正过一处格式不一致(全角冒号统一为半角),`real_world_link`字段逐一核实——凡是无法验证真实存在的本地文件引用一律留空,不留虚假路径。

至此,老手要求的 Foundation Model、Federated Learning、Diffusion 三个100+点专题队列全部完成。
