# 研究决策文档规范

## 总原则

重要研究判断必须存在于有版本的文件中，不能只留在聊天、幻灯片或个人记忆里。

## 两层结构

1. `decisions/`：保存持久决策、证据、备选方案、决策门和后果。
2. `notes/`：保存文献细节、计算、工作分析和实验诊断，为决策提供依据。

## 更新纪律

- 中心问题、novelty、主实验、目标 venue 或 go/pivot/stop 发生实质变化时，新建编号记录。
- 旧记录标为 `superseded`，不得删除或静默改写原始理由。
- 新证据只改变评分、不改变决策时，在原记录的变更日志追加日期和依据。
- 每个 go/pivot/stop 必须链接 experiment artifacts 或文献证据。
- 明确区分：项目观察、文献结论、未来预测和主观概率。
- 概率必须写清条件事件并给范围，不使用虚假的小数点精度。
- 未完成实验不得通过措辞升级为已有证据。

## 必须新建 Decision Record 的情况

- 改变中心研究问题或主 novelty claim；
- 改变 target venue、deadline 或投稿策略；
- 改变实验单位、primary outcome、target split、patch 或统计模型；
- 改变 model、benchmark、budget scope；
- 新直接竞争论文改变 collision boundary；
- feasibility 或 acceptance probability 发生实质变化；
- 进入或退出 go/pivot/stop 状态。

## 强制复审节点

- 72 小时 kill test 之后；
- target seal 之前；
- sealed target 评分之后；
- 正文截止前五天；
- 出现任何直接碰撞论文之后；
- runner、endpoint 或预算假设失效时。

## 文档语言

面向导师与研究决策的人类可读文档遵循 [中文文档规范](document-language-policy.md)。

