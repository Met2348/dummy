# L03 · Reasoning Cache

## 1 · 痛点
推理 model 每次都重头 think → 哪怕同一个问题
- 用户问 "1+1?" 五次 → 五次 thinking

## 2 · 解：reasoning cache
- 把 (question_hash, reasoning_trace, answer) 存 redis
- 命中 → 直接返回，跳过 thinking
- miss → 正常流程 + 写入

## 3 · 命中策略
- 完全 match：hash
- 近似 match：embedding 相似度（要 vector DB）

## 4 · 私密性
- per-user cache（不同用户不能 share）
- 系统 prompt cache（所有用户共享）

## 5 · 实现示意
本课主要是概念，不附独立 src。
