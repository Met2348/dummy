# L06 · WebArena — 真实浏览器 agent

**Zhou et al. 2023, CMU** · arXiv 2307.13854

## 数据

- **812 任务**跨 4 类网站：
  - Shopping (OneStopShop, Magento)
  - Forum (Reddit-like)
  - Gitlab
  - Map
  - CMS (admin)
- 每任务：自然语言 goal + 网站当前状态
- 评判：**end-state 检查**（不是 action sequence）

## 例题

```
Goal: "Cancel my latest order and refund to wallet."
Current state: Shopping site, logged in
Expected end-state:
  - latest_order.status == 'cancelled'
  - wallet_balance increased by order_total
```

## Agent interface

```
observation: HTML / DOM / screenshot
action_space:
  click(element_id)
  type(element_id, text)
  scroll(direction)
  select(dropdown_id, option)
  goto(url)
```

## 评测

`end_state ?= expected` 比 action sequence 更宽松：
- 多个 action sequence 都能到 → 全算对
- 模型 hallucinate "I clicked checkout" → 状态没变 → 算错

## 分数

| 模型 | WebArena (success) |
|------|----|
| GPT-4 + WebArena agent | 14.4% |
| Claude 3 Opus | 11% |
| **GPT-4o + Browser-use** | **34%** |
| **AutoGLM-OS (智谱)** | **49%** |
| 人类 | 78% |

## 难点

1. **DOM 巨大**：一页 5000+ token
2. **动态加载**：JS 渲染
3. **长 horizon**：典型 10-30 步
4. **错误恢复**：点错了能否 backtrack

## VisualWebArena / Mind2Web 系列

- VisualWebArena (2024)：要看 screenshot 才能解
- Mind2Web (2023)：137 网站，泛化测试
- Online-Mind2Web (2025)：真实公网（动态）

## 实操

src/webarena_mock.py 1 题（add-to-cart + checkout）：

```python
from webarena_mock import run_webarena_mock
from common import make_mock_model

m = make_mock_model({"web_1":
    "go to item\nadd to cart\ngo to checkout\nconfirm order"})
rs = run_webarena_mock(m)
print(rs[0]["passed"])
```

## 一句话

> WebArena = "让 LLM 当客服" 的端到端考试。
