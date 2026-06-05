# L07 · Computer Use（Anthropic 2024.10）

## 30 秒核心

> Computer Use = **screenshot + mouse/keyboard action** —— 让 LLM 像人一样操作电脑。

Anthropic 2024.10 公测，Claude 3.5 Sonnet 内置能力。

## 流程

```
1. Screenshot 当前屏幕 → 给 LLM 看
2. LLM 输出 action: {"type":"click","coordinate":[x,y]}
                  | {"type":"type","text":"hello"}
                  | {"type":"key","key":"Return"}
                  | {"type":"screenshot"}
3. Controller 执行 action
4. (loop) → 直到任务完成
```

## API 设计

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    tools=[{
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
    }],
    messages=[
        {"role":"user","content":"Open Firefox and search for ReAct paper"}
    ],
)
```

LLM 自动开始 screenshot + click + type loop。

## Action 类型

| Type | 参数 |
|------|------|
| screenshot | — |
| left_click | coordinate=[x,y] |
| right_click | coordinate |
| middle_click | coordinate |
| double_click | coordinate |
| type | text |
| key | key (e.g. "Return", "ctrl+c") |
| cursor_position | — |
| mouse_move | coordinate |

## 性能现状（OSWorld benchmark）

| Model | OSWorld score |
|-------|--------------:|
| Claude 3.5 Sonnet (2024.10) | 14% |
| Claude 3.7 + extended thinking | 22% |
| **AutoGLM-OS** | **48.9%** |
| Claude 4 (2025) | ~35% |
| 人类 | 72% |

→ 还远未达人类水平，但 2024-2025 提升快。

## 安全考虑

| 风险 | 缓解 |
|------|------|
| Click 删除文件 | confirm before destructive |
| 自动填表 | 用户确认 |
| 远程控制泄露 | sandboxed VM |
| Prompt injection（看到网页恶意 prompt） | screenshot 内容 risk |

Anthropic 强烈推荐：**先在虚拟机里跑**。

## 工程套路（生产）

```
- Container/VM 隔离
- Action allowlist (禁系统设置 / shell)
- Screenshot diff 验证 action 效果
- Periodic state validation
- HITL on irreversible
```

## 实现 (`computer_use_mock.py` 预告)

```python
class MockComputer:
    def __init__(self):
        self.cursor = [0, 0]
        self.screen_text = ""
        self.action_log = []

    def execute(self, action: dict) -> dict:
        a_type = action["type"]
        if a_type == "screenshot":
            return {"image": self._render(), "cursor": self.cursor}
        if a_type == "left_click":
            self.cursor = action["coordinate"]
            self.action_log.append(("click", self.cursor))
            return {"ok": True}
        if a_type == "type":
            self.screen_text += action["text"]
            return {"ok": True}
        return {"error": f"unknown {a_type}"}
```

## 类比：Playwright / Selenium

```
| Computer Use | Playwright |
|--------------|------------|
| LLM 看截图  | 程序读 DOM |
| 像素坐标    | CSS selector |
| 模糊但通用  | 准确但需 selector |
```

Computer Use 是"无 selector 的浏览器自动化"。

## 退出条件

- 能列 5 action type
- 知道 OSWorld 数字
- 知道 sandbox 是 best practice

## 一句话

> Computer Use = screenshot + mouse/keyboard action — Claude 直接操作屏幕，OSWorld 48.9% 是 2025 SOTA。
