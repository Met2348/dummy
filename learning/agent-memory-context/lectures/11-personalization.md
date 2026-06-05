# L11 · 个性化 Memory

## 30 秒

> Personalization = 把用户的**长期偏好、风格、知识层级**编码进 memory，让 agent 因人而异。

## 4 类个性化信号

| 类 | 例 |
|----|---|
| **Preferences** | "偏好 concise 答案" |
| **Knowledge level** | "熟悉 Transformer 但不熟悉 RL" |
| **Communication style** | "中文、技术、详细" |
| **Domain context** | "工程师，做 RAG 项目" |

## 收集方式

| 方式 | 例 |
|------|---|
| Explicit (用户设置) | "Please be concise" 进 settings |
| Implicit (LLM extract) | 从对话推断风格 |
| Behavioral (action) | 用户跳过解释→偏 concise |
| Feedback (thumbs up/down) | 答案被 like → 风格强化 |

## ChatGPT Memory (2024)

OpenAI 引入 implicit memory：
- "Remember that I prefer dark mode"
- 自动跨对话 carry
- 用户可 view / delete

## Anthropic Style Preference (Claude 2025)

Claude Desktop 让用户写 "How should Claude respond?"：
- 角色、格式偏好、语气
- 跨对话 carry
- prompt cache hit 100%

## 个性化 prompt 模板

```
SYSTEM: You are an AI assistant.

USER PROFILE:
- Name: Alice
- Role: ML Engineer
- Preferences: concise / technical / Chinese
- Knowledge: Transformer ✓, RL ✗, RAG ✓
- Current project: legal-RAG-agent

When answering:
- Use Chinese
- Keep concise (50 word max)
- Skip Transformer basics
- Be careful with RL concepts (introduce)
- Relate examples to legal domain
```

## 隐私 & 控制

| 实践 | 做 |
|------|---|
| 让用户 view memory | "/memory list" |
| Delete | "/forget that" |
| Pause memorize | "/no_memory_for_now" |
| Per-context boundary | work vs personal |
| Encrypt sensitive | PII 字段加密 |

## 数字（Mem0 2024）

| 设置 | user satisfaction |
|------|------------------:|
| No memory | 60% |
| Implicit memory | 78% |
| + explicit preferences | 85% |

## 实现 (`personalization.py` 预告)

```python
@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    preferences: dict = field(default_factory=dict)
    knowledge_level: dict[str, str] = field(default_factory=dict)
    style: dict = field(default_factory=dict)

    def update_from_feedback(self, signal: dict): ...
    def to_system_prompt(self) -> str: ...
```

## 失败 case

| 反 | 解 |
|----|----|
| 推断错 (假定 senior 但是 junior) | 显式 confirm |
| 过度个性化 (信息茧房) | 多样性 prompt |
| Stale (3 年前偏好) | TTL on preferences |
| Cross-context 串 (work vs personal) | namespace |

## 退出条件

- 能列 4 类信号
- 能讲 ChatGPT/Claude 个性化
- 能写 UserProfile 模板

## 一句话

> 个性化 = 4 类信号 (pref / level / style / domain) 进 system prompt — user satisfaction +25pp。
