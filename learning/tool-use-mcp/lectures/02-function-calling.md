# L02 · OpenAI Function Calling

## 30 秒核心

> 给 LLM 一组工具 schema (JSON Schema)，LLM 输出 `tool_calls` 数组 → 程序执行 → 回写 result → LLM 继续。

## API 完整 round-trip

```python
# Round 1: LLM decides to call tool
response = openai.chat.completions.create(
  model="gpt-4o",
  messages=[{"role":"user","content":"Weather in Tokyo?"}],
  tools=[{
    "type":"function",
    "function": {
      "name":"get_weather",
      "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
    }
  }]
)

# Response with tool_calls
tool_calls = response.choices[0].message.tool_calls
# [{"id":"call_abc","function":{"name":"get_weather","arguments":'{"city":"Tokyo"}'}}]

# Round 2: execute and send back
result = get_weather(city="Tokyo")
response = openai.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role":"user","content":"Weather in Tokyo?"},
    {"role":"assistant","tool_calls":tool_calls},
    {"role":"tool","tool_call_id":"call_abc","content":json.dumps(result)},
  ]
)
# Final answer
```

## Parallel tool calls (2024.05)

```python
# 一次返回多个 tool_call
[{"name":"get_weather","args":{"city":"Tokyo"}},
 {"name":"get_weather","args":{"city":"Paris"}},
 {"name":"get_time","args":{"city":"NY"}}]

# 并行执行 → 一并 message 回写
```

## tool_choice 控制

| 值 | 行为 |
|----|------|
| `"auto"` | LLM 决 |
| `"none"` | 强制不调 |
| `"required"` | 必调 |
| `{"type":"function","function":{"name":"X"}}` | 必调 X |

## Anthropic tool use API

```python
import anthropic
response = client.messages.create(
  model="claude-sonnet-4",
  tools=[{"name":"get_weather","input_schema":{...}}],
  messages=[{"role":"user","content":"Weather?"}]
)
# response.content = [ToolUseBlock(name="get_weather", input={...})]
```

类似 OpenAI 但 envelope 略不同。MCP 正是解决这种"每家略不同"。

## Schema 通用结构（JSON Schema 子集）

```json
{
  "type":"object",
  "properties":{
    "city":{"type":"string","description":"City name"},
    "unit":{"type":"string","enum":["C","F"]}
  },
  "required":["city"]
}
```

## 实践注意

| 坑 | 解 |
|---|----|
| 嵌套 schema 太深 | 扁平 |
| Enum 过多 | 按使用频次截断 |
| Required 全开 | 让 LLM 推断 default |
| Description 太短 | 一句话清楚意图 |
| Args parser fail | strict JSON mode |

## 实现 (`openai_tools.py` 预告)

```python
def parse_tool_call(json_str):
    obj = json.loads(json_str)
    return obj["name"], obj["arguments"]
```

## 退出条件

- 能默写 OpenAI tools 完整 round-trip
- 能写 JSON Schema for tool
- 知道 4 种 tool_choice

## 一句话

> Function calling = LLM 输出 JSON tool_call → 程序执行 → 回写 → LLM 继续 — 但每家协议略不同。
