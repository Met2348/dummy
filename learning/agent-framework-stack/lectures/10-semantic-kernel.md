# L10 · Semantic Kernel（Microsoft）

## 30 秒核心

> Semantic Kernel = Microsoft **企业 .NET 友好**的 agent framework，C# / Python / Java 三语言。

2023.03 公布，Azure 集成深。

## 核心抽象

| 抽象 | 用 |
|------|---|
| `Kernel` | 主对象 (类 LangChain 的 Runnable) |
| `Plugin` | 一组 function (tools) |
| `Plan` | sequential planner output |
| `Memory` | semantic memory store |
| `Connector` | LLM vendor (Anthropic/OpenAI/Azure OpenAI/Onnx) |

## C# 例

```csharp
using Microsoft.SemanticKernel;

var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(...)
    .Build();

var plugin = kernel.ImportPluginFromType<WeatherPlugin>();
var result = await kernel.InvokeAsync(
    plugin["GetWeather"],
    new() {["city"] = "Tokyo"}
);
```

## Python 例

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(...))

@kernel_function(name="get_weather")
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

kernel.add_plugin([get_weather], plugin_name="weather")
```

## Process Framework (2024)

类 LangGraph state machine：
```csharp
var process = new ProcessBuilder()
    .AddStep<PlannerStep>()
    .AddStep<ExecutorStep>()
    .AddStep<ReviewerStep>()
    .Build();
```

## 优势

| 优 | 解释 |
|----|----|
| C# / .NET 一等 | 企业 Microsoft 栈 |
| Azure 集成 | Azure OpenAI / AI Search 直连 |
| Multi-language | Python / Java / C# |
| OpenAPI 友好 | Auto plugin from OpenAPI |

## 弱势

| 弱 | 解释 |
|----|----|
| Python 社区小于 LangChain | |
| API 频改 | |
| 偏企业 | 个人项目过重 |

## 适合

| 适合 | 不适合 |
|------|------|
| Microsoft / Azure 栈 | Pure OSS Python |
| C# 团队 | Anthropic 全栈 |
| 企业合规 | Quick PoC |

## 退出条件

- 能列 5 抽象
- 知道 3 语言
- 知道 OpenAPI auto-import

## 一句话

> Semantic Kernel = Microsoft 企业 .NET 友好 agent framework — Azure 集成深，C#/Python/Java 三语。
