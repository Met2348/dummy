# L07 · LM Studio（GUI 端侧）

## 1 · 定位
- 桌面 GUI（Mac/Windows）
- 内部用 llama.cpp
- 一键下载 GGUF + 聊天 + API server

## 2 · 适合谁
- 非工程师
- 隐私 / 离线场景
- 模型选择 / 对比

## 3 · 功能
- HuggingFace 模型库浏览
- 量化版本下载
- 本地 chat UI
- OpenAI 兼容 API server
- 多模型同时加载

## 4 · 部署
GUI 一键，无需命令。

## 5 · OpenAI API
启动 server 后：
```bash
curl http://localhost:1234/v1/models
curl http://localhost:1234/v1/chat/completions -d '{...}'
```

## 6 · 限制
- 不支持高吞吐
- 不支持 cluster
- 主要面向个人

## 7 · 对比 Ollama
| | Ollama | LM Studio |
|---|-------|-----------|
| 安装 | brew / package | GUI installer |
| 配置 | Modelfile | GUI |
| 高级 | 强 | 弱 |
| 入门 | 中 | **极易** |

## 8 · 一句话
> Ollama 给 power user，LM Studio 给非技术用户。
