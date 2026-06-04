# L08 · OSWorld — OS-level agent

**Xie et al. 2024, THU** · arXiv 2404.07972

## 数据

- **369 任务**横跨：
  - File operations
  - Email (Thunderbird)
  - Browser (Chrome)
  - Office (LibreOffice Writer/Calc/Impress)
  - VS Code
  - Multimedia (VLC, GIMP)
  - System (file manager, terminal)
- 真实 Ubuntu VM 环境
- 评判：**最终系统状态**（文件、注册表、open windows）

## 例题

```
Goal: "Open my-doc.docx in LibreOffice Writer, change all 'red' to
'blue', save as my-doc-v2.docx."

Initial state: ~/Documents/my-doc.docx exists
Expected end-state:
  - ~/Documents/my-doc-v2.docx exists
  - content has no 'red', has 'blue' in same positions
```

## Agent interface

```
observation: screenshot (1920x1080) + a11y tree
action_space:
  pyautogui_click(x, y)
  pyautogui_typewrite(text)
  pyautogui_hotkey(*keys)
  screenshot()
  ...
```

## 分数

| Model | OSWorld success |
|-------|-----------------|
| GPT-4o (a11y tree) | 12.2% |
| GPT-4o (screenshot) | 5.0% |
| Claude 3.5 Sonnet | 22% |
| **AutoGLM-OS (智谱 2025)** | **48.9%** |
| **Claude 3.7 + extended** | **53%+** |
| 人类 | 72% |

## 难点 5 倍 of WebArena

1. **完整 OS state**：browser + file system + processes
2. **GUI + 多模态**：必须看屏幕
3. **像素级点击**：要精确坐标
4. **耗时**：每 task 5-15 min

## 评测工程

- 完整 Docker + Ubuntu VM
- screenshot 流→ model → action → vm
- VM snapshot 重置防污染

## 实操

OSWorld 代码量大、依赖重。
推荐：去官方 repo 跑 1 个 demo task。

## 一句话

> OSWorld = "让 LLM 替我操作电脑"，是 OS-level agent 的金标准。
