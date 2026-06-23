# 10.5 video-audio-omni — 把 token 化扩到视频/音频, 通往 omni

> **Module 10「多模态/VLM 基础」· 第 5 专题 (时序模态)**
> 文本/图能 token 化了, 视频 (图+时间) 和音频 (波形→谱) 呢? 本专题补齐时序模态, 通往 omni 模型 (一个模型通吃所有模态)。

---

## 这个专题要解决的真问题

- **视频** = 图 + 时间, 逐帧 token 化会爆炸 + 丢运动 → **时空 patch** (压缩 + 编码运动)。
- **音频** = 极长 1D 波形, 不能直接 token 化 → **波形→mel 谱→token** (你 EE 本行)。
- **omni** = 所有模态进同一条 token 流, 统一 transformer 通吃。
- **实时** = 流式/低延迟/同步 (接 harness + 推理优化)。

> 一条主线延续 M10: **token 是通用货币**。这个专题把它扩到时序模态, 让「给 LLM 装所有感官」成为可能。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-video-tokenization.md` | 视频时空 patch: 压缩 token + 编码运动 |
| L2 | `lectures/L2-audio-speech.md` | 波形→STFT→mel 谱→token (EE 本行) |
| L3 | `lectures/L3-omni-models.md` | omni: 所有模态一条流 + 三大挑战 |
| L4 | `lectures/L4-realtime-constraints.md` | 实时: 流式/低延迟/同步 (接 harness/M5) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-video-tokens.ipynb` | 用 `src/temporal_tokens.py` 对合成「移动色块」视频做逐帧 vs 时空 patch, 对比 token 数 + 验证运动保留 |
| `notebooks/N2-audio-tokens.ipynb` | 用 `src/audio_features.py` 把合成音频走完「波形→mel 谱→token」, 可视化 mel 谱 |

## 工具 (`src/`)
- `temporal_tokens.py` — 视频时空 patch + token 数对比 + 运动信号 (numpy)
- `audio_features.py` — STFT + mel 谱 + 音频 token 化 (numpy FFT, 无 librosa)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / numpy (核心) / matplotlib。合成视频/音频离线确定性, 无需真实媒体文件。

## 完成本专题后你应该能
- [ ] 解释视频时空 patch 怎么压缩 token 并编码运动
- [ ] 走通「波形→STFT→mel→token」音频管线
- [ ] 说清 omni 模型架构 + 三大挑战 (token 不平衡/实时/数据)
- [ ] 列出实时多模态的工程约束并连到 M5/harness 专题

---
## 在 Module 10 中的位置
```
  10.1-10.4 (图) → 10.5 视频/音频/omni ◄你在这 → 10.6 评测 → 10.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
