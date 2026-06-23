"""
audio_features.py — 把音频 (波形) 变成 mel 谱、再变成 token, 把 token 化扩到音频/语音.

为什么需要它 (M10.5): 文本、图、视频都能 token 化, 音频也一样。但音频是 1D 波形, 直接当 token
太长 (16kHz = 每秒 16000 个采样)。标准做法: 先变成 **mel 频谱图** (时间×频率的 2D 图), 再像图
一样 patch/量化成 token。这就是 Whisper (ASR) 和音频生成模型的输入处理方式。

本文件给:
  - make_tone        : 合成音频信号 (正弦混合, 可变频率), 确定性
  - stft_magnitude   : 短时傅里叶变换幅度谱 (numpy FFT, 无需 librosa)
  - mel_spectrogram  : mel 滤波后的谱 (近似人耳感知), 音频版的「图」
  - frames_to_tokens : 把 mel 谱按时间帧切成 token 序列

纯 numpy (FFT), 离线确定性, 无需真实音频文件或 librosa。
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_tone(freqs=(220.0, 440.0), sr: int = 4000, dur: float = 0.5,
              seed: int = 0) -> np.ndarray:
    """合成音频: 几个正弦波叠加 + 轻噪声. 返回 1D 波形 (sr*dur 个采样)。确定性。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return (wave + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def stft_magnitude(wave: np.ndarray, n_fft: int = 256, hop: int = 128) -> np.ndarray:
    """短时傅里叶变换幅度谱. 返回 (n_frames, n_fft//2+1). 把 1D 波形变成时间×频率的 2D。"""
    win = np.hanning(n_fft)
    frames = []
    for start in range(0, len(wave) - n_fft + 1, hop):
        seg = wave[start:start + n_fft] * win
        spec = np.abs(np.fft.rfft(seg))
        frames.append(spec)
    return np.stack(frames) if frames else np.zeros((0, n_fft // 2 + 1))


def _mel_filterbank(n_freq: int, n_mel: int, sr: int) -> np.ndarray:
    """简化 mel 滤波器组 (n_mel, n_freq). mel 刻度近似人耳 (低频分辨高、高频分辨低)。"""
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    f_max = sr / 2
    mel_pts = np.linspace(hz2mel(0), hz2mel(f_max), n_mel + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_freq - 1) * hz_pts / f_max).astype(int)
    fb = np.zeros((n_mel, n_freq))
    for m in range(1, n_mel + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l: fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fb[m - 1, k] = (r - k) / (r - c)
    return fb


def mel_spectrogram(wave: np.ndarray, sr: int = 4000, n_fft: int = 256,
                    hop: int = 128, n_mel: int = 16) -> np.ndarray:
    """mel 频谱图 (n_frames, n_mel). 这是音频版的「图」, 可以像图一样 patch/量化。"""
    mag = stft_magnitude(wave, n_fft, hop)
    fb = _mel_filterbank(mag.shape[1], n_mel, sr)
    mel = mag @ fb.T
    return np.log1p(mel)   # log 压缩动态范围 (像人耳)


def frames_to_tokens(mel: np.ndarray, codebook_size: int = 16, seed: int = 0):
    """把 mel 谱的每个时间帧量化成一个离散 token (k-means 码本). 返回 (tokens, codebook)。
    这就是「音频 token 化」—— 和图像 VQ (M10.4) 同思路。"""
    rng = np.random.default_rng(seed)
    if len(mel) == 0:
        return np.array([]), np.zeros((codebook_size, 0))
    K = min(codebook_size, len(mel))
    idx = rng.choice(len(mel), size=K, replace=False)
    cb = mel[idx].copy()
    for _ in range(20):
        d = ((mel[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for c in range(K):
            if (a == c).any():
                cb[c] = mel[a == c].mean(0)
    tokens = ((mel[:, None, :] - cb[None, :, :]) ** 2).sum(-1).argmin(1)
    return tokens, cb


if __name__ == "__main__":
    wave = make_tone(freqs=(220.0, 440.0), sr=4000, dur=0.5, seed=1)
    print(f"合成音频: {len(wave)} 采样 (4kHz, 0.5s)")
    mel = mel_spectrogram(wave)
    print(f"mel 谱: {mel.shape} (时间帧 × mel 频带) —— 音频版的'图'")
    tokens, cb = frames_to_tokens(mel, codebook_size=8, seed=1)
    print(f"音频 token 序列 ({len(tokens)} 个): {tokens.tolist()}")
    print("→ 波形 → mel 谱 → 离散 token, 和图像 VQ 同思路 (M10.4)。音频也能进统一 token 流。")
