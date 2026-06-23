"""
connectors.py — VLM 的三种「视觉→LLM」连接器, 并排实现 + 对比.

为什么这是 VLM 最关键的代码: 视觉塔 (M10.1) 给了视觉 token, LLM 吃文本 token, 但两者
embedding 空间不同、序列怎么拼也不同。**连接器决定「视觉如何接进 LLM」**, 是造 VLM 第一个、
也是最重要的架构决定。本文件把三条主流路线砍到最小可对比:

  1. ProjectionConnector  (LLaVA 路线): 一个 MLP 把视觉 token 投影到 LLM embedding 空间, 当
                           普通 token 拼进输入序列。最简单, 效果出奇好。
  2. CrossAttnConnector   (Flamingo 路线): 在 LLM 层间插 cross-attention, 让文本 token「看」
                           视觉 token; 视觉不占输入序列长度。
  3. EarlyFusionConnector (Chameleon/Fuyu 路线): 视觉和文本从第一层就当同一种 token 混在一起,
                           没有独立连接器, 统一 transformer 处理。

纯 torch (tiny, CPU 秒跑). 无 torch 时退化为参数量对比 (numpy)。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_connectors(vis_dim: int = 32, llm_dim: int = 48, n_vis: int = 16,
                     n_query: int = 4, seed: int = 0):
    """构建三种连接器. 返回 dict {name: module}. 无 torch 返回 None."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"[connectors] 无 torch ({exc!r}); 见 param_comparison 的理论对比。")
        return None

    torch.manual_seed(seed)

    class ProjectionConnector(nn.Module):
        """LLaVA: MLP 投影, 视觉 token 当普通 token 拼进序列 (占 n_vis 个位置)."""
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(vis_dim, llm_dim), nn.GELU(),
                                     nn.Linear(llm_dim, llm_dim))

        def forward(self, vis_tokens, txt_tokens):  # (B,n_vis,vis_dim),(B,n_txt,llm_dim)
            import torch
            v = self.mlp(vis_tokens)                # 投影到 LLM 空间
            return torch.cat([v, txt_tokens], dim=1)  # 拼成 [视觉..., 文本...]; 序列长 +n_vis

    class CrossAttnConnector(nn.Module):
        """Flamingo: 先用 query 把 n_vis 压成 n_query (resampler), 文本 cross-attend 视觉.
        视觉不占输入序列 (通过 cross-attn 注入)。返回融合后的文本表示。"""
        def __init__(self):
            super().__init__()
            self.query = nn.Parameter(__import__("torch").randn(1, n_query, llm_dim) * 0.02)
            self.vis_proj = nn.Linear(vis_dim, llm_dim)
            self.resampler = nn.MultiheadAttention(llm_dim, 4, batch_first=True)
            self.cross = nn.MultiheadAttention(llm_dim, 4, batch_first=True)

        def forward(self, vis_tokens, txt_tokens):
            B = vis_tokens.shape[0]
            v = self.vis_proj(vis_tokens)
            q = self.query.expand(B, -1, -1)
            compressed, _ = self.resampler(q, v, v)     # n_vis → n_query (压缩)
            fused, _ = self.cross(txt_tokens, compressed, compressed)  # 文本看压缩视觉
            return txt_tokens + fused                    # 残差注入; 序列长不变

    class EarlyFusionConnector(nn.Module):
        """Chameleon/Fuyu: 视觉和文本投到同一空间, 从第一层就混作一种 token."""
        def __init__(self):
            super().__init__()
            self.vis_embed = nn.Linear(vis_dim, llm_dim)  # 视觉 token 也进 llm 空间, 无特殊处理

        def forward(self, vis_tokens, txt_tokens):
            import torch
            v = self.vis_embed(vis_tokens)
            # 早融合: 直接当同质 token 拼接, 后续由统一 transformer 处理 (这里只到拼接)
            return torch.cat([v, txt_tokens], dim=1)

    return {
        "projection": ProjectionConnector(),
        "cross_attn": CrossAttnConnector(),
        "early_fusion": EarlyFusionConnector(),
    }


def param_comparison(vis_dim=32, llm_dim=48, n_vis=16, n_query=4) -> list[dict]:
    """三种连接器的定性对比 (不依赖 torch). 用于课件/notebook 表格."""
    return [
        {"connector": "projection (LLaVA)",
         "视觉占输入序列": f"是 (+{n_vis} token)", "参数量": "小 (一个 MLP)",
         "改 LLM 结构": "否", "优点": "最简单, 效果好, 训练快",
         "缺点": "视觉吃上下文; 高分辨率 token 爆炸"},
        {"connector": "cross_attn (Flamingo)",
         "视觉占输入序列": f"否 (压成 {n_query}, cross-attn 注入)", "参数量": "中 (resampler+cross 层)",
         "改 LLM 结构": "是 (插 cross-attn 层)", "优点": "视觉不占上下文; 可处理多图/视频",
         "缺点": "改 LLM 结构, 实现复杂"},
        {"connector": "early_fusion (Chameleon)",
         "视觉占输入序列": "是 (同质 token)", "参数量": "极小 (一个 embed)",
         "改 LLM 结构": "否 (但需从头训)", "优点": "架构统一, 利于理解+生成一体",
         "缺点": "通常需从头训, 不能复用纯文本 LLM"},
    ]


if __name__ == "__main__":
    print("=== 三种 VL 连接器对比 ===")
    for row in param_comparison():
        print(f"\n[{row['connector']}]")
        for k, v in row.items():
            if k != "connector":
                print(f"  {k}: {v}")

    conns = build_connectors()
    if conns is not None:
        import torch
        vis = torch.randn(2, 16, 32)   # 2 张图各 16 视觉 token (来自 M10.1 ViT)
        txt = torch.randn(2, 10, 48)   # 2 段文各 10 token
        print("\n=== 前向输出序列长度 (输入: 16 视觉 + 10 文本) ===")
        for name, m in conns.items():
            out = m(vis, txt)
            print(f"  {name:14}: 输出 {tuple(out.shape)}  "
                  f"(序列长 {out.shape[1]})")
