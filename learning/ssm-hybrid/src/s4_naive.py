"""S4 naive (LTI SSM) - teaching recurrent/convolution contrast."""
from __future__ import annotations

import torch
import torch.nn as nn

from common import discretize_zoh, naive_scan


def hippo_init(d_state: int) -> torch.Tensor:
    """HiPPO-LegS init (simplified diagonal)."""
    return -torch.arange(1, d_state + 1).float()


class S4Naive(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # A: diagonal (negative)
        self.A_log = nn.Parameter(torch.log(-hippo_init(d_state) + 1e-3))
        self.B = nn.Parameter(torch.randn(d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (b, t, d). 每 channel 独立 SSM."""
        b, t, d = u.shape
        A = -torch.exp(self.A_log)                # (d_state,)  negative
        delta = torch.nn.functional.softplus(self.delta)  # (d_model,)
        out = torch.zeros_like(u)
        for ch in range(d):
            A_t = A.unsqueeze(0).expand(t, -1)
            B_t = self.B.unsqueeze(0).expand(t, -1)
            C_t = self.C.unsqueeze(0).expand(t, -1)
            delta_t = delta[ch].expand(t)
            A_bar, B_bar = discretize_zoh(A_t, B_t, delta_t)
            y = naive_scan(A_bar, B_bar, u[0, :, ch], C_t)
            out[0, :, ch] = y
        out = out + self.D * u
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    m = S4Naive(d_model=4, d_state=8)
    u = torch.randn(1, 16, 4)
    y = m(u)
    print(f"S4 out {y.shape}, params {sum(p.numel() for p in m.parameters())}")
