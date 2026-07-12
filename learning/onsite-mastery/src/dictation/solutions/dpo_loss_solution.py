"""dpo_loss_spec 的参考实现。用 numpy，log-sigmoid 走数值稳定的 logaddexp 恒等式。"""
from __future__ import annotations

import numpy as np


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    policy_chosen_logps = np.asarray(policy_chosen_logps, dtype=float)
    policy_rejected_logps = np.asarray(policy_rejected_logps, dtype=float)
    ref_chosen_logps = np.asarray(ref_chosen_logps, dtype=float)
    ref_rejected_logps = np.asarray(ref_rejected_logps, dtype=float)

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)

    # -log(sigmoid(x)) = log(1+exp(-x)) = logaddexp(0, -x)，比直接算 -log(1/(1+exp(-x))) 数值稳定
    loss = np.logaddexp(0.0, -logits)
    return loss
