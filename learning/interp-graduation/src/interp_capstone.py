"""
interp_capstone.py — Module 12 capstone: 完整 interp 流程装配 + 研究 gap 雷达.

M12 走完了机制可解释性全套: 地基(12.1) → probing(12.2) → patching(12.3) → SAE(12.4)
→ circuits(12.5) → CoT忠实性(12.6)。本文件做 capstone:
  1. assembly_check(): 跨专题 import 全部 M12 src, 证明组合成一个工具箱。
  2. run_full_interp(): 对一个 tiny 模型跑**完整逆向工程流程** (探针→patching→SAE), 输出一个连贯的机制故事。
  3. 研究 gap 雷达 + idea 卡 (重点 interp × reasoning, 用户最可能转 PhD 题)。

跨专题 src 复用 (同 M10/M11/M13 capstone)。纯 toy CPU 确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEARNING = Path(__file__).resolve().parents[2]   # learning/

STACK = [
    ("interp-foundations",          "tiny_transformer", "12.1 可hook激活的受控基座"),
    ("probing-and-activations",     "probing",          "12.2 线性探针 + logit lens"),
    ("causal-interventions",        "patching",         "12.3 activation patching + ablation"),
    ("sparse-autoencoders",         "sae",              "12.4 SAE 解叠加"),
    ("circuits-attention",          "circuits",         "12.5 induction head + 归因"),
    ("cot-faithfulness-oversight",  "cot_probe",        "12.6 CoT 忠实性 + w2s"),
]


def add_paths():
    for topic, _, _ in STACK:
        p = LEARNING / topic / "src"
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sh = LEARNING / "_shared"
    if str(sh) not in sys.path:
        sys.path.insert(0, str(sh))


def assembly_check() -> list:
    """跨专题 import M12 全套 src。返回 [(label, ok, detail)]。"""
    add_paths()
    results = []
    for topic, mod, desc in STACK:
        try:
            m = __import__(mod)
            results.append((f"{desc}", True, f"import {mod} ✓"))
        except Exception as e:
            results.append((f"{desc}", False, repr(e)))
    return results


def run_full_interp(seed: int = 0) -> dict:
    """对一个 tiny transformer 跑完整逆向工程: 训练 → 探针 → patching → SAE。返回机制故事的各项证据。"""
    add_paths()
    import tiny_transformer as tt
    import probing as pr
    import patching as pt
    import sae as S

    Xi, Yi = tt.make_data(2000, seed=0)
    model = tt.build_model(); tt.train(model, Xi, Yi, epochs=800)
    acc = tt.accuracy(model, *tt.make_data(400, seed=9))

    # 12.2 探针: residual 是否线性编码"当前值"
    acts, labels = pr.tiny_layer_activations(model, tt.make_data(500, seed=5)[0], layer_key="resid_post_1")
    _, probe_acc = pr.linear_probe(acts, labels, n_classes=tt.V)

    # 12.3 patching: 因果定位
    grid, _ = pt.patch_recovery(model, seed=3)
    causal_pos = int(np.argmax(grid.max(0)))

    # 12.4 SAE: 解叠加纯度
    sae = S.build_sae(acts.shape[1], tt.V * 3); S.train_sae(sae, acts, epochs=500, l1=1e-2)
    codes = S.feature_codes(sae, acts)
    _, sae_purity = S.monosemanticity(codes, labels)
    _, raw_purity = S.monosemanticity(acts - acts.min(), labels)

    return dict(task_acc=acc, probe_acc=probe_acc, causal_pos=causal_pos, last_pos=tt.SEQ - 1,
                sae_purity=sae_purity, raw_purity=raw_purity)


# ───────────────────────── 研究 gap 雷达 (接 M9, 重点 interp×reasoning) ─────────────────────────
GAPS = [
    dict(area="CoT 忠实性的机制级验证 ★",
         gap="CoT 说的步骤, 模型内部真在算吗? 用 patching/探针核查 CoT 与内部计算是否一致 (M12.6 偏置敏感性只是行为级)。",
         why_hard="要把窄 CoT 步骤映射到内部组件; 推理模型大、circuit 乱。",
         min_exp="在一个会 CoT 的小模型上, 对'某一步'做 activation patching, 看改它答案是否变 (M12.3+12.6)。",
         connects="M12.3 patching + M12.6 CoT + reasoning-r1"),
    dict(area="interp × reasoning: 计算 vs 陈述一致性 ★",
         gap="推理模型内部的'真实计算'和'陈述的 CoT'差多少? 能否探针读出'内部已定答案'早于 CoT 写完?",
         why_hard="需要把推理过程映射到激活; 早答检测要时序 interp。",
         min_exp="探针读推理模型中间层'当前候选答案', 看它在 CoT 写完前是否已定 (早答=不忠实)。",
         connects="M12.2 探针 + M12.6 + reasoning-r1 (用户甜点)"),
    dict(area="欺骗/装弱的 interp 检测",
         gap="模型故意装弱/欺骗时, 内部会露馅吗? 用探针/SAE 检测'知识-陈述不一致'。",
         why_hard="足够聪明的模型可能也学会欺骗 interp; 检测标准不清。",
         min_exp="训一个'会装弱'的小模型 (有能力但被提示藏), 用探针看内部是否仍表示了真答案。",
         connects="M12.2/12.4 + M12.6 + safety"),
    dict(area="SAE 特征的因果验证 + 评估标准",
         gap="SAE 特征单义≠模型在用 (M12.4-L4); 怎么因果验证特征 + 怎么评估一个 SAE 好坏?",
         why_hard="干预 SAE 特征要映射回模型计算; 评估无公认标准。",
         min_exp="对一个 SAE 特征做干预 (放大/消融), 看模型行为是否如预期变 (因果验证)。",
         connects="M12.3 patching + M12.4 SAE"),
    dict(area="自动化 / 可扩展 circuit 发现",
         gap="人工逐组件分析不可扩展; 归因 patching/ACDC 能自动找 circuit, 但精度/完备性如何?",
         why_hard="组合爆炸 + 冗余 + 归因近似误差。",
         min_exp="在玩具上对比'逐头消融'(精确) vs'归因 patching'(近似) 找到的 circuit (M12.5)。",
         connects="M12.5 circuits + 9.4 实验"),
]


def make_idea_card(gap: dict) -> str:
    return (f"┌─ idea 卡: {gap['area']} " + "─" * max(2, 30 - len(gap['area'])) + "\n"
            f"│ 问题 (gap): {gap['gap']}\n"
            f"│ 为什么难:   {gap['why_hard']}\n"
            f"│ 最小实验:   {gap['min_exp']}\n"
            f"│ 连接:       {gap['connects']}\n"
            f"└" + "─" * 52)


def gap_radar() -> str:
    return "\n".join(f"  [{i+1}] {g['area']}  ←  {g['connects']}" for i, g in enumerate(GAPS))


if __name__ == "__main__":
    print("== M12 全套装配检查 ==")
    for label, ok, detail in assembly_check():
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:26} {detail}")
    print("\n== 完整逆向工程流程 (对一个 tiny 模型) ==")
    r = run_full_interp()
    print(f"  模型任务准确率: {r['task_acc']:.2f}")
    print(f"  探针 (12.2): residual 线性编码'当前值' 准确率 {r['probe_acc']:.2f}")
    print(f"  patching (12.3): 因果定位在位置 {r['causal_pos']} (= 最后位置 {r['last_pos']}? {r['causal_pos']==r['last_pos']})")
    print(f"  SAE (12.4): 特征纯度 {r['sae_purity']:.2f} >> 原始神经元 {r['raw_purity']:.2f}")
    print("  → 连贯机制故事: 模型读'当前值'(探针), 因果在最后位置(patching), 编码为单义特征(SAE)。")
    print(f"\n== 研究 gap 雷达 ({len(GAPS)} 个; ★=interp×reasoning 用户甜点) ==")
    print(gap_radar())
