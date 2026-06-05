# L12 · Capstone-1 — mini-HELM 全跑

## 跑

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from mini_helm import run_mini_helm, to_md, ascii_radar
scores = run_mini_helm()
print(to_md(scores))
print()
print('### r1_tiny radar:')
print(ascii_radar(scores['r1_tiny']))
print()
print('### phi_tiny radar:')
print(ascii_radar(scores['phi_tiny']))
"
```

## 预期输出

```
# mini-HELM 5-ckpt × 4-dim

| ckpt | knowledge | reasoning | safety | efficiency | avg |
|---|---:|---:|---:|---:|---:|
| vanilla   | 1.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| lora      | 1.00 | 1.00 | 1.00 | 0.95 | 0.99 |
| dpo       | 1.00 | 1.00 | 1.00 | 0.90 | 0.98 |
| r1_tiny   | 1.00 | 1.00 | 1.00 | 0.50 | 0.88 |
| phi_tiny  | 1.00 | 1.00 | 1.00 | 0.70 | 0.93 |

### r1_tiny radar:
knowledge    [####################] 1.00
reasoning    [####################] 1.00
safety       [####################] 1.00
efficiency   [##########          ] 0.50

### phi_tiny radar:
knowledge    [####################] 1.00
reasoning    [####################] 1.00
safety       [####################] 1.00
efficiency   [##############      ] 0.70
```

## 分析

| ckpt | 关键观察 |
|------|---------|
| vanilla | reasoning + safety 全 0，仅效率高 |
| lora | 全维 1.0，最高分（但 mock 数字） |
| dpo | 全维 1.0，效率略输 lora |
| r1_tiny | latency 80ms 拖累效率 |
| phi_tiny | 270M 较大，但 60ms 平衡 |

## 退出条件

- 5 行 × 4 列输出
- vanilla avg ≤ 0.5
- 其他 ≥ 0.85
- ASCII radar 显示 r1_tiny / phi_tiny 的 trade-off

## 一句话

> Capstone-1 = 1 张 mini-HELM 表 + 雷达，看 5 ckpt "X 光照"。
