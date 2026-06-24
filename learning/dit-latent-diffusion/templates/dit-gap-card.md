<!-- DiT/条件扩散研究 gap 卡 (M13.3) · 接 9.3/9.4 -->
# DiT/条件扩散 gap: ____________

## gap 类型 (6 类, 9.3)
☐ 方法 ☐ 评测 ☐ 假设 ☐ 泛化 ☐ 复现 ☐ 理论

## 候选方向
- [ ] CFG guidance scale 的最优区间 (保真 vs 多样) 随数据怎么变
- [ ] DiT scaling law (扩散也遵循 transformer scaling?)
- [ ] latent 空间设计 (VAE vs VQ) 对扩散质量的影响
- [ ] 条件注入方式 (cross-attn vs adaLN vs 条件token) 对比

## 可证伪假设 (9.4-L1)
> H: ____ 在 ____ 下, 比 ____ ____

## 最小验证 (复用 13.3 dit.py + 9.4)
