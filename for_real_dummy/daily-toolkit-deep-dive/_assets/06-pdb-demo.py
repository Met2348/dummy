"""06-pdb-demo.py

配合 06-monitoring-and-debugging.md 第 4 节(pdb 命令行调试)使用的练习脚本。

跟 01-vscode-editor-workflow.md 里的 01-debug-demo.py 是"同一类问题、两种界面":
那边用 VSCode 的图形断点,这里用标准库自带的 pdb——因为不是每次都有图形界面可用
(比如 SSH 到一台只开了终端的远程 GPU 服务器)。bug 的风格也延续同一路数:代码能跑、
不崩溃、但算出来的结果是错的,必须真的停下来看变量才能发现。

场景:running_average() 模拟"记录训练 loss 的滑动平均"——这是日常盯训练日志时
很常见的操作。它有一个索引 bug:for 循环从 1 开始而不是 0,导致窗口内第一个数字
永远被漏掉,但除数还是按完整窗口大小除的,所以算出来的平均值会比真实值偏低,
而且不会报错、不会崩溃,非常容易被当成"训练效果比实际更好"而忽略过去。

用 breakpoint() 在第 4 步训练时停下,配合 s(步入)/n(单步)/p(打印变量)/l(看源码)
四个核心命令,能亲眼看着这个 bug 发生的全过程。
"""
from __future__ import annotations


def running_average(history: list[float], window: int) -> float:
    """计算 history 最近 window 个数字的平均值(模拟 loss 的滑动平均)。"""
    recent = history[-window:]
    total = 0.0
    for i in range(1, len(recent)):  # bug: should start at 0, this skips recent[0]
        total += recent[i]
    return total / window


def log_step(step: int, loss_history: list[float]) -> None:
    """每一步训练后调用一次,打印当前 loss 和最近 3 步的滑动平均。"""
    avg = running_average(loss_history, window=3)
    print(f"step={step} loss={loss_history[-1]:.4f} avg3={avg:.4f}")


def main() -> None:
    losses = [0.90, 0.70, 0.50, 0.40, 0.35, 0.30]
    for step, loss in enumerate(losses, start=1):
        history = losses[:step]
        if step == 4:
            breakpoint()  # pdb starts here (see doc section 4)
        log_step(step, history)


if __name__ == "__main__":
    main()
