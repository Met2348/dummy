"""01-debug-demo.py

配合 01-vscode-editor-workflow.md 第 4 节(调试器)使用的练习脚本。

这个脚本的特点:能跑、不报错、不崩溃,但算出来的结果是错的。这是真实工作中最常见、
也最难排查的一类 bug——如果代码直接崩溃,报错信息和堆栈(traceback)会直接指给你看
出错的位置;但"安静地算错"不会主动告诉你哪里有问题,你必须自己一步步验证。

建议第一次接触这个文件时,不要直接读代码"看出"bug 在哪(这个例子确实很短、很容易
一眼看穿,但请把它当成一次调试器操作练习,而不是找茬游戏)。按 01-vscode-editor-workflow.md
第 4 节的步骤,先运行一遍看到错误结果,再用断点+单步执行真正定位问题。
"""
from __future__ import annotations


def average(nums: list[float]) -> float:
    """计算 nums 的算术平均值。"""
    total = 0
    count = len(nums)
    for i in range(count - 1):
        total += nums[i]
    return total / count


def summarize_scores(scores: list[float]) -> str:
    """对外的入口函数:算出平均分,再包一层格式化文本。"""
    avg = average(scores)
    return f"average score = {avg}"


def main() -> None:
    scores = [10, 20, 30, 40]  # 手算平均值 = (10+20+30+40) / 4 = 25.0
    result = summarize_scores(scores)
    print(result)


if __name__ == "__main__":
    main()
