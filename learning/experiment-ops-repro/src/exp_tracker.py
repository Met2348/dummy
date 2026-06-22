"""
exp_tracker.py — 一个轻量、本地、零依赖的实验追踪器 (mimics wandb 的核心接口).

为什么自己写一个而不直接教 wandb: wandb/mlflow 是优秀工具, 但它们把「追踪到底记了什么」
藏在了 SaaS 后面。这个 80 行的本地版本让你**亲眼看见**一次可复现的实验记录必须包含什么:
  config (跑了什么) + metrics (结果) + 环境指纹 (git sha / python / 包版本 / 时间戳)。
理解了这三件套, 你换用 wandb 只是换个 API, 心智模型完全一样。

它把每个 run 存成一行 JSON (jsonl), 可被 pandas 直接读、可 grep、可进 git diff ——
对个人研究, 这往往比一个云端 dashboard 更耐用 (十年后还打得开)。

用法 (和 wandb 几乎一样):
    import exp_tracker as et
    run = et.init(project="dpo-noise", config={"method":"DPO","noise":0.4,"seed":3})
    run.log({"win_rate": 0.40})
    run.finish()
    runs = et.load_runs("dpo-noise")     # -> list[dict], 喂给 pandas
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_DIR = Path("_runs")  # 默认落地目录 (notebook 里会指定)


def _git_sha() -> str:
    """当前代码的 git commit sha —— 复现的命脉: 没有它, 半年后你不知道这个结果是哪版代码跑的."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        tag = sha.stdout.strip() or "no-git"
        if dirty.stdout.strip():
            tag += "-dirty"   # 有未提交改动 → 结果可能不可复现, 必须标出来
        return tag
    except Exception:
        return "no-git"


def _env_fingerprint(packages=("numpy", "scipy", "torch", "transformers")) -> dict:
    """环境指纹: python 版本 + 关键包版本 + 平台. 复现 checklist 的核心要素之一."""
    import importlib.metadata as md
    pkgs = {}
    for p in packages:
        try:
            pkgs[p] = md.version(p)
        except Exception:
            pkgs[p] = "absent"
    return {"python": platform.python_version(), "platform": platform.system(), "packages": pkgs}


@dataclass
class Run:
    project: str
    config: dict
    out_dir: Path
    metrics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    _seq: int = 0

    def log(self, metrics: dict, step: int | None = None):
        """记录一组指标. 简单起见只保留最后一次 (够本专题教学; 真追踪会存时间序列)."""
        self.metrics.update(metrics)
        self._seq = step if step is not None else self._seq + 1

    def finish(self) -> Path:
        """把这个 run 追加写进 project 的 jsonl. 返回文件路径."""
        record = {
            "project": self.project,
            "config": self.config,
            "metrics": self.metrics,
            **self.meta,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        f = self.out_dir / f"{self.project}.jsonl"
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return f


def init(project: str, config: dict, out_dir: str | Path = DEFAULT_DIR,
         capture_env: bool = True) -> Run:
    """开一个 run. 自动盖上环境指纹 (git sha / python / 包版本 / 时间戳由调用方可选注入)."""
    meta = {}
    if capture_env:
        meta["git_sha"] = _git_sha()
        meta["env"] = _env_fingerprint()
    return Run(project=project, config=dict(config), out_dir=Path(out_dir), meta=meta)


def load_runs(project: str, out_dir: str | Path = DEFAULT_DIR) -> list[dict]:
    """读回一个 project 的所有 run 记录 (list[dict]), 可直接 pd.json_normalize."""
    f = Path(out_dir) / f"{project}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for seed in range(3):
            r = init("smoke", {"method": "DPO", "noise": 0.4, "seed": seed}, out_dir=d)
            r.log({"win_rate": 0.40 + seed * 0.001})
            r.finish()
        runs = load_runs("smoke", out_dir=d)
        print(f"记录了 {len(runs)} 个 run; 第一个的环境指纹:")
        print(json.dumps(runs[0]["env"], ensure_ascii=False, indent=2))
        print("git_sha:", runs[0]["git_sha"])
