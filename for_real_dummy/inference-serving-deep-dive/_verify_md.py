import os
import re
import subprocess
import sys
from pathlib import Path

# Only extract the actual runnable "可运行例子" blocks (not the illustrative
# excerpt snippets under "是什么", which are not claimed to be standalone-runnable).
PATTERN = re.compile(r"\*\*可运行例子[^\n]*\*\*\s*\n```python\n(.*?)```", re.DOTALL)

def extract_blocks(md_path):
    text = Path(md_path).read_text(encoding="utf-8")
    return PATTERN.findall(text)

def main():
    md_path = sys.argv[1]
    blocks = extract_blocks(md_path)
    print(f"Found {len(blocks)} runnable example blocks in {md_path}")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    fails = []
    for i, block in enumerate(blocks, 1):
        result = subprocess.run(
            [sys.executable, "-c", block],
            capture_output=True, text=True, timeout=600, env=env,
            cwd=str(Path(md_path).resolve().parents[2]),  # repo root, so sys.path.insert("learning/...") works
        )
        if result.returncode != 0:
            fails.append((i, block, result.stderr))
            print(f"  [FAIL] block {i}")
        else:
            print(f"  [ok]   block {i}")
    print()
    if fails:
        print(f"{len(fails)}/{len(blocks)} blocks FAILED:")
        for i, block, err in fails:
            print(f"--- block {i} ---")
            print(block[:2000])
            print("--- stderr (tail) ---")
            print(err[-3000:])
        sys.exit(1)
    else:
        print(f"ALL {len(blocks)} blocks passed independently.")

if __name__ == "__main__":
    main()
