import re
import subprocess
import sys
from pathlib import Path

def extract_blocks(md_path):
    text = Path(md_path).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    return blocks

def main():
    md_path = sys.argv[1]
    blocks = extract_blocks(md_path)
    print(f"Found {len(blocks)} python code blocks in {md_path}")
    fails = []
    for i, block in enumerate(blocks, 1):
        result = subprocess.run(
            [sys.executable, "-c", block],
            capture_output=True, text=True, timeout=120,
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
            print(block)
            print("--- stderr ---")
            print(err)
        sys.exit(1)
    else:
        print(f"ALL {len(blocks)} blocks passed independently.")

if __name__ == "__main__":
    main()
