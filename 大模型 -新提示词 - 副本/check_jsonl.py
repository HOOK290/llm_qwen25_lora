import json

path = "train_code.jsonl"  # 改成你要检查的文件
bad = 0
n = 0

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            obj = json.loads(line)
            for k in ("instruction", "input", "output"):
                if k not in obj:
                    raise ValueError(f"missing key {k}")
            if str(obj["output"]).strip() not in {"1", "2", "3"}:
                raise ValueError(f"bad output={obj['output']!r}")
        except Exception as e:
            bad += 1
            print(f"[Line {i}] {e}")

print(f"Checked {n} lines, bad={bad}")
