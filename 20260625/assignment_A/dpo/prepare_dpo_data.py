from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path


def read_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()


def clean_text(value: object) -> str:
    return str(value).strip()


def synthesize_rejected_code(chosen_code: str) -> str:
    lines = chosen_code.split("\n")
    if len(lines) <= 1:
        return chosen_code + " # syntax error"

    strategy = random.choice(["syntax", "logic", "variable"])

    if strategy == "syntax":
        for i in range(len(lines)):
            if ":" in lines[i]:
                lines[i] = lines[i].replace(":", "", 1)
                return "\n".join(lines)

    elif strategy == "logic":
        if "==" in chosen_code:
            return chosen_code.replace("==", "!=", 1)
        if "<" in chosen_code:
            return chosen_code.replace("<", ">", 1)

    words = re.findall(r"\b[a-zA-Z_]{4,}\b", chosen_code)
    if words:
        target_word = random.choice(words)
        return chosen_code.replace(target_word, target_word + "_bug", 1)

    return chosen_code + " # logic error"


def convert_rows(rows: list[dict]) -> list[dict]:
    converted = []
    clean_count = 0
    synth_count = 0

    for row in rows:
        instruction = clean_text(row.get("prompt", ""))
        chosen = clean_text(row.get("chosen", ""))
        rejected = clean_text(row.get("rejected", ""))

        if not instruction or not chosen:
            continue

        if rejected:
            converted.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
            clean_count += 1

        if rejected and random.random() < 0.4:
            synthetic_rejected = synthesize_rejected_code(chosen)
            if synthetic_rejected != rejected:
                converted.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "chosen": chosen,
                        "rejected": synthetic_rejected,
                    }
                )
                synth_count += 1

    print(
        f"[A3 Info] Training set contain: {clean_count} clean samples + {synth_count} synthesized samples."
    )
    return converted


def convert_effect_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        instruction = clean_text(row.get("prompt", ""))
        chosen = clean_text(row.get("chosen", ""))
        if not instruction or not chosen:
            continue

        converted.append(
            {
                "instruction": instruction,
                "input": "",
                "output": chosen,
            }
        )

    return converted


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    seed = int(os.environ.get("SEED", "42"))
    test_size = int(os.environ.get("TEST_SIZE", "500"))
    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
    source_dir = Path(
        os.environ.get("SOURCE_DIR", "/home/wanghe/20260625/py-dpo-v0.1")
    )
    output_dir = Path(os.environ.get("OUTPUT_DIR", "dpo/data"))

    parquet_files = sorted(source_dir.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted((source_dir / "data").glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")

    all_rows = []
    for parquet_path in parquet_files:
        all_rows.extend(read_parquet(parquet_path))

    rng = random.Random(seed)
    shuffled_rows = list(all_rows)
    rng.shuffle(shuffled_rows)

    raw_test_rows = shuffled_rows[:test_size]
    raw_train_rows = shuffled_rows[test_size:]
    if max_train_samples > 0:
        raw_train_rows = raw_train_rows[:max_train_samples]

    train_rows = convert_rows(raw_train_rows)
    effect_test_rows = convert_effect_rows(raw_test_rows)

    if not train_rows:
        raise RuntimeError("No valid DPO rows were converted.")
    if not effect_test_rows:
        raise RuntimeError("No valid generation test rows were converted.")

    save_json(output_dir / "code_dpo_train.json", train_rows)
    save_json(output_dir / "code_effect_test.json", effect_test_rows)
    save_json(
        output_dir / "dataset_info.json",
        {
            "code_dpo_train": {
                "file_name": "code_dpo_train.json",
                "ranking": True,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            },
            "code_effect_test": {
                "file_name": "code_effect_test.json",
            },
        },
    )

    print(f"Read {len(all_rows)} source rows from {source_dir}")
    print(
        f"Wrote {len(train_rows)} train rows to {output_dir / 'code_dpo_train.json'}"
    )
    print(
        f"Wrote {len(effect_test_rows)} generation test rows to {output_dir / 'code_effect_test.json'}"
    )
    print(f"Wrote dataset registry to {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
