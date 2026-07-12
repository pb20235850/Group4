import argparse
import os
import glob
import pandas as pd
import json
import ast
import re

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_python_syntax(text):
    """
    只校验 ```python ``` 包裹的代码块。
    无代码块 → 直接放行（可能是纯文本解释型回答）。
    有代码块但解析失败 → 才拦截。
    """
    blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not blocks:
        return True, ""

    for block in blocks:
        if not block.strip():
            continue
        try:
            ast.parse(block)
        except SyntaxError:
            return False, "AST Parse Error in code block"

    return True, ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--min_output_len', type=int, default=20)
    parser.add_argument('--max_output_len', type=int, default=4096)
    parser.add_argument('--remove_duplicates', type=bool, default=True)
    parser.add_argument('--check_syntax', type=bool, default=True)
    args = parser.parse_args()

    parquet_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Missing parquet files in {args.data_dir}")

    input_file = parquet_files[0]
    df_raw = pd.read_parquet(input_file)

    print(f"Read {len(df_raw)} raw rows from 1 parquet file(s).")

    null_counts = df_raw.isnull().sum().to_dict()
    dup_mask = df_raw.duplicated(subset=['instruction', 'input'], keep='first')
    dup_count = dup_mask.sum()

    inst_lens = df_raw['instruction'].fillna('').str.len()
    output_lens = df_raw['output'].fillna('').str.len()

    print("\n====== [Data Quality Statistics - Raw] =====")
    print(f"Total Raw Samples: {len(df_raw)}")
    print(f"Null Ratios: inst={null_counts.get('instruction',0)/len(df_raw):.4f}, input={null_counts.get('input',0)/len(df_raw):.4f}, output={null_counts.get('output',0)/len(df_raw):.4f}")
    print(f"Duplicate Rate (inst+input): {dup_count/len(df_raw):.4f} ({dup_count} rows)")
    print(f"Instruction Length: Mean={inst_lens.mean():.1f}, Max={inst_lens.max()}, Min={inst_lens.min()}")
    print(f"Output Length:      Mean={output_lens.mean():.1f}, Max={output_lens.max()}, Min={output_lens.min()}")
    print("=============================================\n")

    good_cases = []
    bad_cases = []
    seen_keys = set()

    for idx, row in df_raw.iterrows():
        inst_str = str(row.get('instruction', '')).strip()
        input_str = str(row.get('input', '')).strip() if pd.notna(row.get('input')) else ""
        out_str = str(row.get('output', '')).strip()

        sample_dict = {"instruction": inst_str, "input": input_str, "output": out_str}

        if not inst_str or not out_str:
            bad_cases.append({"reason": "Null or Empty Field", "sample": sample_dict})
            continue

        if args.remove_duplicates:
            key = (inst_str, input_str)
            if key in seen_keys:
                bad_cases.append({"reason": "Duplicate Entry", "sample": sample_dict})
                continue
            seen_keys.add(key)

        if len(out_str) < args.min_output_len:
            bad_cases.append({"reason": f"Output too short (<{args.min_output_len})", "sample": sample_dict})
            continue
        if len(out_str) > args.max_output_len:
            bad_cases.append({"reason": f"Output too long (>{args.max_output_len})", "sample": sample_dict})
            continue

        if args.check_syntax:
            is_valid_syntax, err_msg = check_python_syntax(out_str)
            if not is_valid_syntax:
                bad_cases.append({"reason": f"Syntax/Completeness Error: {err_msg}", "sample": sample_dict})
                continue

        good_cases.append(sample_dict)

    print(f"Kept {len(good_cases)} valid unique rows.")

    df_clean = pd.DataFrame(good_cases)
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    total_clean = len(df_clean)
    n_train = int(total_clean * 0.9)
    n_val = int(total_clean * 0.05)

    train_data = df_clean.iloc[:n_train].to_dict(orient='records')
    valid_data = df_clean.iloc[n_train:n_train + n_val].to_dict(orient='records')
    test_data = df_clean.iloc[n_train + n_val:].to_dict(orient='records')

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(train_data, os.path.join(args.output_dir, "code_sft_train.json"))
    save_json(valid_data, os.path.join(args.output_dir, "code_sft_valid.json"))
    save_json(test_data, os.path.join(args.output_dir, "code_sft_test.json"))

    print(f"Wrote train: {len(train_data)} -> sft/data/code_sft_train.json")
    print(f"Wrote valid: {len(valid_data)} -> sft/data/code_sft_valid.json")
    print(f"Wrote test : {len(test_data)} -> sft/data/code_sft_test.json")

    dataset_info = {
        "code_sft_train": {"file_name": "code_sft_train.json"},
        "code_sft_valid": {"file_name": "code_sft_valid.json"},
        "code_sft_test": {"file_name": "code_sft_test.json"}
    }
    save_json(dataset_info, os.path.join(args.output_dir, "dataset_info.json"))
    print("Wrote registry -> sft/data/dataset_info.json")

    save_json(train_data[:50], os.path.join(args.output_dir, "sample_preview.json"))
    save_json(bad_cases, os.path.join(args.output_dir, "bad_cases.json"))

if __name__ == '__main__':
    main()
