import json
import argparse
from typing import Set

import pandas as pd


def load_non_false_questions(jsonl_path: str) -> Set[str]:
    """
    从 jsonl 中提取所有 success != False 的 question
    包括：
    - success == True
    - success == "valid"
    - 其他任何不严格等于 False 的值
    """
    questions = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[JSONL 解析失败] 第 {line_num} 行: {e}")
                continue

            success = item.get("success", False)
            if success is not False:
                question = item.get("question", "")
                if isinstance(question, str) and question.strip():
                    questions.add(question.strip())

    return questions


def extract_question_from_extra_info(extra_info) -> str:
    """
    从 parquet 的 extra_info 字段中提取 question
    兼容 extra_info 是 dict 或 JSON 字符串
    """
    if isinstance(extra_info, dict):
        question = extra_info.get("question", "")
        return question.strip() if isinstance(question, str) else ""

    if isinstance(extra_info, str):
        try:
            obj = json.loads(extra_info)
            question = obj.get("question", "")
            return question.strip() if isinstance(question, str) else ""
        except Exception:
            return ""

    return ""


def filter_parquet_excluding_questions(
    success_jsonl_path: str,
    parquet_path: str,
    output_parquet_path: str
):
    """
    从 success != False 的 jsonl 中提取 question，
    然后从 parquet 中排除这些 question，对剩余样本写入新 parquet
    """
    exclude_questions = load_non_false_questions(success_jsonl_path)
    print(f"需要排除的 question 数量: {len(exclude_questions)}")

    df = pd.read_parquet(parquet_path)
    print(f"原 parquet 行数: {len(df)}")

    df["__question__"] = df["extra_info"].map(extract_question_from_extra_info)

    filtered_df = df[~df["__question__"].isin(exclude_questions)].copy()
    filtered_df = filtered_df.drop(columns="__question__")

    print(f"排除后保留的 parquet 行数: {len(filtered_df)}")

    filtered_df.to_parquet(output_parquet_path, index=False)
    print(f"已写入新 parquet 文件: {output_parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--success_jsonl_path",
        type=str,
        required=True,
        help="包含 success != False 样本的 jsonl 文件"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="原始 parquet 文件"
    )
    parser.add_argument(
        "--output_parquet_path",
        type=str,
        required=True,
        help="输出的新 parquet 文件"
    )

    args = parser.parse_args()

    filter_parquet_excluding_questions(
        success_jsonl_path=args.success_jsonl_path,
        parquet_path=args.parquet_path,
        output_parquet_path=args.output_parquet_path
    )


"""
python filter_true_questions.py \
  --success_jsonl_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/v1/sft_new_success_result.jsonl \
  --parquet_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/sft-hotpot2500-nq2500-seed.parquet \
  --output_parquet_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/remaining.parquet
"""