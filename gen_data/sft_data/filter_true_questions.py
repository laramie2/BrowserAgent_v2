import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_true_success(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def load_non_false_questions(result_jsonl_path: str) -> Set[str]:
    """
    从结果 jsonl 中提取所有 success 非 false 的 question。
    例如 success 为 true、"true"、"valid" 都会被保留。
    """
    questions = set()

    with open(result_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[结果文件解析失败] 第 {line_num} 行: {e}")
                continue

            if not is_true_success(item.get("success", False)):
                continue

            question = normalize_text(item.get("question", ""))
            if question:
                questions.add(question)

    return questions


def load_non_false_ids(result_jsonl_path: str) -> Set[str]:
    """
    若结果 jsonl 和生成数据都有 id，则优先也支持通过 id 匹配。
    """
    ids = set()

    with open(result_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[结果文件解析失败] 第 {line_num} 行: {e}")
                continue

            if not is_true_success(item.get("success", False)):
                continue

            item_id = normalize_text(item.get("id", ""))
            if item_id:
                ids.add(item_id)

    return ids


def extract_question_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    从生成数据 messages 中提取 user 里的 Objective。
    参考 split.py 中对主体数据的识别逻辑。
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        match = re.search(
            r"Objective:\s*(.*?)\s*(?:\nObservation:|$)",
            content,
            re.DOTALL,
        )
        if match:
            return normalize_text(match.group(1))

    return ""


def extract_question_from_extra_info(extra_info: Any) -> str:
    """
    从 parquet 的 extra_info 字段中提取 question。
    兼容 extra_info 是 dict 或 JSON 字符串。
    """
    if isinstance(extra_info, dict):
        return normalize_text(extra_info.get("question", ""))

    if isinstance(extra_info, str):
        try:
            obj = json.loads(extra_info)
        except Exception:
            return ""
        return normalize_text(obj.get("question", ""))

    return ""


def extract_id_from_extra_info(extra_info: Any) -> str:
    if isinstance(extra_info, dict):
        return normalize_text(extra_info.get("id", ""))

    if isinstance(extra_info, str):
        try:
            obj = json.loads(extra_info)
        except Exception:
            return ""
        return normalize_text(obj.get("id", ""))

    return ""


def extract_question_from_jsonl_item(item: Dict[str, Any]) -> str:
    question = normalize_text(item.get("question", ""))
    if question:
        return question

    messages = item.get("messages", [])
    if isinstance(messages, list):
        question = extract_question_from_messages(messages)
        if question:
            return question

    return extract_question_from_extra_info(item.get("extra_info"))


def extract_id_from_jsonl_item(item: Dict[str, Any]) -> str:
    item_id = normalize_text(item.get("id", ""))
    if item_id:
        return item_id
    return extract_id_from_extra_info(item.get("extra_info"))


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def filter_jsonl_by_result_success(
    result_jsonl_path: str,
    generated_data_path: str,
    output_path: str,
) -> None:
    keep_questions = load_non_false_questions(result_jsonl_path)
    keep_ids = load_non_false_ids(result_jsonl_path)
    print(f"结果文件中 success 非 false 的 question 数量: {len(keep_questions)}")
    print(f"结果文件中 success 非 false 的 id 数量: {len(keep_ids)}")

    total_count = 0
    keep_count = 0
    unmatched_count = 0

    ensure_parent_dir(output_path)
    with open(generated_data_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total_count += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[生成数据解析失败] 第 {line_num} 行: {e}")
                continue

            item_id = extract_id_from_jsonl_item(item)
            question = extract_question_from_jsonl_item(item)
            matched = (item_id and item_id in keep_ids) or (
                question and question in keep_questions
            )

            if matched:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                keep_count += 1
            else:
                unmatched_count += 1

    print("===== JSONL 生成数据提取完成 =====")
    print(f"生成数据总条数: {total_count}")
    print(f"提取条数: {keep_count}")
    print(f"未提取条数: {unmatched_count}")
    print(f"输出文件: {output_path}")


def filter_parquet_by_result_success(
    result_jsonl_path: str,
    generated_data_path: str,
    output_path: str,
) -> None:
    """
    从结果 jsonl 中提取 success 非 false 的 question/id，
    然后从 parquet 生成数据中提取对应样本。
    """
    keep_questions = load_non_false_questions(result_jsonl_path)
    keep_ids = load_non_false_ids(result_jsonl_path)
    print(f"结果文件中 success 非 false 的 question 数量: {len(keep_questions)}")
    print(f"结果文件中 success 非 false 的 id 数量: {len(keep_ids)}")

    df = pd.read_parquet(generated_data_path)
    print(f"原 parquet 行数: {len(df)}")

    matched_mask = pd.Series(False, index=df.index)

    if "id" in df.columns and keep_ids:
        matched_mask = matched_mask | df["id"].map(lambda x: normalize_text(x) in keep_ids)

    if "question" in df.columns and keep_questions:
        matched_mask = matched_mask | df["question"].map(
            lambda x: normalize_text(x) in keep_questions
        )

    if "extra_info" in df.columns:
        if keep_questions:
            matched_mask = matched_mask | df["extra_info"].map(
                lambda x: extract_question_from_extra_info(x) in keep_questions
            )
        if keep_ids:
            matched_mask = matched_mask | df["extra_info"].map(
                lambda x: extract_id_from_extra_info(x) in keep_ids
            )

    filtered_df = df[matched_mask].copy()
    print(f"提取后的 parquet 行数: {len(filtered_df)}")

    ensure_parent_dir(output_path)
    filtered_df.to_parquet(output_path, index=False)
    print(f"已写入新 parquet 文件: {output_path}")


def filter_generated_data_by_result_success(
    result_jsonl_path: str,
    generated_data_path: str,
    output_path: str,
) -> None:
    suffix = Path(generated_data_path).suffix.lower()
    if suffix == ".jsonl":
        filter_jsonl_by_result_success(
            result_jsonl_path=result_jsonl_path,
            generated_data_path=generated_data_path,
            output_path=output_path,
        )
    elif suffix == ".parquet":
        filter_parquet_by_result_success(
            result_jsonl_path=result_jsonl_path,
            generated_data_path=generated_data_path,
            output_path=output_path,
        )
    else:
        raise ValueError(f"不支持的生成数据格式: {generated_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据结果 jsonl 的 success 字段，提取生成数据中 success 非 false 的样本。"
    )
    parser.add_argument(
        "--result_jsonl_path",
        "--success_jsonl_path",
        dest="result_jsonl_path",
        type=str,
        required=True,
        help="带 id/question/ground_truth/final_answer/success 字段的结果 jsonl 文件"
    )
    parser.add_argument(
        "--generated_data_path",
        "--parquet_path",
        dest="generated_data_path",
        type=str,
        required=True,
        help="待提取的生成数据文件，支持 jsonl 或 parquet"
    )
    parser.add_argument(
        "--output_path",
        "--output_parquet_path",
        dest="output_path",
        type=str,
        required=True,
        help="输出文件路径；输入为 jsonl 时输出 jsonl，输入为 parquet 时输出 parquet"
    )

    args = parser.parse_args()

    filter_generated_data_by_result_success(
        result_jsonl_path=args.result_jsonl_path,
        generated_data_path=args.generated_data_path,
        output_path=args.output_path
    )


"""
JSONL 生成数据示例：

python filter_true_questions.py \
  --result_jsonl_path /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success_result.jsonl \
  --generated_data_path /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft-hotpot10000-nq10000.jsonl \
  --output_path /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success.jsonl

Parquet 生成数据示例：

python filter_true_questions.py \
  --result_jsonl_path /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success_result.jsonl \
  --generated_data_path /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/v3/sft-hotpot10000-nq10000-seed.parquet \
  --output_path /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success.parquet
"""
