import json
import argparse
import pandas as pd


def extract_question(extra_info):
    """
    从 extra_info 中提取 question
    兼容：
    - dict
    - JSON string
    """
    if isinstance(extra_info, dict):
        q = extra_info.get("question", "")
        return q.strip() if isinstance(q, str) else ""

    if isinstance(extra_info, str):
        try:
            obj = json.loads(extra_info)
            q = obj.get("question", "")
            return q.strip() if isinstance(q, str) else ""
        except Exception:
            return ""

    return ""


def extract_questions_to_json(parquet_path: str, output_json: str, deduplicate: bool = True):
    df = pd.read_parquet(parquet_path)

    questions = df["extra_info"].map(extract_question)

    # 去掉空值
    questions = questions[questions != ""]

    if deduplicate:
        questions = questions.drop_duplicates()

    questions_list = questions.tolist()

    print(f"提取问题数量: {len(questions_list)}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(questions_list, f, ensure_ascii=False, indent=2)

    print(f"已写入: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="不去重"
    )

    args = parser.parse_args()

    extract_questions_to_json(
        parquet_path=args.parquet_path,
        output_json=args.output_json,
        deduplicate=not args.no_dedup
    )

"""
python extract_questions.py \
  --parquet_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/v3/sft-hotpot10000-nq10000-seed.parquet \
  --output_json /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/v3/obj.json
"""
