import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def extract_objective_from_messages(messages: List[dict]) -> str:
    """
    从主体数据的 messages 中提取 user 里的 Objective
    例如：
    Objective: xxx
    Observation: yyy
    提取出 xxx
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(
                r'Objective:\s*(.*?)\s*(?:\nObservation:|$)',
                content,
                re.DOTALL
            )
            if match:
                return match.group(1).strip()
    return ""


def load_main_data(main_file: str) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    读取主体数据，并建立 objective -> 多条数据 的映射
    因为你说“多条同 objective 的数据构成一个任务的数据”，
    所以这里一个 objective 可能对应多条主体数据。
    """
    all_data = []
    objective_to_samples = {}

    with open(main_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[主体文件解析失败] 第 {line_num} 行: {e}")
                continue

            objective = extract_objective_from_messages(data.get("messages", []))
            if not objective:
                print(f"[警告] 主体文件第 {line_num} 行未提取到 Objective，已跳过")
                continue

            all_data.append(data)
            objective_to_samples.setdefault(objective, []).append(data)

    return all_data, objective_to_samples


def is_success_result(result_item: dict) -> bool:
    """
    True 或 "valid" 都算成功，其余都算失败。
    """
    success_raw = result_item.get("success", False)
    return (success_raw is True) or (success_raw == "valid")


def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def split_result_jsonl_by_success(
    result_file: str,
    out_result_success: str,
    out_result_fail: str
):
    """
    功能 1：只根据结果 jsonl 中每条数据的 success 字段，
    将结果 jsonl 拆成成功/失败两个文件。
    """
    result_success_count = 0
    result_fail_count = 0

    ensure_parent_dir(out_result_success)
    ensure_parent_dir(out_result_fail)

    with open(result_file, "r", encoding="utf-8") as fr, \
         open(out_result_success, "w", encoding="utf-8") as frs, \
         open(out_result_fail, "w", encoding="utf-8") as frf:

        for line_num, line in enumerate(fr, 1):
            line = line.strip()
            if not line:
                continue

            try:
                result_item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[结果文件解析失败] 第 {line_num} 行: {e}")
                continue

            if is_success_result(result_item):
                frs.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                result_success_count += 1
            else:
                frf.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                result_fail_count += 1

    print("===== 结果文件拆分完成 =====")
    print(f"结果成功数据条数: {result_success_count}")
    print(f"结果失败数据条数: {result_fail_count}")
    print(f"结果成功文件: {out_result_success}")
    print(f"结果失败文件: {out_result_fail}")


def split_main_jsonl_by_result_success(
    main_file: str,
    result_file: str,
    out_main_success: str,
    out_main_fail: str
):
    """
    功能 2：给定带 success 字段的结果 jsonl，用 result.question 去匹配主体数据
    messages 里的 Objective，并按结果 success 将主体数据拆成成功/失败两个文件。
    """
    _, objective_to_samples = load_main_data(main_file)

    main_success_count = 0
    main_fail_count = 0
    unmatched_result_count = 0

    ensure_parent_dir(out_main_success)
    ensure_parent_dir(out_main_fail)

    with open(result_file, "r", encoding="utf-8") as fr, \
         open(out_main_success, "w", encoding="utf-8") as fms, \
         open(out_main_fail, "w", encoding="utf-8") as fmf:

        for line_num, line in enumerate(fr, 1):
            line = line.strip()
            if not line:
                continue

            try:
                result_item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[结果文件解析失败] 第 {line_num} 行: {e}")
                continue

            question = result_item.get("question", "").strip()
            success = is_success_result(result_item)

            matched_samples = objective_to_samples.get(question, [])

            if not matched_samples:
                unmatched_result_count += 1
                print(f"[未匹配] 结果文件第 {line_num} 行 question 未在主体文件中找到对应 Objective:")
                print(f"         {question}")
                continue

            if success:
                for sample in matched_samples:
                    fms.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    main_success_count += 1
            else:
                for sample in matched_samples:
                    fmf.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    main_fail_count += 1

    print("===== 主体数据拆分完成 =====")
    print(f"主体成功数据条数: {main_success_count}")
    print(f"主体失败数据条数: {main_fail_count}")
    print(f"未匹配的结果条数: {unmatched_result_count}")
    print(f"主体成功文件: {out_main_success}")
    print(f"主体失败文件: {out_main_fail}")


def split_files_by_success(
    main_file: str,
    result_file: str,
    out_main_success: str,
    out_main_fail: str,
    out_result_success: str,
    out_result_fail: str
):
    """
    兼容旧功能：同时拆分结果 jsonl 和主体数据 jsonl。
    """
    split_result_jsonl_by_success(
        result_file=result_file,
        out_result_success=out_result_success,
        out_result_fail=out_result_fail
    )
    split_main_jsonl_by_result_success(
        main_file=main_file,
        result_file=result_file,
        out_main_success=out_main_success,
        out_main_fail=out_main_fail
    )


def require_args(args, names: List[str]):
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise SystemExit(f"缺少必要参数: {', '.join('--' + name for name in missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按结果 jsonl 的 success 字段拆分结果文件，或用结果文件拆分主体数据文件。"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["result", "main", "all"],
        required=True,
        help="result=只拆结果 jsonl；main=只拆主体数据 jsonl；all=两个都执行"
    )
    parser.add_argument("--main_file", type=str, default=None, help="主体数据 jsonl 文件")
    parser.add_argument("--result_file", type=str, default=None, help="带 success 字段的结果 jsonl 文件")
    parser.add_argument("--out_main_success", type=str, default=None, help="输出：主体成功数据")
    parser.add_argument("--out_main_fail", type=str, default=None, help="输出：主体失败数据")
    parser.add_argument("--out_result_success", type=str, default=None, help="输出：结果成功数据")
    parser.add_argument("--out_result_fail", type=str, default=None, help="输出：结果失败数据")

    args = parser.parse_args()

    if args.mode == "result":
        require_args(args, ["result_file", "out_result_success", "out_result_fail"])
        split_result_jsonl_by_success(
            result_file=args.result_file,
            out_result_success=args.out_result_success,
            out_result_fail=args.out_result_fail
        )
    elif args.mode == "main":
        require_args(args, ["main_file", "result_file", "out_main_success", "out_main_fail"])
        split_main_jsonl_by_result_success(
            main_file=args.main_file,
            result_file=args.result_file,
            out_main_success=args.out_main_success,
            out_main_fail=args.out_main_fail
        )
    else:
        require_args(
            args,
            [
                "main_file",
                "result_file",
                "out_main_success",
                "out_main_fail",
                "out_result_success",
                "out_result_fail",
            ]
        )
        split_files_by_success(
            main_file=args.main_file,
            result_file=args.result_file,
            out_main_success=args.out_main_success,
            out_main_fail=args.out_main_fail,
            out_result_success=args.out_result_success,
            out_result_fail=args.out_result_fail
        )


"""
1. 只拆结果 jsonl：

python split.py \
  --mode result \
  --result_file /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft-hotpot10000-nq10000_results.jsonl \
  --out_result_success /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success_result.jsonl \
  --out_result_fail /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_fail_result.jsonl

2. 根据指定结果 jsonl 拆主体数据 jsonl：

python split.py \
  --mode main \
  --main_file /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft-hotpot10000-nq10000.jsonl \
  --result_file /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft-hotpot10000-nq10000_results.jsonl \
  --out_main_success /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success.jsonl \
  --out_main_fail /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_fail.jsonl

3. 兼容旧行为：结果和主体数据都拆：

python split.py \
  --mode all \
  --main_file /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft-hotpot10000-nq10000.jsonl \
  --result_file /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft-hotpot10000-nq10000_results.jsonl \
  --out_main_success /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success.jsonl \
  --out_main_fail /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_fail.jsonl \
  --out_result_success /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_success_result.jsonl \
  --out_result_fail /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_fail_result.jsonl
"""
