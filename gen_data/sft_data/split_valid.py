import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


SYSTEM_PROMPT = """You are a strict QA answer labeler.

Given a question, a ground-truth answer, and a predicted final answer, assign
exactly one label for the sample's success field: true, valid, or false.

Field-specific instructions:
- question: Determines the target being asked for and the expected answer type.
- ground_truth: Reference answer. It may be a canonical answer, an option
  content, an alias, or one representative form of a concept.
- final_answer: The model's answer. Judge only whether this answer should be
  labeled true, valid, or false.

Labels:
- true: final_answer is semantically equivalent to ground_truth for this
  question. This includes aliases, abbreviations, minor spelling differences,
  omitted middle names, option labels that map to the ground truth, option
  contents that match the ground truth, or different concepts that clearly
  refer to the same real-world thing.
- valid: final_answer is a real attempted answer to the same question, but it
  is not equivalent to ground_truth. Use valid for any plausible but wrong answer
  with the expected answer type. This includes wrong dates, wrong people, wrong
  places, wrong organizations, wrong titles, wrong options, overly broad or
  overly narrow entity names, and answers with extra or missing entities.
- false: final_answer is not a real attempted answer to the same question. Use
  false only for N/A, empty answers, browser action residues such as
  click/type/scroll commands, reasoning artifacts, format fragments, generic
  evaluation phrases, conceptual meta-answers like "same concept" or "not enough
  information", or an answer to a different question.

Important distinction:
- The label false means "invalid / not answering the question". It does not mean
  "factually wrong".
- If final_answer is factually wrong but still tries to answer the original
  question, label it valid, not false.
- If final_answer has the same expected answer type as ground_truth, it is
  usually valid unless it is clearly about a different question.
- If you are deciding between valid and false, choose valid when final_answer
  looks like an answer candidate for the original question.

Examples:
- question: "Who is best known as the singer and front woman for the band T'Pau?"
  ground_truth: "Carol Ann Decker"
  final_answer: "Carol Decker"
  output: {"success": "true"}
- question: "when did ariana grande get signed to republic records"
  ground_truth: "2013"
  final_answer: "2011"
  output: {"success": "valid"}
- question: "Which genus includes more species of flowering plants, Damasonium or Selinum?"
  ground_truth: "Damasonium"
  final_answer: "Selinum"
  output: {"success": "valid"}
- question: "who sang suite judy blue eyes at woodstock"
  ground_truth: "Crosby, Stills & Nash"
  final_answer: "Crosby, Stills, Nash & Young"
  output: {"success": "valid"}
- question: "where is the next world cup soccer being held"
  ground_truth: "Qatar"
  final_answer: "Canada, Mexico, and the United States"
  output: {"success": "valid"}
- question: "while writing time the number after the two dots stands for"
  ground_truth: "minutes"
  final_answer: "click [217] [Full"
  output: {"success": "false"}
- question: "where is if you are the one filmed"
  ground_truth: "Nanjing, China"
  final_answer: "N/A"
  output: {"success": "false"}

Return only a JSON object in this exact format:
{"success": "true"}
or
{"success": "valid"}
or
{"success": "false"}
"""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(x) for x in value)
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\"'`\[]+|[\"'`\]]+$", "", value)
    return value.strip()


def exact_match(final_answer: Any, ground_truth: Any) -> bool:
    final_text = normalize_text(final_answer)
    gt = ground_truth
    if isinstance(gt, (list, tuple, set)):
        return any(final_text == normalize_text(answer) for answer in gt)
    return final_text == normalize_text(gt)


class LLMJudge:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
        max_retries: int = 3,
        temperature: float = 0.0,
        json_mode: bool = True,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.json_mode = json_mode
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def judge(self, question: Any, final_answer: Any, ground_truth: Any) -> Tuple[str, str]:
        label, _, raw_content = self.judge_with_error(question, final_answer, ground_truth)
        return label, raw_content

    def judge_with_error(
        self, question: Any, final_answer: Any, ground_truth: Any
    ) -> Tuple[str, Optional[str], str]:
        user_prompt = (
            "Assign exactly one success label for this sample: true, valid, or false.\n\n"
            "Remember: false means invalid or answering a different question. "
            "A wrong but plausible answer to this same question must be valid.\n\n"
            "FIELD question:\n"
            f"{json.dumps(question, ensure_ascii=False)}\n\n"
            "FIELD ground_truth:\n"
            f"{json.dumps(ground_truth, ensure_ascii=False)}\n\n"
            "FIELD final_answer:\n"
            f"{json.dumps(final_answer, ensure_ascii=False)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 128,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.base_url + "chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    if self.json_mode and resp.status_code == 400 and "response_format" in resp.text:
                        payload.pop("response_format", None)
                        self.json_mode = False
                        print("[提示] 当前 API 不支持 response_format，已自动关闭 JSON 模式并重试。")
                        continue
                    if resp.status_code in {400, 401, 403, 404}:
                        return "false", last_error, resp.text[:500]
                    time.sleep(min(2 ** attempt, 10))
                    continue

                data = resp.json()
                content = extract_response_content(data)
                if not content.strip():
                    raw_response = json.dumps(data, ensure_ascii=False)[:3000]
                    return "false", f"empty message.content: {raw_response}", raw_response
                return parse_success_label(content), None, content
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, 10))

        return "false", last_error, ""


def extract_response_content(data: Dict[str, Any]) -> str:
    try:
        message = data["choices"][0].get("message", {})
    except Exception:
        return ""

    content = message.get("content", "")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    parts.append(str(text))
            elif part:
                parts.append(str(part))
        if parts:
            return "\n".join(parts)

    for key in ("reasoning_content", "reasoning", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_deepseek_model(base_url: str, model: str) -> str:
    if "deepseek.com" not in base_url.lower():
        return model
    candidates = {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-chat",
        "deepseek-reasoner",
    }
    lower_model = model.lower()
    if model != lower_model and lower_model in candidates:
        print(f"[提示] DeepSeek API 模型名区分大小写，已将 {model} 修正为 {lower_model}")
        return lower_model
    return model


def preflight_llm(judge: LLMJudge, skip_preflight: bool) -> None:
    if skip_preflight:
        return

    label, error, raw_content = judge.judge_with_error(
        question="Who is best known as the singer and front woman for the band T'Pau?",
        ground_truth="Carol Ann Decker",
        final_answer="Carol Decker",
    )
    if error:
        print("[LLM 预检查失败] 无法正常调用模型，已停止处理，避免整批数据全部失败。")
        print(f"[错误信息] {error}")
        print("[建议] DeepSeek 官方 API 模型名示例: deepseek-v4-flash, deepseek-v4-pro, deepseek-chat")
        sys.exit(1)
    print(f"[LLM 预检查通过] 接口连通，示例标签: {label}")
    print(f"[LLM 预检查原始输出] {raw_content}")
    if label != "true":
        print("[LLM 预检查失败] 示例应判为 true，但当前模型/提示词/解析得到的不是 true。")
        print("请先检查上面的原始输出，或用 --skip_preflight 跳过该检查。")
        sys.exit(1)


def parse_success_label(content: str) -> str:
    content = content.strip()
    candidates = [content]
    json_match = re.search(r"\{.*?\}", content, flags=re.DOTALL)
    if json_match and json_match.group(0) != content:
        candidates.append(json_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            raw_label = data.get("success", data.get("label", data.get("result", "")))
            label = normalize_success_label(raw_label)
            if label:
                return label
        except Exception:
            pass

    return normalize_success_label(content) or "false"


def normalize_success_label(value: Any) -> Optional[str]:
    if value is True:
        return "true"
    if value is False or value is None:
        return "false"

    text = str(value).strip().lower()
    text = re.sub(r"^[\"'`\s{:\[]+|[\"'`\s}\]]+$", "", text)
    text = text.replace("success", "").replace("label", "").replace("result", "")
    text = re.sub(r"[^a-z]+", " ", text).strip()
    tokens = text.split()

    if "valid" in tokens:
        return "valid"
    if "true" in tokens or "correct" in tokens:
        return "true"
    if "false" in tokens or "invalid" in tokens or "incorrect" in tokens:
        return "false"
    return None


def is_success(item: Dict[str, Any]) -> bool:
    success = item.get("success", False)
    return success is True or success == "valid"


def load_jsonl(path: str) -> List[Tuple[int, Dict[str, Any]]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_num, json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"[JSONL 解析失败] 第 {line_num} 行: {exc}")
    return records


def judge_one(
    line_num: int,
    item: Dict[str, Any],
    judge: LLMJudge,
    use_exact_shortcut: bool,
    recheck_success: bool,
) -> Tuple[int, Dict[str, Any], bool, str, str]:
    if is_success(item) and not recheck_success:
        return line_num, item, True, "already_success", ""

    final_answer = item.get("final_answer", "")
    ground_truth = item.get("ground_truth", "")

    if use_exact_shortcut and exact_match(final_answer, ground_truth):
        item["success"] = True
        return line_num, item, True, "exact_match", ""

    label, raw_content = judge.judge(
        question=item.get("question", ""),
        final_answer=final_answer,
        ground_truth=ground_truth,
    )
    if label == "true":
        item["success"] = True
        return line_num, item, True, "llm_true", raw_content
    if label == "valid":
        item["success"] = "valid"
        return line_num, item, True, "llm_valid", raw_content

    item["success"] = False
    return line_num, item, False, "llm_false", raw_content


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_debug_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_valid_by_llm(
    input_file: str,
    out_success: str,
    out_fail: str,
    base_url: str,
    model: str,
    api_key: str,
    workers: int = 1,
    timeout: int = 120,
    max_retries: int = 3,
    use_exact_shortcut: bool = False,
    recheck_success: bool = False,
    limit: Optional[int] = None,
    skip_preflight: bool = False,
    debug_output: Optional[str] = None,
) -> None:
    records = load_jsonl(input_file)
    if limit is not None:
        records = records[:limit]

    model = normalize_deepseek_model(base_url, model)
    judge = LLMJudge(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    preflight_llm(judge, skip_preflight)

    results: List[Tuple[int, Dict[str, Any], bool, str, str]] = []
    total = len(records)

    if workers <= 1:
        for idx, (line_num, item) in enumerate(records, 1):
            results.append(
                judge_one(line_num, item, judge, use_exact_shortcut, recheck_success)
            )
            if idx % 50 == 0 or idx == total:
                print(f"[进度] {idx}/{total}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    judge_one,
                    line_num,
                    item,
                    judge,
                    use_exact_shortcut,
                    recheck_success,
                )
                for line_num, item in records
            ]
            for idx, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if idx % 50 == 0 or idx == total:
                    print(f"[进度] {idx}/{total}")

    results.sort(key=lambda x: x[0])

    success_items = []
    fail_items = []
    stat = {
        "already_success": 0,
        "exact_match": 0,
        "llm_true": 0,
        "llm_valid": 0,
        "llm_false": 0,
    }

    debug_rows = []

    for line_num, item, matched, reason, raw_content in results:
        stat[reason] = stat.get(reason, 0) + 1
        if matched:
            success_items.append(item)
        else:
            fail_items.append(item)
        if debug_output:
            debug_rows.append(
                {
                    "line_num": line_num,
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "ground_truth": item.get("ground_truth"),
                    "final_answer": item.get("final_answer"),
                    "parsed_success": item.get("success"),
                    "reason": reason,
                    "raw_llm_output": raw_content,
                }
            )

    write_jsonl(out_success, success_items)
    write_jsonl(out_fail, fail_items)
    if debug_output:
        write_debug_jsonl(debug_output, debug_rows)

    print("===== 处理完成 =====")
    print(f"输入数据条数: {total}")
    print(f"保留数据条数(success=true 或 success='valid'): {len(success_items)}")
    print(f"失败数据条数(success=false): {len(fail_items)}")
    print(f"已成功跳过条数: {stat['already_success']}")
    print(f"精确匹配修正条数: {stat['exact_match']}")
    print(f"LLM 输出 true 条数: {stat['llm_true']}")
    print(f"LLM 输出 valid 条数: {stat['llm_valid']}")
    print(f"LLM 输出 false 条数: {stat['llm_false']}")
    print(f"保留输出文件: {out_success}")
    print(f"失败输出文件: {out_fail}")
    if debug_output:
        print(f"调试输出文件: {debug_output}")


def default_output_path(input_file: str, suffix: str) -> str:
    path = Path(input_file)
    return str(path.with_name(path.stem + suffix + path.suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 LLM 一次调用输出 success=true/false/valid 三分类，并拆分保留/失败 jsonl。"
    )
    parser.add_argument("--input_file", type=str, required=True, help="输入 jsonl 文件")
    parser.add_argument(
        "--out_success",
        type=str,
        default=None,
        help="输出：success=true 或 success='valid' 的数据 jsonl，默认 <input>_valid.jsonl",
    )
    parser.add_argument(
        "--out_fail",
        type=str,
        default=None,
        help="输出：success=false 的数据 jsonl，默认 <input>_invalid.jsonl",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8008/v1"),
        help="OpenAI-compatible API base url",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "custom-llm"),
        help="LLM 模型名",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key，本地 vLLM 可使用 EMPTY",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发请求数")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时时间")
    parser.add_argument("--max_retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条，调试用")
    parser.add_argument(
        "--debug_output",
        type=str,
        default=None,
        help="可选：输出每条样本的原始 LLM 回复和解析标签，便于调试",
    )
    parser.add_argument(
        "--skip_preflight",
        action="store_true",
        help="跳过 LLM 连通性预检查",
    )
    parser.add_argument(
        "--exact_shortcut",
        action="store_true",
        help="开启精确匹配捷径，final_answer 与 ground_truth 完全相同的样本不调用 LLM",
    )
    parser.add_argument(
        "--recheck_success",
        action="store_true",
        help="对已经 success=True 或 success='valid' 的样本也重新判断",
    )

    args = parser.parse_args()

    split_valid_by_llm(
        input_file=args.input_file,
        out_success=args.out_success or default_output_path(args.input_file, "_valid"),
        out_fail=args.out_fail or default_output_path(args.input_file, "_invalid"),
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
        use_exact_shortcut=args.exact_shortcut,
        recheck_success=args.recheck_success,
        limit=args.limit,
        skip_preflight=args.skip_preflight,
        debug_output=args.debug_output,
    )


"""
示例：
export OPENAI_API_KEY="sk-5b242d486cf743029a466acd3924046c"

python split_valid.py \
  --input_file /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_new_fail_result.jsonl \
  --out_success /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_valid_result.jsonl \
  --out_fail /data/yutao/lzt/BrowserAgent_v2/gen_data/sft_data/sft-hotpot10000-nq10000/sft_invalid_result.jsonl \
  --base_url https://api.deepseek.com \
  --model deepseek-v4-flash \
  --workers 4 \
  --limit 100
"""
