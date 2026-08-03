#!/usr/bin/env python3
"""Generate BrowserAgent SFT teacher trajectories with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = PROJECT_ROOT / "prompt/system_prompt_with_history_info_enhance.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "sft/dataset/raw/generated_teacher.jsonl"
DEFAULT_TOOL_URL = "http://127.0.0.1:5000/get_observation"
DEFAULT_BROWSER_URL = (
    "http://localhost:22015/wikipedia_en_all_maxi_2022-05/"
    "A/User:The_other_Kiwix_guy/Landing"
)

USER_PROMPT = """Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""


def safe_nested(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def answer_list(extra_info: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    selected = extra_info.get("selected_answer")
    if selected not in (None, ""):
        values.append(selected)
    golden = safe_nested(extra_info.get("golden_answers", []))
    if isinstance(golden, (list, tuple)):
        values.extend(golden)
    elif golden not in (None, ""):
        values.append(golden)
    return [str(value) for value in values if value not in (None, "")]


def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def answer_matches(prediction: str, answers: Sequence[str]) -> bool:
    prediction = normalized(prediction)
    return bool(
        prediction
        and any(normalized(answer) in prediction for answer in answers if normalized(answer))
    )


def extract_command(text: str) -> str:
    tagged = re.findall(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL)
    if tagged:
        return tagged[-1].strip()
    fenced = re.findall(
        r"```(?:[A-Za-z0-9_+-]+[^\S\n]*\n)?(.+?)```",
        text,
        re.DOTALL,
    )
    return fenced[-1].strip() if fenced else ""


def extract_conclusion(text: str) -> str:
    blocks = re.findall(r"<conclusion>\s*(.*?)\s*</conclusion>", text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def extract_stop_answer(command: str) -> str | None:
    match = re.fullmatch(r"stop\s*\[(.*?)\]", command.strip(), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


class ThreadLocalSession:
    def __init__(self, pool_size: int) -> None:
        self.local = threading.local()
        self.pool_size = pool_size

    def get(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.pool_size,
                pool_maxsize=self.pool_size,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self.local.session = session
        return self.local.session


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        retries: int,
        temperature: float,
        max_tokens: int,
        workers: int,
    ) -> None:
        base_url = base_url.rstrip("/")
        self.endpoint = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sessions = ThreadLocalSession(workers)

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.sessions.get().post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                message = body["choices"][0]["message"]
                content = str(message.get("content") or "")
                reasoning = str(message.get("reasoning_content") or "")
                if reasoning and "<think>" not in content:
                    content = f"<think>\n{reasoning}\n</think>\n{content}"
                if not content.strip():
                    raise ValueError("OpenAI-compatible endpoint returned empty content")
                usage = body.get("usage") or {}
                return content, {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                }
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
                last_error = error
                if attempt >= self.retries:
                    break
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"teacher request failed after {self.retries + 1} attempts: {last_error}")


class BrowserToolClient:
    def __init__(self, url: str, browser_url: str, timeout: float, workers: int) -> None:
        self.url = url
        self.browser_url = browser_url
        self.timeout = timeout
        self.sessions = ThreadLocalSession(workers)

    def step(self, trajectory_id: str, action: str, finish: bool) -> str:
        response = self.sessions.get().post(
            self.url,
            json={
                "trajectory_ids": [trajectory_id],
                "actions": [action],
                "finish": [finish],
                "extra_fields": [{"url": self.browser_url}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        observations = body.get("observations") or []
        raw = observations[0] if observations else ""
        text = json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)
        if "Observation:\n" in text:
            text = text.split("Observation:\n", 1)[1]
            if "\nParsed Previous Action:" in text:
                text = text.split("\nParsed Previous Action:", 1)[0]
        return text


@dataclass(frozen=True)
class Task:
    source_index: int
    question: str
    answers: list[str]


@dataclass(frozen=True)
class GenerationConfig:
    system_prompt: str
    max_steps: int
    subset: str


def format_record(
    config: GenerationConfig,
    user_prompt: str,
    response: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": config.system_prompt.strip()},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ],
        "subset": config.subset,
        "stage": "sft",
        "token_usage": usage,
    }


def run_task(
    task: Task,
    config: GenerationConfig,
    teacher: OpenAIChatClient,
    browser: BrowserToolClient,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    trajectory_id = str(uuid.uuid4())
    records: list[dict[str, Any]] = []
    history_actions: list[str] = []
    history_info: list[str] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    closed = False

    try:
        observation = browser.step(trajectory_id, "", False)
        for step in range(1, config.max_steps + 1):
            user_prompt = USER_PROMPT.format(
                task.question,
                observation,
                "\n".join(history_actions),
                "\n".join(history_info),
            )
            response, usage = teacher.generate(config.system_prompt, user_prompt)
            for key in totals:
                totals[key] += usage[key]
            records.append(format_record(config, user_prompt, response, usage))

            command = extract_command(response)
            if not command:
                raise ValueError("teacher response does not contain an action")
            conclusion = extract_conclusion(response)
            history_actions.append(command)
            if conclusion:
                history_info.append(conclusion)

            final_answer = extract_stop_answer(command)
            if final_answer is not None:
                browser.step(trajectory_id, response, True)
                closed = True
                success = answer_matches(final_answer, task.answers)
                return records, {
                    "source_index": task.source_index,
                    "trajectory_id": trajectory_id,
                    "question": task.question,
                    "ground_truth": task.answers,
                    "final_answer": final_answer,
                    "success": success,
                    "status": "finished",
                    "steps": step,
                }, totals
            observation = browser.step(trajectory_id, response, False)

        browser.step(trajectory_id, "", True)
        closed = True
        return records, {
            "source_index": task.source_index,
            "trajectory_id": trajectory_id,
            "question": task.question,
            "ground_truth": task.answers,
            "final_answer": "",
            "success": False,
            "status": "max_steps",
            "steps": config.max_steps,
        }, totals
    except Exception as error:
        return records, {
            "source_index": task.source_index,
            "trajectory_id": trajectory_id,
            "question": task.question,
            "ground_truth": task.answers,
            "final_answer": "",
            "success": False,
            "status": "error",
            "error": repr(error),
            "steps": len(records),
        }, totals
    finally:
        if not closed:
            try:
                browser.step(trajectory_id, "", True)
            except Exception:
                pass


def load_tasks(path: Path, max_samples: int | None, seed: int) -> list[Task]:
    frame = pd.read_parquet(path)
    if max_samples is not None and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=seed)
    tasks: list[Task] = []
    for source_index, row in frame.iterrows():
        extra = safe_nested(row.get("extra_info", {}))
        if not isinstance(extra, dict):
            continue
        question = str(extra.get("question") or "").strip()
        answers = answer_list(extra)
        if question and answers:
            tasks.append(Task(int(source_index), question, answers))
    return tasks


def completed_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["source_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", "--input", dest="data_path", type=Path, required=True)
    parser.add_argument("--output-file", "--output", dest="output_file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--tool-server-url", default=os.environ.get("TOOL_SERVER_URL", DEFAULT_TOOL_URL))
    parser.add_argument("--browser-url", default=os.environ.get("BROWSER_URL", DEFAULT_BROWSER_URL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--tool-timeout", type=float, default=1200.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--subset", default="browseragent_teacher")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.base_url:
        parser.error("set OPENAI_BASE_URL or pass --base-url")
    if not args.model:
        parser.error("set OPENAI_MODEL or pass --model")
    if not args.data_path.is_file():
        parser.error(f"seed parquet does not exist: {args.data_path}")
    if not args.system_prompt.is_file():
        parser.error(f"system prompt does not exist: {args.system_prompt}")
    if args.workers < 1 or args.max_steps < 1 or args.max_retries < 0:
        parser.error("workers/max-steps must be positive and max-retries non-negative")

    output_file = args.output_file.resolve()
    result_file = (
        args.result_file.resolve()
        if args.result_file
        else output_file.with_name(f"{output_file.stem}_results.jsonl")
    )
    if output_file.exists() and not result_file.exists() and not args.overwrite:
        parser.error(
            f"output exists but resume metadata is missing: {result_file}; "
            "pass --overwrite to start again or --result-file to select it"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_file.write_text("", encoding="utf-8")
        result_file.write_text("", encoding="utf-8")

    tasks = load_tasks(args.data_path, args.max_samples, args.seed)
    done = completed_indices(result_file)
    tasks = [task for task in tasks if task.source_index not in done]
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    config = GenerationConfig(system_prompt, args.max_steps, args.subset)
    teacher = OpenAIChatClient(
        args.base_url,
        args.api_key,
        args.model,
        args.request_timeout,
        args.max_retries,
        args.temperature,
        args.max_tokens,
        args.workers,
    )
    browser = BrowserToolClient(
        args.tool_server_url,
        args.browser_url,
        args.tool_timeout,
        args.workers,
    )

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    successful = 0
    print(f"Pending tasks: {len(tasks)}; workers: {args.workers}")
    with output_file.open("a", encoding="utf-8") as data_stream, result_file.open(
        "a", encoding="utf-8"
    ) as result_stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            generated = executor.map(
                lambda task: run_task(task, config, teacher, browser), tasks
            )
            for index, (records, result, usage) in enumerate(generated, 1):
                if result["success"] or args.keep_failed:
                    for record in records:
                        data_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                result_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                data_stream.flush()
                result_stream.flush()
                successful += int(result["success"])
                for key in totals:
                    totals[key] += usage[key]
                print(
                    f"[{index}/{len(tasks)}] status={result['status']} "
                    f"success={result['success']} steps={result['steps']}"
                )

    print(f"Successful trajectories: {successful}/{len(tasks)}")
    print(f"SFT teacher JSONL: {output_file}")
    print(f"Generation results: {result_file}")
    print(f"Token usage: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
