#!/usr/bin/env python3
"""Render teacher trajectories into VTC images for Swift SFT training."""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vtc_renderer import VTCTool  # noqa: E402


_local_vtc: VTCTool | None = None


def init_worker() -> None:
    """Create one renderer per worker; VTCTool is not process-serializable."""
    global _local_vtc
    _local_vtc = VTCTool()


def parse_user_content(content: str) -> dict[str, str]:
    """Extract the fields emitted by BrowserAgent teacher trajectories."""
    result = {
        "objective": "",
        "observation": "",
        "history_action": "",
        "history_info": "",
    }
    patterns = {
        "objective": r"Objective:\s*(.*?)\nObservation:",
        "observation": r"Observation:\s*(.*?)\nHISTORY_ACTION:",
        "history_action": r"HISTORY_ACTION:\s*(.*?)\nHISTORY_info:",
        "history_info": r"HISTORY_info:\s*(.*)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
    return result


def generate_image_for_observation(
    observation: str, output_dir: str, simple: bool = False
) -> str:
    """Render an observation once and reuse it by content hash."""
    if _local_vtc is None:
        raise RuntimeError("VTC worker was not initialized")

    digest = hashlib.sha256(observation.encode("utf-8")).hexdigest()
    image_path = Path(output_dir) / f"obs_{digest}.png"
    if image_path.exists():
        return str(image_path)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if simple:
        image, _ = _local_vtc.render_text_to_image_simple(
            observation, width=1024, aspect_ratio="1:1"
        )
    else:
        image, _ = _local_vtc.render_text_to_image(
            observation,
            use_compact_mode=True,
            max_width=2048,
            max_height=2048,
        )

    temporary = image_path.with_name(f".{image_path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, image_path)
    return str(image_path)


def task_generator(
    input_file: str, system_override: str | None = None
) -> Iterator[list[dict[str, Any]]]:
    """Stream contiguous steps with the same objective as one task."""
    current_task: list[dict[str, Any]] = []
    current_objective: str | None = None

    with Path(input_file).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_file}: {error}"
                ) from error

            messages = item.get("messages") or []
            system_message = next(
                (message for message in messages if message.get("role") == "system"),
                None,
            )
            user_message = next(
                (message for message in messages if message.get("role") == "user"),
                None,
            )
            assistant_message = next(
                (message for message in messages if message.get("role") == "assistant"),
                None,
            )
            if user_message is None or assistant_message is None:
                raise ValueError(
                    f"Line {line_number} must contain user and assistant messages"
                )

            parsed_user = parse_user_content(str(user_message.get("content", "")))
            objective = parsed_user["objective"]
            if not objective or not parsed_user["observation"]:
                raise ValueError(
                    f"Line {line_number} does not contain Objective and Observation fields"
                )

            if current_objective is not None and objective != current_objective:
                yield current_task
                current_task = []
            current_objective = objective
            current_task.append(
                {
                    "system": (
                        system_override
                        if system_override is not None
                        else str((system_message or {}).get("content", ""))
                    ),
                    "parsed_user": parsed_user,
                    "assistant": str(assistant_message.get("content", "")),
                    "subset": item.get("subset", "vision_dataset"),
                    "stage": item.get("stage", "sft"),
                }
            )

    if current_task:
        yield current_task


def process_single_task(
    task_steps: list[dict[str, Any]],
    image_output_dir: str,
    level: str,
    format_type: str,
    simple: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    task_images: list[str] = []
    task_messages: list[dict[str, Any]] = []
    subset = f"{task_steps[0]['subset']}_vision"
    stage = task_steps[0]["stage"]
    if level == "task":
        task_messages.append(
            {"role": "system", "content": task_steps[0]["system"]}
        )

    step_outputs: list[dict[str, Any]] = []
    for step in task_steps:
        parsed = step["parsed_user"]
        rendered = generate_image_for_observation(
            parsed["observation"], image_output_dir, simple=simple
        )
        relative_image = str(Path("images") / Path(rendered).name)
        task_images.append(relative_image)

        user_text = f"Objective: {parsed['objective']}\n"
        if format_type == "openai":
            user_text += (
                "Observation: Please refer to the provided webpage screenshot "
                "for the current UI state.\n"
            )
        else:
            user_text += "Observation: <image>\n"
        if parsed["history_action"]:
            user_text += f"HISTORY_ACTION: {parsed['history_action']}\n"
        if parsed["history_info"]:
            user_text += f"HISTORY_info: {parsed['history_info']}\n"

        user_content: Any = user_text
        if format_type == "openai":
            user_content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": relative_image},
                },
            ]

        if level == "step":
            converted = {
                "messages": [
                    {"role": "system", "content": step["system"]},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": step["assistant"]},
                ],
                "subset": subset,
                "stage": stage,
            }
            if format_type == "opensource":
                converted["images"] = [relative_image]
            step_outputs.append(converted)
        else:
            task_messages.extend(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": step["assistant"]},
                ]
            )

    if level == "step":
        return step_outputs, len(task_steps)
    converted_task = {
        "messages": task_messages,
        "subset": subset,
        "stage": stage,
    }
    if format_type == "opensource":
        converted_task["images"] = task_images
    return [converted_task], len(task_steps)


def convert_dataset(
    input_file: Path,
    output_file: Path,
    image_output_dir: Path,
    level: str,
    format_type: str,
    workers: int | None = None,
    system_override: str | None = None,
    simple: bool = False,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tasks = task_generator(str(input_file), system_override)
    process = functools.partial(
        process_single_task,
        image_output_dir=str(image_output_dir),
        level=level,
        format_type=format_type,
        simple=simple,
    )
    converted_tasks = 0
    converted_steps = 0
    with output_file.open("w", encoding="utf-8") as output:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker
        ) as executor:
            print(f"Starting VTC conversion with {executor._max_workers} workers")
            for items, step_count in tqdm(
                executor.map(process, tasks), desc="Processing tasks"
            ):
                for item in items:
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                converted_tasks += 1
                converted_steps += step_count

    print(
        f"Converted {converted_tasks} tasks / {converted_steps} steps to {output_file}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert text BrowserAgent trajectories to a VTC vision dataset"
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        default="browseragent-sft",
        help="output name below sft/dataset (default: browseragent-sft)",
    )
    parser.add_argument("--level", choices=("step", "task"), default="task")
    parser.add_argument(
        "--format", choices=("openai", "opensource"), default="opensource"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--system-msg-path", type=Path)
    parser.add_argument("--simple", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace data.jsonl while reusing already rendered images",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.dataset_name):
        parser.error("--dataset-name must be a single safe directory name")

    input_file = args.input.expanduser().resolve()
    if not input_file.is_file():
        parser.error(f"input JSONL does not exist: {input_file}")
    output_dir = DATASET_ROOT / args.dataset_name
    output_file = output_dir / "data.jsonl"
    if output_file.exists() and not args.overwrite:
        parser.error(f"output exists: {output_file}; pass --overwrite to replace it")

    system_override = None
    if args.system_msg_path:
        try:
            system_override = args.system_msg_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            parser.error(str(error))

    convert_dataset(
        input_file=input_file,
        output_file=output_file,
        image_output_dir=output_dir / "images",
        level=args.level,
        format_type=args.format,
        workers=args.workers,
        system_override=system_override,
        simple=args.simple,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
