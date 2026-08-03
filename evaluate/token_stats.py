#!/usr/bin/env python3
"""Count per-step prompt tokens for evaluation results.

The two primary variants match evaluate/pipeline.py:
1. no_image_compression: the raw text observation is placed in the user prompt.
2. image_compression: the observation is rendered by VTCTool and attached as an
   image, while the user prompt contains the same placeholder used by pipeline.py.

All token counts are produced by the local model tokenizer/processor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vtc_renderer import VTCTool  # noqa: E402


PIPELINE_USER_PROMPT_TEMPLATE = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""
PIPELINE_IMAGE_OBS_TEXT = "<Image provided attached. Please refer to the visual observation.>"
HISTORY_CURRENT_IMAGE_TEXT = "<Current observation image is attached.>"
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "prompt" / "system_prompt_with_history_info.txt"
if not DEFAULT_SYSTEM_PROMPT.exists():
    DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "prompt" / "system_prompt_with_history_info.txt"


@dataclass
class MaxRecord:
    tokens: int = 0
    file: str = ""
    line: int = 0
    trajectory_id: str = ""
    sample_idx: Any = None
    trial_idx: Any = None
    step: Any = None

    def update(self, tokens: int, context: dict[str, Any]) -> None:
        if tokens <= self.tokens:
            return
        self.tokens = tokens
        self.file = context["file"]
        self.line = context["line"]
        self.trajectory_id = context.get("trajectory_id", "")
        self.sample_idx = context.get("sample_idx")
        self.trial_idx = context.get("trial_idx")
        self.step = context.get("step")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "file": self.file,
            "line": self.line,
            "trajectory_id": self.trajectory_id,
            "sample_idx": self.sample_idx,
            "trial_idx": self.trial_idx,
            "step": self.step,
        }


@dataclass
class RunningStats:
    count: int = 0
    total: int = 0
    min_tokens: int | None = None
    max_record: MaxRecord = field(default_factory=MaxRecord)

    def add(self, tokens: int, context: dict[str, Any]) -> None:
        self.count += 1
        self.total += tokens
        self.min_tokens = tokens if self.min_tokens is None else min(self.min_tokens, tokens)
        self.max_record.update(tokens, context)

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def max_tokens(self) -> int:
        return self.max_record.tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.count,
            "total_tokens": self.total,
            "avg_step_tokens": self.average,
            "max_step_tokens": self.max_tokens,
            "min_step_tokens": self.min_tokens or 0,
            "max_step": self.max_record.as_dict(),
        }


@dataclass
class DeltaStats:
    count: int = 0
    total_saved: int = 0
    max_saved: int | None = None
    min_saved: int | None = None

    def add(self, no_compression_tokens: int, compression_tokens: int) -> None:
        saved = no_compression_tokens - compression_tokens
        self.count += 1
        self.total_saved += saved
        self.max_saved = saved if self.max_saved is None else max(self.max_saved, saved)
        self.min_saved = saved if self.min_saved is None else min(self.min_saved, saved)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.count,
            "avg_tokens_saved_per_step": self.total_saved / self.count if self.count else 0.0,
            "max_tokens_saved_single_step": self.max_saved or 0,
            "min_tokens_saved_single_step": self.min_saved or 0,
        }


@dataclass
class Aggregate:
    files: int = 0
    trajectories: int = 0
    steps: int = 0
    no_image_compression: RunningStats = field(default_factory=RunningStats)
    image_compression: RunningStats = field(default_factory=RunningStats)
    delta: DeltaStats = field(default_factory=DeltaStats)
    invalid_lines: int = 0
    skipped_steps: int = 0

    def add_step(
        self,
        no_compression_tokens: int,
        compression_tokens: int,
        context: dict[str, Any],
    ) -> None:
        self.steps += 1
        self.no_image_compression.add(no_compression_tokens, context)
        self.image_compression.add(compression_tokens, context)
        self.delta.add(no_compression_tokens, compression_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "trajectories": self.trajectories,
            "steps": self.steps,
            "invalid_lines": self.invalid_lines,
            "skipped_steps": self.skipped_steps,
            "no_image_compression": self.no_image_compression.as_dict(),
            "image_compression": self.image_compression.as_dict(),
            "delta": self.delta.as_dict(),
        }


class ModelTokenCounter:
    def __init__(self, model_path: Path, processor_use_fast: bool = False):
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for model-tokenizer based counting"
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=processor_use_fast,
            local_files_only=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
            )
        self.text_cache: dict[str, int] = {}
        self.mm_cache: dict[str, int] = {}

    def count_text_chat(self, system_prompt: str, user_prompt: str) -> int:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cache_key = stable_hash({"messages": messages, "mode": "text"})
        cached = self.text_cache.get(cache_key)
        if cached is not None:
            return cached

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        count = len(input_ids)
        self.text_cache[cache_key] = count
        return count

    def count_multimodal_chat(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        images: list[Image.Image],
    ) -> int:
        image_fingerprints = [
            (image.width, image.height, image.mode)
            for image in images
        ]
        cache_key = stable_hash(
            {
                "system": system_prompt,
                "user_content": scrub_content_for_hash(user_content),
                "image_fingerprints": image_fingerprints,
                "mode": "multimodal",
            }
        )
        cached = self.mm_cache.get(cache_key)
        if cached is not None:
            return cached

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.processor(
            text=[prompt_text],
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
        )
        count = int(model_inputs["input_ids"].shape[-1])
        self.mm_cache[cache_key] = count
        return count


class ObservationImageProvider:
    def __init__(
        self,
        prefer_existing_images: bool = True,
        max_width: int = 2048,
        max_height: int = 2048,
        compression_factor: float = 1.0,
    ):
        self.prefer_existing_images = prefer_existing_images
        self.max_width = max_width
        self.max_height = max_height
        self.compression_factor = compression_factor
        self.vtc = VTCTool()
        self.render_cache: dict[str, Image.Image] = {}

    def image_for_text(self, text: str) -> Image.Image:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self.render_cache.get(key)
        if cached is not None:
            return cached.copy()

        image, _ = self.vtc.render_text_to_image(
            text,
            use_compact_mode=True,
            max_width=self.max_width,
            max_height=self.max_height,
        )
        if self.compression_factor and self.compression_factor != 1.0:
            image = self.vtc.compress_image_arrays([image], self.compression_factor)[0]
        image = image.convert("RGB")
        self.render_cache[key] = image.copy()
        return image

    def image_for_path_or_text(self, path_value: Any, fallback_text: str, result_file: Path) -> Image.Image:
        path = resolve_result_path(path_value, result_file)
        if self.prefer_existing_images and path is not None and path.exists():
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        return self.image_for_text(fallback_text)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def scrub_content_for_hash(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scrubbed: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image" or "image" in part or "image_url" in part:
            scrubbed.append({"type": "image"})
        else:
            scrubbed.append(part)
    return scrubbed


def resolve_result_path(path_value: Any, result_file: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = result_file.parent / path
        if candidate.exists():
            return candidate
    return path


def iter_result_files(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.rglob(glob_pattern))
    return [
        path for path in files
        if path.is_file() and not path.name.endswith("_metrics.jsonl")
    ]


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def build_pipeline_user_prompt(
    question: Any,
    observation: Any,
    history_actions: str,
    history_info: str,
) -> str:
    return PIPELINE_USER_PROMPT_TEMPLATE.format(
        str(question or ""),
        str(observation or ""),
        history_actions,
        history_info,
    )


def build_pipeline_multimodal_content(user_prompt: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": user_prompt},
        {"type": "image"},
    ]


def build_stored_multimodal_content(
    step: dict[str, Any],
    images: list[Image.Image],
) -> list[dict[str, Any]]:
    prompt = str(step.get("prompt") or "")
    history_context = step.get("history_context") or []
    if history_context:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt.strip()}]
        content.append({"type": "text", "text": "Current observation image:"})
        content.append({"type": "image"})
        for entry in history_context:
            step_idx = entry.get("step")
            content.append({"type": "text", "text": f"Historical step {step_idx} observation image:"})
            content.append({"type": "image"})
            content.append({"type": "text", "text": f"Historical step {step_idx} model output image:"})
            content.append({"type": "image"})
        expected_images = 1 + len(history_context) * 2
        return content[: len(content) - max(0, expected_images - len(images)) * 2]

    return [
        {"type": "text", "text": prompt},
        {"type": "image"},
    ]


def has_compressed_saved_prompt(step: dict[str, Any]) -> bool:
    prompt = str(step.get("prompt") or "")
    return bool(
        step.get("image_path")
        or step.get("history_context")
        or PIPELINE_IMAGE_OBS_TEXT in prompt
        or HISTORY_CURRENT_IMAGE_TEXT in prompt
        or "image is attached" in prompt
        or "Image provided attached" in prompt
    )


def images_for_stored_prompt(
    trajectory_steps_by_index: dict[Any, dict[str, Any]],
    step: dict[str, Any],
    result_file: Path,
    image_provider: ObservationImageProvider,
) -> list[Image.Image]:
    images = [
        image_provider.image_for_path_or_text(
            step.get("image_path"),
            str(step.get("observation") or ""),
            result_file,
        )
    ]
    for entry in step.get("history_context") or []:
        hist_step = trajectory_steps_by_index.get(entry.get("step"), {})
        images.append(
            image_provider.image_for_path_or_text(
                entry.get("observation_image_path"),
                str(hist_step.get("observation") or ""),
                result_file,
            )
        )
        images.append(
            image_provider.image_for_path_or_text(
                entry.get("model_response_image_path"),
                str(hist_step.get("model_response") or ""),
                result_file,
            )
        )
    return images


def step_context(
    result_file: Path,
    line_no: int,
    trajectory: dict[str, Any],
    step: dict[str, Any],
) -> dict[str, Any]:
    return {
        "file": str(result_file),
        "line": line_no,
        "trajectory_id": trajectory.get("id", ""),
        "sample_idx": trajectory.get("sample_idx"),
        "trial_idx": trajectory.get("trial_idx"),
        "step": step.get("step"),
    }


def count_file(
    result_file: Path,
    system_prompt: str,
    counter: ModelTokenCounter,
    image_provider: ObservationImageProvider,
    compressed_prompt_source: str,
    max_trajectories: int | None,
    max_steps: int | None,
    step_writer: Any | None,
) -> Aggregate:
    aggregate = Aggregate(files=1)
    processed_trajectories = 0

    for line_no, trajectory in read_jsonl(result_file):
        if trajectory is None:
            aggregate.invalid_lines += 1
            continue
        processed_trajectories += 1
        if max_trajectories is not None and processed_trajectories > max_trajectories:
            break
        aggregate.trajectories += 1

        steps = trajectory.get("steps") or []
        steps_by_index = {step.get("step"): step for step in steps if isinstance(step, dict)}
        history_actions = "\n"
        history_info = "\n"

        for step in steps:
            if not isinstance(step, dict):
                aggregate.skipped_steps += 1
                continue
            if max_steps is not None and aggregate.steps >= max_steps:
                return aggregate

            observation = str(step.get("observation") or "")
            question = trajectory.get("question", "")
            no_compression_prompt = build_pipeline_user_prompt(
                question,
                observation,
                history_actions,
                history_info,
            )
            compressed_pipeline_prompt = build_pipeline_user_prompt(
                question,
                PIPELINE_IMAGE_OBS_TEXT,
                history_actions,
                history_info,
            )

            context = step_context(result_file, line_no, trajectory, step)
            no_compression_tokens = counter.count_text_chat(system_prompt, no_compression_prompt)

            use_stored_compressed_prompt = (
                compressed_prompt_source == "stored"
                and step.get("prompt")
                and has_compressed_saved_prompt(step)
            )
            if use_stored_compressed_prompt:
                images = images_for_stored_prompt(steps_by_index, step, result_file, image_provider)
                content = build_stored_multimodal_content(step, images)
            else:
                images = [
                    image_provider.image_for_path_or_text(
                        step.get("image_path"),
                        observation,
                        result_file,
                    )
                ]
                content = build_pipeline_multimodal_content(compressed_pipeline_prompt)

            compressed_tokens = counter.count_multimodal_chat(system_prompt, content, images)
            aggregate.add_step(no_compression_tokens, compressed_tokens, context)

            if step_writer is not None:
                step_writer.write(json.dumps({
                    **context,
                    "no_image_compression_tokens": no_compression_tokens,
                    "image_compression_tokens": compressed_tokens,
                    "tokens_saved_by_image_compression": no_compression_tokens - compressed_tokens,
                    "compressed_prompt_source": "stored" if use_stored_compressed_prompt else "pipeline",
                    "attached_images": len(images),
                }, ensure_ascii=False) + "\n")

            action = str(step.get("action") or "")
            conclusion = str(step.get("conclusion") or "")
            if action:
                history_actions += action + "\n"
            if conclusion:
                history_info += conclusion + "\n"

    return aggregate


def merge_aggregate(target: Aggregate, source: Aggregate) -> None:
    target.files += source.files
    target.trajectories += source.trajectories
    target.steps += source.steps
    target.invalid_lines += source.invalid_lines
    target.skipped_steps += source.skipped_steps
    target.no_image_compression.count += source.no_image_compression.count
    target.no_image_compression.total += source.no_image_compression.total
    if source.no_image_compression.min_tokens is not None:
        if target.no_image_compression.min_tokens is None:
            target.no_image_compression.min_tokens = source.no_image_compression.min_tokens
        else:
            target.no_image_compression.min_tokens = min(
                target.no_image_compression.min_tokens,
                source.no_image_compression.min_tokens,
            )
    target.no_image_compression.max_record.update(
        source.no_image_compression.max_tokens,
        source.no_image_compression.max_record.as_dict() | {
            "file": source.no_image_compression.max_record.file,
            "line": source.no_image_compression.max_record.line,
        },
    )

    target.image_compression.count += source.image_compression.count
    target.image_compression.total += source.image_compression.total
    if source.image_compression.min_tokens is not None:
        if target.image_compression.min_tokens is None:
            target.image_compression.min_tokens = source.image_compression.min_tokens
        else:
            target.image_compression.min_tokens = min(
                target.image_compression.min_tokens,
                source.image_compression.min_tokens,
            )
    target.image_compression.max_record.update(
        source.image_compression.max_tokens,
        source.image_compression.max_record.as_dict() | {
            "file": source.image_compression.max_record.file,
            "line": source.image_compression.max_record.line,
        },
    )

    target.delta.count += source.delta.count
    target.delta.total_saved += source.delta.total_saved
    if source.delta.max_saved is not None:
        target.delta.max_saved = (
            source.delta.max_saved if target.delta.max_saved is None
            else max(target.delta.max_saved, source.delta.max_saved)
        )
    if source.delta.min_saved is not None:
        target.delta.min_saved = (
            source.delta.min_saved if target.delta.min_saved is None
            else min(target.delta.min_saved, source.delta.min_saved)
        )


def write_csv(path: Path, per_file: dict[str, Any]) -> None:
    rows = []
    for file_name, stats in per_file.items():
        rows.append({
            "file": file_name,
            "trajectories": stats["trajectories"],
            "steps": stats["steps"],
            "no_image_avg_step_tokens": stats["no_image_compression"]["avg_step_tokens"],
            "no_image_max_step_tokens": stats["no_image_compression"]["max_step_tokens"],
            "image_avg_step_tokens": stats["image_compression"]["avg_step_tokens"],
            "image_max_step_tokens": stats["image_compression"]["max_step_tokens"],
            "avg_tokens_saved_per_step": stats["delta"]["avg_tokens_saved_per_step"],
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "file",
            "trajectories",
            "steps",
            "no_image_avg_step_tokens",
            "no_image_max_step_tokens",
            "image_avg_step_tokens",
            "image_max_step_tokens",
            "avg_tokens_saved_per_step",
        ])
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count average and max per-step model input tokens for evaluation results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "evaluate" / "results",
        help="A result jsonl file or a directory to scan recursively.",
    )
    parser.add_argument(
        "--glob",
        default="*_results.jsonl",
        help="Glob used when --input is a directory. Metrics jsonl files are excluded.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct",
        help="Local base model path. Only tokenizer/processor are loaded.",
    )
    parser.add_argument(
        "--system_prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used by evaluate/pipeline.py.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT / "evaluate" / "results" / "token_usage_summary.json",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional per-file CSV summary.",
    )
    parser.add_argument(
        "--output_steps_jsonl",
        type=Path,
        default=None,
        help="Optional per-step detail JSONL.",
    )
    parser.add_argument(
        "--compressed_prompt_source",
        choices=("stored", "pipeline"),
        default="stored",
        help=(
            "stored uses each step's saved prompt and saved/rebuilt images for compressed runs; "
            "pipeline rebuilds the one-image prompt from evaluate/pipeline.py."
        ),
    )
    parser.add_argument(
        "--rerender_images",
        action="store_true",
        help="Ignore existing image files and re-render every observation with VTCTool.",
    )
    parser.add_argument(
        "--vtc_max_width",
        type=positive_int,
        default=2048,
        help="Max width used when re-rendering observations with VTCTool.",
    )
    parser.add_argument(
        "--vtc_max_height",
        type=positive_int,
        default=2048,
        help="Max height used when re-rendering observations with VTCTool.",
    )
    parser.add_argument(
        "--vtc_compression_factor",
        type=positive_float,
        default=1.0,
        help="Optional post-render image downsampling factor for VTCTool images.",
    )
    parser.add_argument(
        "--processor_use_fast",
        action="store_true",
        help="Use the fast HF image processor. By default the saved slow processor behavior is used.",
    )
    parser.add_argument(
        "--max_trajectories",
        type=positive_int,
        default=None,
        help="Debug limit per file.",
    )
    parser.add_argument(
        "--max_steps",
        type=positive_int,
        default=None,
        help="Debug limit per file.",
    )
    parser.add_argument(
        "--progress_every",
        type=positive_int,
        default=20,
        help="Print progress every N files.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.system_prompt.exists():
        raise FileNotFoundError(args.system_prompt)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    files = iter_result_files(args.input, args.glob)
    if not files:
        raise FileNotFoundError(f"No result files found under {args.input} with glob {args.glob!r}")

    counter = ModelTokenCounter(args.model_path, processor_use_fast=args.processor_use_fast)
    image_provider = ObservationImageProvider(
        prefer_existing_images=not args.rerender_images,
        max_width=args.vtc_max_width,
        max_height=args.vtc_max_height,
        compression_factor=args.vtc_compression_factor,
    )
    overall = Aggregate()
    per_file: dict[str, Any] = {}

    step_handle = None
    if args.output_steps_jsonl is not None:
        args.output_steps_jsonl.parent.mkdir(parents=True, exist_ok=True)
        step_handle = args.output_steps_jsonl.open("w", encoding="utf-8")

    try:
        for index, result_file in enumerate(files, start=1):
            file_stats = count_file(
                result_file=result_file,
                system_prompt=system_prompt,
                counter=counter,
                image_provider=image_provider,
                compressed_prompt_source=args.compressed_prompt_source,
                max_trajectories=args.max_trajectories,
                max_steps=args.max_steps,
                step_writer=step_handle,
            )
            merge_aggregate(overall, file_stats)
            per_file[str(result_file)] = file_stats.as_dict()
            if args.progress_every and (index % args.progress_every == 0 or index == len(files)):
                print(
                    f"[{index}/{len(files)}] processed {overall.steps} steps",
                    file=sys.stderr,
                )
    finally:
        if step_handle is not None:
            step_handle.close()

    summary = {
        "model_path": str(args.model_path),
        "system_prompt": str(args.system_prompt),
        "input": str(args.input),
        "glob": args.glob,
        "compressed_prompt_source": args.compressed_prompt_source,
        "vtc_rendering": {
            "rerender_images": args.rerender_images,
            "max_width": args.vtc_max_width,
            "max_height": args.vtc_max_height,
            "compression_factor": args.vtc_compression_factor,
        },
        "notes": {
            "no_image_compression": "Raw text observation in the evaluate/pipeline.py user prompt.",
            "image_compression": (
                "VTC-rendered observation image counted through the Qwen2.5-VL processor. "
                "Missing saved images are rebuilt from stored observation/model_response text."
            ),
        },
        "overall": overall.as_dict(),
        "per_file": per_file,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, per_file)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Wrote summary JSON: {args.output_json}")
    if args.output_csv is not None:
        print(f"Wrote per-file CSV: {args.output_csv}")
    if args.output_steps_jsonl is not None:
        print(f"Wrote per-step JSONL: {args.output_steps_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
