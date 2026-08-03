import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERL_TOOL_ROOT = PROJECT_ROOT / "verl-tool"
for _path in (PROJECT_ROOT, VERL_TOOL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vtc_renderer import VTCTool  # noqa: E402


DEFAULT_URL = "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing/"
IMAGE_OBS_TEXT = "<Image provided. Please use the visual browser observation to decide the next action.>"
LOCAL_NO_PROXY_HOSTS = [
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
]


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def safe_load_nested(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def normalize_answers(value: Any) -> List[str]:
    value = safe_load_nested(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def extract_prompt_text(prompt: Any, role: str) -> str:
    prompt = safe_load_nested(prompt)
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return ""
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def extract_question_from_prompt(prompt: Any) -> str:
    user_text = extract_prompt_text(prompt, "user")
    match = re.search(r"Objective:\s*(.*?)(?:\nURL:|\nObservation:|\Z)", user_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def fallback_metric_heuristic(refs: Sequence[str], pred: str) -> float:
    def clean(text: str) -> str:
        text = str(text).strip().lower()
        if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
            text = text[1:-1]
        return text

    def token_f1(ref: str, hyp: str) -> float:
        ref_tokens = set(re.findall(r"\w+", ref))
        hyp_tokens = set(re.findall(r"\w+", hyp))
        if not ref_tokens or not hyp_tokens:
            return 0.0
        overlap = ref_tokens & hyp_tokens
        if not overlap:
            return 0.0
        precision = len(overlap) / len(hyp_tokens)
        recall = len(overlap) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def edit_distance_ratio(ref: str, hyp: str) -> float:
        dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            dp[i][0] = i
        for j in range(len(hyp) + 1):
            dp[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[len(ref)][len(hyp)] / (max(len(ref), len(hyp)) or 1)

    def fuzzy_match(ref: str, hyp: str) -> float:
        ref = clean(ref)
        hyp = clean(hyp)
        matcher = SequenceMatcher(None, ref, hyp)
        lcs_len = sum(block.size for block in matcher.get_matching_blocks())
        char_lcs = lcs_len / (max(len(ref), len(hyp)) or 1)
        score = 0.7 * char_lcs + 0.3 * token_f1(ref, hyp) - 0.1 * edit_distance_ratio(ref, hyp)
        return max(0.0, min(float(score), 1.0))

    refs = [str(ref) for ref in refs if ref is not None]
    return max((fuzzy_match(ref, pred) for ref in refs), default=0.0)


def fallback_format_score(s: str) -> float:
    def is_valid_action_syntax(action_text: str) -> bool:
        action_text = action_text.strip()
        patterns = [
            r"^click\s+\[\d+\]\s+\[.*\]$",
            r"^type\s+\[\d+\]\s+\[.*\]\s+\[(0|1|press_enter_after=0|press_enter_after=1)\]$",
            r"^hover\s+\[\d+\]\s+\[.*\]$",
            r"^press\s+\[.+\]$",
            r"^scroll\s+\[(down|up)\]$",
            r"^new_tab$",
            r"^tab_focus\s+\[\d+\]$",
            r"^close_tab$",
            r"^goto\s+\[.+\]$",
            r"^go_back$",
            r"^go_forward$",
            r"^stop\s+\[.*\]$",
        ]
        return any(re.fullmatch(pattern, action_text, flags=re.DOTALL) for pattern in patterns)

    score = 0.0
    if "<think>" not in s or "</think>" not in s:
        return 0.0
    score += 0.3
    if s[: s.index("<think>")].strip() == "":
        score += 0.1
    tail_parts = s.split("</think>", maxsplit=1)
    if len(tail_parts) < 2:
        return round(score, 3)
    tail_content = tail_parts[1].strip()
    match = re.search(r"```((.|\n)*?)```", tail_content)
    if not match:
        return round(score, 3)
    score += 0.2
    action_text = match.group(1).strip()
    if tail_content == f"```{action_text}```":
        score += 0.2
    if is_valid_action_syntax(action_text):
        score += 0.2
    return round(score, 3)


def configure_proxy(proxy_url: Optional[str]) -> None:
    if not proxy_url:
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = proxy_url

    existing = []
    for key in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(key, "")
        existing.extend([item.strip() for item in value.split(",") if item.strip()])
    merged = []
    for item in existing + LOCAL_NO_PROXY_HOSTS:
        if item not in merged:
            merged.append(item)
    no_proxy = ",".join(merged)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy


def stable_sample_uid(source_name: str, row_index: int, row: pd.Series) -> str:
    extra = safe_load_nested(row.get("extra_info", {}))
    if isinstance(extra, dict):
        for key in ("id", "index", "seed"):
            if extra.get(key) not in (None, ""):
                return f"{source_name}:{extra[key]}"
    payload = json.dumps(row.to_dict(), ensure_ascii=False, cls=NumpyJSONEncoder, sort_keys=True)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"{source_name}:row{row_index}:{digest}"


@dataclass
class Sample:
    uid: str
    source_name: str
    source_path: str
    source_row_index: int
    question: str
    ground_truths: List[str]
    url: str
    row: Dict[str, Any]


def load_samples(data_paths: Sequence[str], max_samples_per_dataset: Optional[int], seed: int) -> List[Sample]:
    samples: List[Sample] = []

    for data_path in data_paths:
        path = Path(data_path)
        df = pd.read_parquet(path)
        if max_samples_per_dataset is not None and len(df) > max_samples_per_dataset:
            df = df.sample(n=max_samples_per_dataset, random_state=seed).sort_index()

        source_name = path.parent.name
        if source_name in ("dataset", "data") or source_name.startswith("train_"):
            source_name = path.stem

        for row_index, row in df.iterrows():
            extra = safe_load_nested(row.get("extra_info", {}))
            reward_model = safe_load_nested(row.get("reward_model", {}))
            if not isinstance(extra, dict):
                extra = {}
            if not isinstance(reward_model, dict):
                reward_model = {}

            question = str(extra.get("question") or extract_question_from_prompt(row.get("prompt"))).strip()
            ground_truths = normalize_answers(reward_model.get("ground_truth"))
            if not ground_truths:
                ground_truths = normalize_answers(extra.get("golden_answers"))
            if not ground_truths and extra.get("selected_answer") not in (None, ""):
                ground_truths = [str(extra["selected_answer"])]
            if not ground_truths and extra.get("gt") not in (None, ""):
                ground_truths = [str(extra["gt"])]

            url = str(extra.get("url") or DEFAULT_URL)
            uid = stable_sample_uid(source_name, int(row_index), row)
            samples.append(
                Sample(
                    uid=uid,
                    source_name=source_name,
                    source_path=str(path),
                    source_row_index=int(row_index),
                    question=question,
                    ground_truths=ground_truths,
                    url=url,
                    row=row.to_dict(),
                )
            )

    return samples


class BrowserEnvClient:
    def __init__(self, env_url: str, default_url: str = DEFAULT_URL, timeout: int = 1200) -> None:
        self.env_url = env_url
        self.default_url = default_url
        self.timeout = timeout
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def step(self, trajectory_id: str, action: str, finish: bool, url: Optional[str]) -> str:
        payload = {
            "trajectory_ids": [trajectory_id],
            "actions": [action],
            "finish": [finish],
            "extra_fields": [{"url": url or self.default_url}],
        }
        resp = self.session.post(self.env_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        raw_obs = data.get("observations", [""])[0]
        return self.clean_observation(raw_obs)

    @staticmethod
    def clean_observation(raw_obs: Any) -> str:
        if isinstance(raw_obs, dict):
            return json.dumps(raw_obs, ensure_ascii=False)
        text = str(raw_obs)
        if "Observation:\n" in text:
            text = text.split("Observation:\n", 1)[1]
            if "\nParsed Previous Action:" in text:
                text = text.split("\nParsed Previous Action:", 1)[0]
        return text


class VLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.base_url = base_url if base_url.endswith("/") else f"{base_url}/"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def generate(self, system_prompt: str, user_prompt: str, image_base64: str) -> Tuple[str, float]:
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        start = time.perf_counter()
        resp = self.session.post(
            f"{self.base_url}chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        latency = time.perf_counter() - start
        if resp.status_code != 200:
            raise RuntimeError(f"LLM server returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"].get("content") or "", latency


class RewardComputer:
    def __init__(self, fuzzy_weight: float = 0.9, structure_weight: float = 0.1, backend: str = "local") -> None:
        self.fuzzy_weight = fuzzy_weight
        self.structure_weight = structure_weight
        self.backend = backend
        self.metric_heuristic = fallback_metric_heuristic
        self.browser_format_score = fallback_format_score

        if backend == "mini_webarena":
            try:
                from mini_webarena.evaluator import metric_heuristic
                from mini_webarena.rl_utils import format_score as browser_format_score
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise ImportError(
                    "Could not import mini_webarena reward helpers used by BrowserAgent.py. "
                    "Use --reward_backend local to avoid NLTK downloads, or pass --proxy_url for the import-time download. "
                    f"Original error: {repr(exc)}"
                ) from exc
            self.metric_heuristic = metric_heuristic
            self.browser_format_score = browser_format_score
        elif backend != "local":
            raise ValueError(f"Unsupported reward backend: {backend}")

    @staticmethod
    def extract_last_stop_content(text: str) -> str:
        matches = re.findall(r"```stop\s*\[([^\]]*)\]```", text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"<action>\s*stop\s*\[([^\]]*)\]\s*</action>", text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"stop\s*\[([^\]]*)\]", text, flags=re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else ""

    def answer_score(self, trajectory_text: str, ground_truths: Sequence[str]) -> float:
        if not ground_truths:
            return 0.0
        pred = self.extract_last_stop_content(trajectory_text)
        try:
            return float(self.metric_heuristic(ground_truths, pred))
        except LookupError as exc:
            print(f"[ANSWER_SCORE_FALLBACK] missing tokenizer data, using local fuzzy scorer: {repr(exc)}")
            return float(fallback_metric_heuristic(ground_truths, pred))

    def format_score(self, actions: Sequence[str], uid: Optional[str] = None) -> float:
        scores = []
        for idx, action in enumerate(actions):
            try:
                scores.append(float(self.browser_format_score(action)))
            except Exception as exc:
                print(f"[FORMAT_SCORE_ERROR] uid={uid} action_idx={idx} err={repr(exc)}")
                scores.append(0.0)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def final_reward(self, actions: Sequence[str], trajectory_text: str, ground_truths: Sequence[str], uid: str) -> Dict[str, float]:
        answer_reward = self.answer_score(trajectory_text, ground_truths)
        format_reward = self.format_score(actions, uid=uid)
        final = self.fuzzy_weight * answer_reward + self.structure_weight * format_reward
        return {
            "reward": float(final),
            "answer_score": float(answer_reward),
            "format_score": float(format_reward),
        }


class RolloutRunner:
    def __init__(
        self,
        env: BrowserEnvClient,
        llm: VLMClient,
        reward: RewardComputer,
        system_prompt: str,
        image_output_dir: Path,
        max_steps: int,
        compression_factor: float,
        render_width: int,
        render_aspect_ratio: str,
        use_simple_render: bool,
        save_images: bool,
    ) -> None:
        self.env = env
        self.llm = llm
        self.reward = reward
        self.system_prompt = system_prompt
        self.image_output_dir = image_output_dir
        self.max_steps = max_steps
        self.compression_factor = compression_factor
        self.render_width = render_width
        self.render_aspect_ratio = render_aspect_ratio
        self.use_simple_render = use_simple_render
        self.save_images = save_images
        self.vtc_tool = VTCTool()
        self.render_lock = threading.Lock()
        self.user_prompt_template = (
            "Objective: {question}\n"
            "Observation: {observation}\n"
            "HISTORY_ACTION: {history_actions}\n"
            "HISTORY_info: {history_info}\n"
        )

    @staticmethod
    def extract_action(text: str) -> str:
        tag_blocks = re.findall(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL | re.IGNORECASE)
        if tag_blocks:
            return tag_blocks[-1].strip()
        fenced_blocks = re.findall(r"```\s*([^\s].*?[^\s])\s*```", text, re.DOTALL)
        if fenced_blocks:
            return fenced_blocks[-1].strip().replace("```", "").strip()
        return ""

    @staticmethod
    def extract_conclusion(text: str) -> str:
        blocks = re.findall(r"<conclusion>\s*(.*?)\s*</conclusion>", text, re.DOTALL | re.IGNORECASE)
        return blocks[-1].strip() if blocks else ""

    def render_observation(self, observation: str, trajectory_id: str, step_idx: int) -> Tuple[str, Optional[str], float]:
        start = time.perf_counter()
        with self.render_lock:
            if self.use_simple_render:
                img, _ = self.vtc_tool.render_text_to_image_simple(
                    observation,
                    width=self.render_width,
                    aspect_ratio=self.render_aspect_ratio,
                )
            else:
                img, _ = self.vtc_tool.render_text_to_image(
                    observation,
                    use_compact_mode=True,
                    max_width=2048,
                    max_height=2048,
                )
            if self.compression_factor > 1.0:
                img = self.vtc_tool.compress_image_arrays([img], compression_factor=self.compression_factor)[0]

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image_path = None
        if self.save_images:
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            image_path = self.image_output_dir / f"obs_{trajectory_id}_step_{step_idx}.png"
            img.save(image_path)
        return image_base64, str(image_path) if image_path else None, time.perf_counter() - start

    def run(self, sample: Sample, trial_idx: int) -> Dict[str, Any]:
        trajectory_id = str(uuid.uuid4())
        history_actions = "\n"
        history_info = "\n"
        steps: List[Dict[str, Any]] = []
        model_responses: List[str] = []
        parsed_actions: List[str] = []
        final_answer = ""
        status = "ok"
        error = ""

        try:
            current_obs = self.env.step(trajectory_id, "", False, sample.url)
            for step_idx in range(self.max_steps):
                image_base64, image_path, render_latency = self.render_observation(current_obs, trajectory_id, step_idx)
                user_prompt = self.user_prompt_template.format(
                    question=sample.question,
                    observation=IMAGE_OBS_TEXT,
                    history_actions=history_actions,
                    history_info=history_info,
                )
                response_text, llm_latency = self.llm.generate(self.system_prompt, user_prompt, image_base64)
                action = self.extract_action(response_text)
                conclusion = self.extract_conclusion(response_text)
                is_stop = "stop" in action.lower()

                step_record = {
                    "step": step_idx,
                    "observation": current_obs,
                    "image_path": image_path,
                    "prompt": user_prompt,
                    "model_response": response_text,
                    "action": action,
                    "conclusion": conclusion,
                    "render_latency": round(render_latency, 4),
                    "llm_latency": round(llm_latency, 4),
                }

                env_start = time.perf_counter()
                next_obs = self.env.step(trajectory_id, response_text, is_stop, sample.url)
                step_record["env_latency"] = round(time.perf_counter() - env_start, 4)
                steps.append(step_record)
                model_responses.append(response_text)
                parsed_actions.append(action)

                if action:
                    history_actions += action + "\n"
                if conclusion:
                    history_info += conclusion + "\n"

                current_obs = next_obs
                if is_stop:
                    final_answer = RewardComputer.extract_last_stop_content(response_text)
                    break
            else:
                status = "max_steps"
                try:
                    self.env.step(trajectory_id, "", True, sample.url)
                except Exception:
                    pass
        except Exception as exc:
            status = "error"
            error = repr(exc)

        trajectory_text = "\n\n".join(model_responses)
        if status == "error":
            scores = {"reward": 0.0, "answer_score": 0.0, "format_score": 0.0}
        else:
            scores = self.reward.final_reward(model_responses, trajectory_text, sample.ground_truths, sample.uid)

        return {
            "uid": sample.uid,
            "source_name": sample.source_name,
            "source_path": sample.source_path,
            "source_row_index": sample.source_row_index,
            "trial_idx": trial_idx,
            "trajectory_id": trajectory_id,
            "question": sample.question,
            "ground_truths": sample.ground_truths,
            "url": sample.url,
            "status": status,
            "error": error,
            "reward": scores["reward"],
            "answer_score": scores["answer_score"],
            "format_score": scores["format_score"],
            "success": bool(scores["answer_score"] > 0.0),
            "final_answer": final_answer,
            "num_steps": len(steps),
            "actions": parsed_actions,
            "steps": steps,
        }


class JSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, item: Dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=False, cls=NumpyJSONEncoder)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def completed_trials(rollout_file: Path) -> set[Tuple[str, int]]:
    done = set()
    for item in iter_jsonl(rollout_file) or []:
        if item.get("status") in ("ok", "max_steps", "error"):
            done.add((str(item["uid"]), int(item["trial_idx"])))
    return done


DIFFICULTY_BUCKETS = [
    "trivial",
    "easy_high",
    "medium_high",
    "medium_mid",
    "medium_low",
    "hard",
    "unsolved",
]


def classify_difficulty(success_count: int, mean_reward: float) -> str:
    if success_count >= 8:
        return "trivial"
    if 6 <= success_count <= 7:
        return "easy_high"
    if success_count == 5:
        return "medium_high"
    if 3 <= success_count <= 4:
        return "medium_mid"
    if success_count == 2:
        return "medium_low"
    if success_count == 1 or mean_reward > 0.05:
        return "hard"
    return "unsolved"


def aggregate_rollouts(samples: Sequence[Sample], rollout_file: Path, k: int) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    by_uid: Dict[str, List[Dict[str, Any]]] = {}
    for item in iter_jsonl(rollout_file) or []:
        by_uid.setdefault(str(item.get("uid")), []).append(item)

    rows = []
    by_uid_summary: Dict[str, Dict[str, Any]] = {}
    sample_map = {sample.uid: sample for sample in samples}

    for uid, sample in sample_map.items():
        trials = sorted(by_uid.get(uid, []), key=lambda x: int(x.get("trial_idx", 0)))
        rewards = [float(x.get("reward", 0.0)) for x in trials]
        answer_scores = [float(x.get("answer_score", 0.0)) for x in trials]
        format_scores = [float(x.get("format_score", 0.0)) for x in trials]
        successes = [bool(x.get("success", False)) for x in trials]
        solve_rate = float(sum(successes) / len(successes)) if successes else 0.0
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        success_count = int(sum(successes))
        summary = {
            "uid": uid,
            "source_name": sample.source_name,
            "source_path": sample.source_path,
            "source_row_index": sample.source_row_index,
            "question": sample.question,
            "ground_truths": sample.ground_truths,
            "num_rollouts": len(trials),
            "target_rollouts": k,
            "mean_reward": mean_reward,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "mean_answer_score": float(np.mean(answer_scores)) if answer_scores else 0.0,
            "mean_format_score": float(np.mean(format_scores)) if format_scores else 0.0,
            "solve_rate": solve_rate,
            "success_count": success_count,
            "difficulty": classify_difficulty(success_count, mean_reward),
            "trial_rewards": rewards,
            "trial_successes": successes,
        }
        rows.append(summary)
        by_uid_summary[uid] = summary

    return pd.DataFrame(rows), by_uid_summary


def augment_row_for_curriculum(sample: Sample, summary: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(sample.row)
    extra = safe_load_nested(row.get("extra_info", {}))
    if not isinstance(extra, dict):
        extra = {}
    extra["filter_uid"] = summary["uid"]
    extra["difficulty"] = summary["difficulty"]
    extra["rollout_mean_reward"] = summary["mean_reward"]
    extra["rollout_solve_rate"] = summary["solve_rate"]
    extra["rollout_success_count"] = summary["success_count"]
    extra["rollout_num_trials"] = summary["num_rollouts"]
    row["extra_info"] = extra
    return row


def export_curriculum(samples: Sequence[Sample], summary_by_uid: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    curriculum_dir = output_dir / "curriculum"
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    rows_by_difficulty: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in DIFFICULTY_BUCKETS}
    all_rows: List[Dict[str, Any]] = []

    for sample in samples:
        summary = summary_by_uid.get(sample.uid)
        if not summary:
            continue
        row = augment_row_for_curriculum(sample, summary)
        rows_by_difficulty[summary["difficulty"]].append(row)
        all_rows.append(row)

    if all_rows:
        pd.DataFrame(all_rows).to_parquet(curriculum_dir / "all_scored.parquet", index=False)

    manifest = {}
    for difficulty, rows in rows_by_difficulty.items():
        difficulty_dir = curriculum_dir / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_parquet(difficulty_dir / "data.parquet", index=False)
        manifest[difficulty] = len(rows)

    stage_defs = {
        "stage_1_easy_high.parquet": ["easy_high"],
        "stage_2_easy_medium_high.parquet": ["easy_high", "medium_high"],
        "stage_3_core_medium.parquet": ["medium_high", "medium_mid"],
        "stage_4_medium_hard.parquet": ["medium_mid", "medium_low", "hard"],
        "stage_5_all_trainable.parquet": ["easy_high", "medium_high", "medium_mid", "medium_low", "hard"],
    }
    for filename, difficulties in stage_defs.items():
        stage_rows: List[Dict[str, Any]] = []
        for difficulty in difficulties:
            stage_rows.extend(rows_by_difficulty[difficulty])
        if stage_rows:
            pd.DataFrame(stage_rows).to_parquet(curriculum_dir / filename, index=False)
        manifest[filename] = len(stage_rows)

    with (curriculum_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def read_system_prompt(path: Optional[str], samples: Sequence[Sample]) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    for sample in samples:
        content = extract_prompt_text(sample.row.get("prompt"), "system")
        if content:
            return content
    default_path = PROJECT_ROOT / "prompt" / "system_prompt_with_history_info.txt"
    with default_path.open("r", encoding="utf-8") as f:
        return f.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out SFT BrowserAgent on parquet datasets and bucket samples by reward difficulty."
    )
    parser.add_argument(
        "--data-paths", "--data_paths", dest="data_paths",
        nargs="+",
        default=[
            str(PROJECT_ROOT / "RL/dataset/nq/train_5000_labelled.parquet"),
            str(PROJECT_ROOT / "RL/dataset/hotpot/train_5000_labelled.parquet"),
        ],
        help="One or more parquet datasets to score. Defaults to the local nq/hotpot 5000 labelled sets.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=str(PROJECT_ROOT / "RL/filter_results"))
    parser.add_argument("--system-prompt", "--system_prompt", dest="system_prompt", type=str, default=None)
    parser.add_argument("--env-url", "--env_url", dest="env_url", default=os.environ.get("TOOL_SERVER_URL", "http://127.0.0.1:5000/get_observation"))
    parser.add_argument("--base-url", "--base_url", dest="base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8008/v1"))
    parser.add_argument("--api-key", "--api_key", dest="api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "browseragent-sft"))
    parser.add_argument("--k", "--num_rollouts", dest="k", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--env_timeout", type=int, default=1200)
    parser.add_argument("--compression_factor", type=float, default=2.0)
    parser.add_argument("--render_width", type=int, default=1024)
    parser.add_argument("--render_aspect_ratio", type=str, default="4:3")
    parser.add_argument(
        "--reward_backend",
        choices=["local", "mini_webarena"],
        default="local",
        help="local avoids mini_webarena.evaluator import-time NLTK downloads; mini_webarena uses the original helpers.",
    )
    parser.add_argument(
        "--proxy_url",
        type=str,
        default=None,
        help="Optional proxy for external downloads/API calls, e.g. http://127.0.0.1:17897. Localhost is added to NO_PROXY.",
    )
    parser.add_argument("--compact_render", action="store_true", help="Use VTC compact text render instead of simple render.")
    parser.add_argument("--save_images", action="store_true", help="Save compressed observation images used by rollouts.")
    parser.add_argument("--no_resume", action="store_true", help="Do not skip completed uid/trial pairs in rollouts.jsonl.")
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Reuse output_dir/rollouts.jsonl and only rebuild sample_scores plus curriculum buckets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_proxy(args.proxy_url)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_file = output_dir / "rollouts.jsonl"
    summary_jsonl = output_dir / "sample_scores.jsonl"
    summary_parquet = output_dir / "sample_scores.parquet"

    samples = load_samples(args.data_paths, args.max_samples_per_dataset, args.seed)
    if not samples:
        raise ValueError("No samples loaded from --data_paths.")

    if args.aggregate_only:
        if not rollout_file.exists():
            raise FileNotFoundError(f"--aggregate_only requires an existing rollout log: {rollout_file}")
        print(f"Loaded {len(samples)} samples from {len(args.data_paths)} parquet files.")
        print(f"Aggregate-only mode: reusing rollout log without running new rollouts: {rollout_file}")
    else:
        system_prompt = read_system_prompt(args.system_prompt, samples)
        env = BrowserEnvClient(args.env_url, timeout=args.env_timeout)
        llm = VLMClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.llm_timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        reward = RewardComputer(backend=args.reward_backend)
        runner = RolloutRunner(
            env=env,
            llm=llm,
            reward=reward,
            system_prompt=system_prompt,
            image_output_dir=output_dir / "obs_images",
            max_steps=args.max_steps,
            compression_factor=args.compression_factor,
            render_width=args.render_width,
            render_aspect_ratio=args.render_aspect_ratio,
            use_simple_render=not args.compact_render,
            save_images=args.save_images,
        )

        done = set() if args.no_resume else completed_trials(rollout_file)
        jobs = [(sample, trial_idx) for sample in samples for trial_idx in range(1, args.k + 1) if (sample.uid, trial_idx) not in done]
        writer = JSONLWriter(rollout_file)

        print(
            f"Loaded {len(samples)} samples from {len(args.data_paths)} parquet files. "
            f"Target rollouts={len(samples) * args.k}, pending={len(jobs)}, workers={args.num_workers}."
        )
        print(f"Rollout log: {rollout_file}")

        completed = 0
        started_at = time.time()
        if jobs:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_job = {
                    executor.submit(runner.run, sample, trial_idx): (sample.uid, trial_idx)
                    for sample, trial_idx in jobs
                }
                for future in as_completed(future_to_job):
                    uid, trial_idx = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "uid": uid,
                            "trial_idx": trial_idx,
                            "status": "error",
                            "error": repr(exc),
                            "reward": 0.0,
                            "answer_score": 0.0,
                            "format_score": 0.0,
                            "success": False,
                        }
                    writer.append(result)
                    completed += 1
                    if completed % max(1, min(50, args.num_workers)) == 0 or completed == len(jobs):
                        elapsed = time.time() - started_at
                        print(f"Finished {completed}/{len(jobs)} pending rollouts, elapsed={elapsed/60:.1f} min")

    summary_df, summary_by_uid = aggregate_rollouts(samples, rollout_file, args.k)
    summary_df.to_json(summary_jsonl, orient="records", lines=True, force_ascii=False)
    summary_df.to_parquet(summary_parquet, index=False)
    export_curriculum(samples, summary_by_uid, output_dir)

    counts = summary_df["difficulty"].value_counts().to_dict() if not summary_df.empty else {}
    print(f"Saved sample scores: {summary_parquet}")
    print(f"Saved curriculum parquet files under: {output_dir / 'curriculum'}")
    print(f"Difficulty counts: {counts}")


if __name__ == "__main__":
    main()
