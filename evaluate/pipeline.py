import json
import pandas as pd
import requests
import argparse
import uuid
import re
import os
import base64
import threading
import time
import io
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from vtc_renderer import VTCTool


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_answer(s: str) -> str:
    """Normalize answers using the same rules as AgentOCR search EM."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def ensure_answer_list(golden_answers: Any) -> List[str]:
    """Convert parquet/JSON answer containers to a JSON-serializable list."""
    if golden_answers is None:
        return []
    if isinstance(golden_answers, str):
        return [golden_answers]
    if hasattr(golden_answers, "tolist"):
        golden_answers = golden_answers.tolist()
    if not isinstance(golden_answers, (list, tuple)):
        golden_answers = [golden_answers]
    return [str(answer) for answer in golden_answers if answer is not None]


def em_check(prediction: str, golden_answers: Any) -> int:
    normalized_prediction = normalize_answer(prediction)
    return int(any(
        normalize_answer(golden_answer) == normalized_prediction
        for golden_answer in ensure_answer_list(golden_answers)
    ))


def substring_check(prediction: str, golden_answers: Any) -> int:
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in ensure_answer_list(golden_answers):
        normalized_golden_answer = normalize_answer(golden_answer)
        if normalized_golden_answer and normalized_golden_answer in normalized_prediction:
            return 1
    return 0


def generate_image_for_observation(
    vtc_tool: Any,
    ob_text: str,
    output_dir: str,
    step_id: str,
    compression_factor: float,
    max_width: int,
    max_height: int,
) -> Tuple[str, str, float]:
    """调用 VTC 渲染文字为图像并在内存中转换为 Base64，同时异步保存，返回(Base64, 路径, 耗时)"""
    start_render = time.perf_counter()
    img, char_count = vtc_tool.render_text_to_image(
        ob_text, 
        use_compact_mode=True, 
        max_width=max_width,
        max_height=max_height,
    )

    if compression_factor != 1.0:
        img = vtc_tool.compress_image_arrays(
            [img], compression_factor=compression_factor
        )[0]
    
    render_time = time.perf_counter() - start_render

    # 内存中转 Base64，避免读盘耗时
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 物理保存图片留作记录（注：如需进一步提速，可使用 ThreadPoolExecutor 将这三行放进异步任务）
    os.makedirs(output_dir, exist_ok=True)
    img_filename = f"obs_{step_id}.png"
    img_path = os.path.join(output_dir, img_filename)
    img.save(img_path)
    
    return img_base64, img_path, render_time

class DataLoader:
    """数据加载器：负责读取与预处理数据集"""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        if self.data_path.endswith('.parquet'):
            return pd.read_parquet(self.data_path)
        if self.data_path.endswith('.jsonl'):
            return pd.read_json(self.data_path, lines=True)
        raise ValueError("Unsupported data format. Please use .parquet or .jsonl")

    def load_and_select(self, max_samples: int, sample_seed: Optional[int] = None) -> pd.DataFrame:
        df = self.load()
        total_samples = min(max_samples, len(df))
        if sample_seed is None or sample_seed == 0:
            return df.head(total_samples).reset_index(drop=True)
        return df.sample(n=total_samples, random_state=sample_seed).reset_index(drop=True)

class TextBrowserEnv:
    """环境交互器：负责与 verl-tool 服务器（TextBrowser）交互"""
    def __init__(
        self,
        env_url: str = "http://localhost:5000/get_observation",
        request_timeout: float = 300.0,
        pool_size: int = 128,
        browser_url: str = "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    ):
        self.env_url = env_url
        self.request_timeout = request_timeout
        self.default_extra_fields = [{
            # "url": "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
            "url": browser_url
        }]
        
        # ==================== 新增：启用 Session 保持长连接 ====================
        self.session = requests.Session()
        # 设置连接池大小。为了匹配你设定的 num_workers=32，这里设置一个稍大的裕量(100)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            pool_block=True,
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        # =======================================================================

    def step(self, trajectory_id: str, action: str, is_finish: bool) -> str:
        data = {
            "trajectory_ids": [trajectory_id],
            "actions": [action],
            "finish": [is_finish],
            "extra_fields": self.default_extra_fields
        }
        try:
            # 修改点：使用 self.session.post 代替 requests.post
            resp = self.session.post(
                self.env_url,
                json=data,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            # # =================================================================
            # # 🔴 原有逻辑：将服务器完整的返回结果写入本地日志文件
            # # =================================================================
            # log_dir = "./debug_logs"
            # os.makedirs(log_dir, exist_ok=True)
            # log_file = os.path.join(log_dir, "server_full_responses.log")
            
            # try:
            #     with open(log_file, "a", encoding="utf-8") as f:
            #         f.write(f"\n{'='*80}\n")
            #         f.write(f"⏱️ Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Trajectory: {trajectory_id}\n")
            #         f.write(f"➡️ Sent Action: {repr(action)}\n")
            #         f.write(f"⬅️ Full Server Response:\n")
            #         f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            #         f.write(f"{'='*80}\n")
            # except Exception as log_e:
            #     print(f"[Log Error] Failed to write log: {log_e}")
            # # =================================================================
            
            raw_obs = result.get('observations', [""])[0]
            cleaned_obs = self._clean_observation(raw_obs)
            if cleaned_obs.lstrip().startswith("Error:"):
                raise RuntimeError(cleaned_obs)
            return cleaned_obs
        except Exception as e:
            print(f"[Env Error | Trajectory {trajectory_id}] {e}")
            raise RuntimeError(
                f"Text-browser request failed for trajectory {trajectory_id}: {e}"
            ) from e

    def _clean_observation(self, raw_obs: Any) -> str:
        """清洗环境返回的观测文本，加入严格的类型校验"""
        if isinstance(raw_obs, dict):
            import json
            return json.dumps(raw_obs, ensure_ascii=False)
            
        try:
            if isinstance(raw_obs, str) and 'Observation:\n' in raw_obs:
                obs = raw_obs.split('Observation:\n')[1]
                if '\nParsed Previous Action:' in obs:
                    obs = obs.split('\nParsed Previous Action:')[0]
                return obs
            return str(raw_obs)
        except Exception:
            return str(raw_obs)


class LLMClient:
    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8008/v1/",
        request_timeout: float = 300.0,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        pool_size: int = 128,
    ):
        self.base_url = base_url if base_url.endswith('/') else base_url + '/'
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # ==================== 新增：启用 Session 保持长连接 ====================
        self.session = requests.Session()
        # 同理，为与 vLLM 通信的接口配置连接池
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            pool_block=True,
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        # =======================================================================

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, system_prompt: str, user_prompt: str, model: str = "custom-llm", 
                 image_base64: Optional[str] = None, temperature: Optional[float] = None) -> Tuple[str, float]:
        start_llm = time.perf_counter()
        
        if image_base64:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        else:
            user_content = user_prompt

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            # 使用 self.session.post 保持长连接
            resp = self.session.post(
                f"{self.base_url}chat/completions",
                json=payload,
                timeout=self.request_timeout,
            )
            llm_time = time.perf_counter() - start_llm
            
            if resp.status_code != 200:
                print(f"\n❌ [vLLM 服务端报错] 状态码: {resp.status_code}")
                print(f"❌ [详细原因]: {resp.text}\n")
                raise RuntimeError(
                    f"vLLM returned HTTP {resp.status_code}: {resp.text[:1000]}"
                )
            
            data = resp.json()
            if "choices" not in data:
                print(f"\n❓ [诡异的成功响应] vLLM 返回了 200，但内容不符合标准格式:")
                print(f"❓ [实际内容]: {resp.text}\n")
                raise RuntimeError(
                    f"vLLM response has no choices: {resp.text[:1000]}"
                )
            
            return data["choices"][0]["message"]["content"], llm_time
            
        except Exception as e:
            llm_time = time.perf_counter() - start_llm
            print(f"\n🚨 [请求链路崩溃] 异常类型: {type(e).__name__}, 信息: {str(e)}\n")
            if 'resp' in locals():
                print(f"🚨 [实际收到的报文]: {resp.text[:500]}\n")
            raise RuntimeError(f"vLLM request failed: {e}") from e

class TrajectoryPipeline:
    """主 Pipeline：调度数据、环境和模型，记录轨迹"""
    def __init__(self, env: TextBrowserEnv, llm: LLMClient, system_prompt_path: str, output_file: str, 
                 use_vlm: bool = False, vtc_tool: Optional[VTCTool] = None,
                 image_output_dir: str = "./obs_images", compression_factor: float = 1.2,
                 image_max_width: int = 2048, image_max_height: int = 2048):
        self.env = env
        self.llm = llm
        self.output_file = output_file
        self.use_vlm = use_vlm
        self.vtc_tool = vtc_tool
        self.image_output_dir = image_output_dir
        self.compression_factor = compression_factor
        self.image_max_width = image_max_width
        self.image_max_height = image_max_height
        
        self.file_lock = threading.Lock()
        
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
#         self.user_prompt_template = """
# Objective: {}
# Observation: {}
# HISTORY_info: {}
# """
        self.user_prompt_template = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

    def extract_command(self, text: str) -> str:
        tags = ["action"]
        for tag in tags:
            pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
            blocks = re.findall(pattern, text, re.DOTALL)
            if blocks:
                return blocks[-1].strip()
        
        blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
        if not blocks:
            return ""
        return blocks[-1].strip().replace("```", "").strip()

    def extract_conclusion(self, text: str) -> str:
        blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)
        if not blocks:
            return ""
        return blocks[-1].strip()

    def save_trajectory(self, trajectory_data: Dict[str, Any]):
        with self.file_lock:
            with open(self.output_file, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")

    def run_single_episode(self, question: str, ground_truth: List[str], sample_idx: int, trial_idx: int = 1, max_steps: int = 30, model: str = "custom-llm") -> bool:
        tar_id = str(uuid.uuid4())
        history_actions = "\n"
        history_info = "\n"
        ground_truth = ensure_answer_list(ground_truth)
        
        # 记录首次环境耗时
        t0_env = time.perf_counter()
        current_obs = self.env.step(tar_id, "", False)
        initial_env_latency = time.perf_counter() - t0_env
        
        # === 干净的主轨迹数据 ===
        trajectory = {
            "id": tar_id,
            "trial_idx": trial_idx,      
            "sample_idx": sample_idx,
            "question": question,
            "ground_truth": ground_truth,
            "steps": [],
            "final_conclusion": "",
            "success": False,
            "success_em": False,
            "success_substring": False,
            "inference_config": {
                "use_vlm": self.use_vlm,
                "compression_factor": self.compression_factor,
                "image_max_width": self.image_max_width,
                "image_max_height": self.image_max_height,
                "max_steps": max_steps,
                "model": model,
            }
        }

        # === 单独记录耗时数据结构 ===
        metrics_log = {
            "id": tar_id,
            "trial_idx": trial_idx,
            "sample_idx": sample_idx,
            "initial_env_latency": initial_env_latency,
            "steps": []
        }

        for step in range(max_steps):
            img_path = None
            img_b64 = None
            text_obs = current_obs
            step_render_lat = 0.0
            
            # 1. 图像渲染
            if self.use_vlm and self.vtc_tool:
                step_id = f"{tar_id}_step_{step}"
                img_b64, img_path, step_render_lat = generate_image_for_observation(
                    self.vtc_tool,
                    current_obs,
                    self.image_output_dir,
                    step_id,
                    self.compression_factor,
                    self.image_max_width,
                    self.image_max_height,
                )
                text_obs = "<Image provided attached. Please refer to the visual observation.>"

            # real_prompt = self.user_prompt_template.format(question, text_obs, history_info)
            real_prompt = self.user_prompt_template.format(question, text_obs, history_actions, history_info)
            
            # 2. 模型推理
            response_text, step_llm_lat = self.llm.generate(
                system_prompt=self.system_prompt, 
                user_prompt=real_prompt, 
                model=model, 
                image_base64=img_b64
            )
            
            action = self.extract_command(response_text)
            conclusion = self.extract_conclusion(response_text)
            
            # 主结果中不再保存 metrics
            step_data = {
                "step": step,
                "observation": current_obs,
                "image_path": img_path,
                "prompt": real_prompt,
                "model_response": response_text,
                "action": action,
                "conclusion": conclusion
            }
            trajectory["steps"].append(step_data)

            if action: history_actions += action + "\n"
            if conclusion: history_info += conclusion + "\n"

            is_stop = "stop" in action.lower()
            
            # 3. 环境交互
            t_env = time.perf_counter()
            current_obs = self.env.step(tar_id, response_text, is_stop)
            step_env_lat = time.perf_counter() - t_env

            # 追加耗时信息到单独的 log 结构
            metrics_log["steps"].append({
                "step": step,
                "render_latency": round(step_render_lat, 4),
                "llm_latency": round(step_llm_lat, 4),
                "env_latency": round(step_env_lat, 4)
            })

            # 控制台输出一下进度（多线程下也能有个直观感知）
            # print(f"[Task {sample_idx} | Step {step}] Render: {step_render_lat:.2f}s | LLM: {step_llm_lat:.2f}s | Env: {step_env_lat:.2f}s")

            if is_stop:
                match = re.search(r'stop\s*\[(.*?)\]', action, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                else:
                    final_answer = action.replace("stop", "").strip(" []")
                
                trajectory["final_conclusion"] = final_answer
                trajectory["success_em"] = bool(em_check(final_answer, ground_truth)) if ground_truth else False
                trajectory["success_substring"] = bool(substring_check(final_answer, ground_truth)) if ground_truth else False
                trajectory["success"] = trajectory["success_em"]
                break
                
        # 1. 保存主轨迹数据
        self.save_trajectory(trajectory)
        
        # 2. 独立保存耗时指标到另一个文件 (需要在 init 里加锁并开辟一个文件，或直接临时追加)
        with self.file_lock: # 复用 file_lock 保证写入安全
            metrics_file = self.output_file.replace(".jsonl", "_metrics.jsonl")
            with open(metrics_file, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(metrics_log, ensure_ascii=False) + "\n")

        return trajectory["success"]

def main():
    parser = argparse.ArgumentParser(description="Multi-turn Text/Vision Browser Agent Pipeline")
    parser.add_argument(
        '--output_file',
        type=str,
        default=str(PROJECT_ROOT / 'evaluate/results/test_results.jsonl'),
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=str(PROJECT_ROOT / 'prompt/system_prompt_with_history_info.txt'),
    )
    parser.add_argument('--max_samples', type=int, default=300)
    parser.add_argument('--sample_seed', '--sample-seed', type=str, default='', help="Random seed for sample selection. Empty or 0 keeps sequential order.")
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Skip (trial_idx, sample_idx) pairs already present in output_file.",
    )
    parser.add_argument('--base_url', type=str, default='http://localhost:8008/v1/')
    parser.add_argument('--env_url', '--env-url', type=str, default='http://localhost:5000/get_observation')
    parser.add_argument(
        '--browser_url', '--browser-url',
        type=str,
        default='http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing',
    )
    parser.add_argument('--model', type=str, default='custom-llm')
    parser.add_argument('--max_steps', '--max-steps', type=int, default=30)
    parser.add_argument('--max_tokens', '--max-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--llm_request_timeout', '--llm-request-timeout', type=float, default=300.0)
    parser.add_argument('--env_request_timeout', '--env-request-timeout', type=float, default=300.0)
    
    # 测试轮数
    parser.add_argument('--num_trials', type=int, default=8, help="在相同的样本上重复测试的轮数，用于计算均值")
    
    # VLM 相关参数
    parser.add_argument('--use_vlm', action='store_true', help="Enable Vision-Language Model mode")
    parser.add_argument('--image_output_dir', type=str, default='./obs_images', help="Directory to save generated observation images")
    parser.add_argument('--compression_factor', '--compression-factor', type=float, default=1.2)
    parser.add_argument('--image_max_width', '--image-max-width', type=int, default=2048)
    parser.add_argument('--image_max_height', '--image-max-height', type=int, default=2048)
    
    # 并行 worker 数量参数
    parser.add_argument('--num_workers', type=int, default=4, help="Number of concurrent execution threads")
    
    args = parser.parse_args()

    if args.num_workers <= 0:
        parser.error('--num-workers must be positive')
    if args.max_steps <= 0 or args.max_tokens <= 0:
        parser.error('--max-steps and --max-tokens must be positive')
    if args.compression_factor <= 0:
        parser.error('--compression-factor must be positive')
    if args.image_max_width <= 0 or args.image_max_height <= 0:
        parser.error('--image dimensions must be positive')

    output_parent = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_parent, exist_ok=True)

    data_loader = DataLoader(args.data_path)
    env = TextBrowserEnv(
        env_url=args.env_url,
        request_timeout=args.env_request_timeout,
        pool_size=args.num_workers,
        browser_url=args.browser_url,
    )
    llm = LLMClient(
        base_url=args.base_url,
        request_timeout=args.llm_request_timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        pool_size=args.num_workers,
    )
    vtc_tool = VTCTool() if args.use_vlm else None
    
    pipeline = TrajectoryPipeline(
        env=env, 
        llm=llm, 
        system_prompt_path=args.system_prompt, 
        output_file=args.output_file,
        use_vlm=args.use_vlm,
        vtc_tool=vtc_tool,
        image_output_dir=args.image_output_dir,
        compression_factor=args.compression_factor,
        image_max_width=args.image_max_width,
        image_max_height=args.image_max_height,
    )

    sample_seed_text = str(args.sample_seed).strip()
    sample_seed = None if sample_seed_text in ("", "0") else int(sample_seed_text)
    data_df = data_loader.load_and_select(max_samples=args.max_samples, sample_seed=sample_seed)
    total_samples = len(data_df)
    sample_mode = "sequential" if sample_seed is None else f"random(seed={sample_seed})"

    # 提取固定批次的数据
    tasks = []
    for i, row in data_df.iterrows():
        question = row["extra_info"]["question"]
        gt = ensure_answer_list(row["extra_info"].get("golden_answers"))
        if not gt:
            raise ValueError(f"Sample {i + 1} has no extra_info.golden_answers")
        tasks.append((i + 1, question, gt))

    print(
        f"🚀 Started evaluation! Max Samples: {total_samples} | "
        f"Sample Mode: {sample_mode} | Total Trials: {args.num_trials} | "
        f"Workers: {args.num_workers} | Compression: {args.compression_factor} | "
        f"Env: {args.env_url} | vLLM: {args.base_url}"
    )

    completed_pairs = set()
    if args.resume and os.path.isfile(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as completed_file:
            lines = completed_file.readlines()
            nonempty_line_numbers = [
                line_number
                for line_number, line in enumerate(lines, start=1)
                if line.strip()
            ]
            last_nonempty = nonempty_line_numbers[-1] if nonempty_line_numbers else 0
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    completed_pairs.add((
                        int(record.get("trial_idx", 1)),
                        int(record["sample_idx"]),
                    ))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    location = "final" if line_number == last_nonempty else "interior"
                    print(
                        f"⚠️ Ignoring malformed {location} resume record at "
                        f"{args.output_file}:{line_number}: {exc}"
                    )
        print(
            f"♻️ Resume enabled: found {len(completed_pairs)} completed "
            f"(trial_idx, sample_idx) pairs in {args.output_file}"
        )
    
    # 记录每次 trial 的准确率
    trial_accuracies = []
    total_failures = 0

    start_time = time.time()

    # 遍历设定的所有测试轮次
    for trial in range(1, args.num_trials + 1):
        print(f"\n{'='*20} 🟢 Starting Trial {trial}/{args.num_trials} {'='*20}")
        
        completed_count = 0
        success_count = 0
        failed_count = 0
        pending_tasks = [
            task for task in tasks if (trial, task[0]) not in completed_pairs
        ]
        skipped_count = len(tasks) - len(pending_tasks)
        if skipped_count:
            print(
                f"[Trial {trial}] ♻️ Skipping {skipped_count} completed samples; "
                f"{len(pending_tasks)} remain"
            )
        if not pending_tasks:
            print(
                f"[Trial {trial}] ✅ Nothing to resume; "
                f"all {len(tasks)} samples are complete"
            )
            continue
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交任务时带上当前的 trial 号码
            future_to_task = {
                executor.submit(
                    pipeline.run_single_episode,
                    q,
                    gt,
                    idx,
                    trial,
                    max_steps=args.max_steps,
                    model=args.model,
                ): idx
                for idx, q, gt in pending_tasks
            }
            
            for future in as_completed(future_to_task):
                idx = future_to_task[future]
                try:
                    is_success = future.result()
                    completed_count += 1
                    if is_success:
                        success_count += 1
                    print(f"[Trial {trial}] ✅ Finished sample {idx}/{total_samples} | Accuracy: {success_count}/{completed_count} ({(success_count/completed_count)*100:.2f}%)")
                except Exception as exc:
                    failed_count += 1
                    total_failures += 1
                    print(f"[Trial {trial}] ❌ Sample {idx} generated an exception: {exc}")
        
        # 单论测试结束，计算该轮准确率
        current_trial_acc = (success_count / completed_count) * 100 if completed_count > 0 else 0.0
        trial_accuracies.append(current_trial_acc)
        print(
            f"{'='*20} 🔴 Finished Trial {trial} | Final Accuracy: "
            f"{current_trial_acc:.2f}% | Failed (not saved): {failed_count} {'='*20}"
        )

    # 记录结束时间并计算耗时
    end_time = time.time()
    total_seconds = end_time - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 最终结果统计
    print("\n\n" + "🌟" * 25)
    print("ALL TRIALS COMPLETED - FINAL STATISTICS")
    print("🌟" * 25)
    
    for i, acc in enumerate(trial_accuracies):
        print(f"  Trial {i + 1} Accuracy: {acc:.2f}%")
        
    mean_accuracy = sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0.0
    print(f"\n  🏆 Average Accuracy across {args.num_trials} trials: **{mean_accuracy:.2f}%**")
    print("🌟" * 25 + "\n")

    # 打印格式化后的总耗时
    print(f"  ⏱️  Total Time Elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("🌟" * 25 + "\n")
    if total_failures:
        print(
            f"Evaluation has {total_failures} failed sample-evaluations. "
            "They were not saved and will be retried with --resume."
        )
        raise SystemExit(2)

if __name__ == "__main__":
    main()
