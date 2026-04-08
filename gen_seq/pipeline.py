import json
import pandas as pd
import requests
import argparse
import uuid
import re
import os
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from VTC_tool.VTC_tool import VTCTool

def generate_image_for_observation(vtc_tool: VTCTool, ob_text: str, output_dir: str, step_id: str) -> str:
    """调用 VTC 渲染文字为图像并保存，返回相对路径"""
    img, char_count = vtc_tool.render_text_to_image(
        ob_text, 
        use_compact_mode=True, 
        max_width=1024, 
        max_height=1024
    )
    os.makedirs(output_dir, exist_ok=True)
    img_filename = f"obs_{step_id}.png"
    img_path = os.path.join(output_dir, img_filename)
    img.save(img_path)
    return img_path

class DataLoader:
    """数据加载器：负责读取与预处理数据集"""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_and_sample(self, frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
        if self.data_path.endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.jsonl'):
            df = pd.read_json(self.data_path, lines=True)
        else:
            raise ValueError("Unsupported data format. Please use .parquet or .jsonl")
        
        return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

class TextBrowserEnv:
    """环境交互器：负责与 verl-tool 服务器（TextBrowser）交互"""
    def __init__(self, env_url: str = "http://localhost:5000/get_observation"):
        self.env_url = env_url
        self.default_extra_fields = [{
            "url": "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        }]

    def step(self, trajectory_id: str, action: str, is_finish: bool) -> str:
        data = {
            "trajectory_ids": [trajectory_id],
            "actions": [action],
            "finish": [is_finish],
            "extra_fields": self.default_extra_fields
        }
        try:
            resp = requests.post(self.env_url, json=data, timeout=1200)
            resp.raise_for_status()
            result = resp.json()
            raw_obs = result.get('observations', [""])[0]
            return self._clean_observation(raw_obs)
        except Exception as e:
            print(f"[Env Error | Trajectory {trajectory_id}] {e}")
            return f"Error: {str(e)}"

    def _clean_observation(self, raw_obs: str) -> str:
        """清洗环境返回的观测文本"""
        try:
            if 'Observation:\n' in raw_obs:
                obs = raw_obs.split('Observation:\n')[1]
                if '\nParsed Previous Action:' in obs:
                    obs = obs.split('\nParsed Previous Action:')[0]
                return obs
            return raw_obs
        except Exception:
            return raw_obs

class LLMClient:
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8008/v1/"):
        self.base_url = base_url if base_url.endswith('/') else base_url + '/'
        self.api_key = api_key

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, system_prompt: str, user_prompt: str, model: str = "custom-llm", temperature: float = 0.3, image_path: Optional[str] = None) -> str:
        # 构造 Payload
        if image_path and os.path.exists(image_path):
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._encode_image(image_path)}"}}
            ]
        else:
            user_content = user_prompt

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": 1024
        }
        
        try:
            resp = requests.post(f"{self.base_url}chat/completions", json=payload, timeout=120)
            
            if resp.status_code != 200:
                print(f"\n❌ [vLLM 服务端报错] 状态码: {resp.status_code}")
                print(f"❌ [详细原因]: {resp.text}\n")
                return ""
            
            data = resp.json()
            if "choices" not in data:
                print(f"\n❓ [诡异的成功响应] vLLM 返回了 200，但内容不符合标准格式:")
                print(f"❓ [实际内容]: {resp.text}\n")
                return ""
            
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"\n🚨 [请求链路崩溃] 异常类型: {type(e).__name__}, 信息: {str(e)}\n")
            if 'resp' in locals():
                print(f"🚨 [实际收到的报文]: {resp.text[:500]}\n")
            return ""

class TrajectoryPipeline:
    """主 Pipeline：调度数据、环境和模型，记录轨迹"""
    def __init__(self, env: TextBrowserEnv, llm: LLMClient, system_prompt_path: str, output_file: str, 
                 use_vlm: bool = False, vtc_tool: Optional[VTCTool] = None, image_output_dir: str = "./obs_images"):
        self.env = env
        self.llm = llm
        self.output_file = output_file
        self.use_vlm = use_vlm
        self.vtc_tool = vtc_tool
        self.image_output_dir = image_output_dir
        
        # 【关键修改】：为文件写入添加线程锁
        self.file_lock = threading.Lock()
        
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        self.user_prompt_template = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

    def extract_command(self, text: str) -> str:
        tags = []
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
        # 【关键修改】：使用锁保护文件写入操作
        with self.file_lock:
            with open(self.output_file, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")

    def run_single_episode(self, question: str, ground_truth: str, sample_idx: int, max_steps: int = 30, model: str = "custom-llm"):
        tar_id = str(uuid.uuid4())
        history_actions = "\n"
        history_info = "\n"
        
        # 初始化环境，获取初始观测
        current_obs = self.env.step(tar_id, "", False)
        
        trajectory = {
            "id": tar_id,
            "sample_idx": sample_idx, # 记录样本编号方便对照
            "question": question,
            "ground_truth": ground_truth,
            "steps": [],
            "final_conclusion": "",
            "success": False
        }

        for step in range(max_steps):
            img_path = None
            text_obs = current_obs
            
            # 1. 如果使用 VLM，将当前观测转换为图像，同时隐去 Prompt 中的文本观察
            if self.use_vlm and self.vtc_tool:
                step_id = f"{tar_id}_step_{step}"
                img_path = generate_image_for_observation(
                    self.vtc_tool, current_obs, self.image_output_dir, step_id
                )
                text_obs = "<Image provided attached. Please refer to the visual observation.>"

            # 2. 组装 Prompt
            real_prompt = self.user_prompt_template.format(
                question, text_obs, history_actions, history_info
            )
            
            # 3. 模型推理
            response_text = self.llm.generate(
                system_prompt=self.system_prompt, 
                user_prompt=real_prompt, 
                model=model, 
                image_path=img_path
            )
            
            # 4. 解析动作与总结
            action = self.extract_command(response_text)
            conclusion = self.extract_conclusion(response_text)
            
            # 记录轨迹
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

            # 5. 判断终止条件并与环境交互
            is_stop = "stop" in action.lower()
            current_obs = self.env.step(tar_id, response_text, is_stop)

            if is_stop:
                match = re.search(r'stop\s*\[(.*?)\]', action, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                else:
                    final_answer = action.replace("stop", "").strip(" []")
                
                trajectory["final_conclusion"] = final_answer
                
                if final_answer == "N/A" or final_answer == '"N/A"':
                    trajectory["success"] = False
                else:
                    trajectory["success"] = (ground_truth.lower() in final_answer.lower()) if ground_truth else False
                
                break
                
        # 保存完整轨迹
        self.save_trajectory(trajectory)
        return trajectory["success"]


def main():
    parser = argparse.ArgumentParser(description="Multi-turn Text/Vision Browser Agent Pipeline")
    parser.add_argument('--output_file', type=str, default='./results/test_results.jsonl')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--system_prompt', type=str, default='./system_prompt_with_history_info.txt')
    parser.add_argument('--max_samples', type=int, default=300)
    parser.add_argument('--base_url', type=str, default='http://localhost:8008/v1/')
    parser.add_argument('--model', type=str, default='custom-llm')
    
    # VLM 相关参数
    parser.add_argument('--use_vlm', action='store_true', help="Enable Vision-Language Model mode")
    parser.add_argument('--image_output_dir', type=str, default='./obs_images', help="Directory to save generated observation images")
    
    # 并行 worker 数量参数
    parser.add_argument('--num_workers', type=int, default=4, help="Number of concurrent execution threads")
    
    args = parser.parse_args()

    # 初始化各个组件
    data_loader = DataLoader(args.data_path)
    env = TextBrowserEnv(env_url="http://localhost:5000/get_observation")
    llm = LLMClient(base_url=args.base_url)
    vtc_tool = VTCTool() if args.use_vlm else None
    
    pipeline = TrajectoryPipeline(
        env=env, 
        llm=llm, 
        system_prompt_path=args.system_prompt, 
        output_file=args.output_file,
        use_vlm=args.use_vlm,
        vtc_tool=vtc_tool,
        image_output_dir=args.image_output_dir
    )

    data_df = data_loader.load_and_sample()
    total_samples = min(args.max_samples, len(data_df))

    # 提取需要跑的任务
    tasks = []
    for i, row in data_df.iterrows():
        if i >= total_samples:
            break
        question = row["extra_info"]["question"]
        gt = row["extra_info"]["selected_answer"]
        tasks.append((i + 1, question, gt))

    print(f"🚀 Starting evaluation with {args.num_workers} parallel workers...")
    
    completed_count = 0
    success_count = 0
    
    # 【关键修改】：使用 ThreadPoolExecutor 进行多线程并发
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(pipeline.run_single_episode, q, gt, idx, model=args.model): idx 
            for idx, q, gt in tasks
        }
        
        # 接收完成的任务
        for future in as_completed(future_to_task):
            idx = future_to_task[future]
            try:
                is_success = future.result()
                completed_count += 1
                if is_success:
                    success_count += 1
                print(f"✅ Finished sample {idx}/{total_samples} | Current Accuracy: {success_count}/{completed_count} ({(success_count/completed_count)*100:.2f}%)")
            except Exception as exc:
                completed_count += 1
                print(f"❌ Sample {idx} generated an exception: {exc}")

if __name__ == "__main__":
    main()


"""
python -m gen_seq.pipeline \
    --output_file=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_step-opsrc-5000_rl-200/NQ_test_results.jsonl \
    --data_path=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/benchmark/v1/nq/test-00000-of-00001.parquet \
    --system_prompt=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/system_prompt_with_history_info.txt \
    --max_samples=300 \
    --base_url=http://localhost:8008/v1/ \
    --model='custom-llm-1' \
    --use_vlm \
    --image_output_dir=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_step-opsrc-5000_rl-200/NQ_obs_images \
    --num_workers 4

python -m gen_seq.pipeline \
    --output_file=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_step-opsrc-5000_rl-200/hotpot_test_results.jsonl \
    --data_path=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/benchmark/v1/hotpot/validation-00000-of-00001.parquet \
    --system_prompt=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/system_prompt_with_history_info.txt \
    --max_samples=300 \
    --base_url=http://localhost:8008/v1/ \
    --model='custom-llm-1' \
    --use_vlm \
    --image_output_dir=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_step-opsrc-5000_rl-200/hotpot_obs_images \
    --num_workers 4

python -m gen_seq.pipeline \
    --output_file=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/origin_test_results.jsonl \
    --data_path=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/benchmark/v1/nq/test-00000-of-00001.parquet \
    --system_prompt=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/system_prompt_with_history_info.txt \
    --max_samples=30 \
    --base_url=http://localhost:8008/v1/ \
    --model='custom-llm'
    --num_workers 4
"""


# 启动本地 vLLM OpenAI 兼容服务器
# 如果是多模态模型，可加上 --enforce-eager 或针对特定模型的参数
"""
CUDA_VISIBLE_DEVICES=3,4 \
python -m vllm.entrypoints.openai.api_server \
    --model /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/output/Qwen2.5-VL-7B-Instruct-task-opsrc-5000stp-merged \
    --served-model-name custom-llm-1 \
    --host 0.0.0.0 \
    --port 8009 \
    --trust-remote-code \
    --max-model-len 8192 \
    --tensor-parallel-size 2

CUDA_VISIBLE_DEVICES=1,2 \
python -m vllm.entrypoints.openai.api_server \
    --model /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/models/Qwen2.5-VL-7B-Instruct \
    --served-model-name custom-llm \
    --host 0.0.0.0 \
    --port 8008 \
    --trust-remote-code \
    --max-model-len 8192 \
    --tensor-parallel-size 2
"""