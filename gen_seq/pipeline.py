import json
import pandas as pd
import requests
from openai import OpenAI
import argparse
import uuid
import re
from typing import List, Dict, Any, Optional

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
            print(f"[Env Error] {e}")
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
    """大模型推理客户端：兼容 OpenAI API，支持扩展多模态"""
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8008/v1/"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system_prompt: str, user_prompt: str, model: str = "custom-llm", temperature: float = 0.3) -> str:
        # 预留多模态扩展接口：后续如果引入 VTC，可以在 message content 中按 OpenAI 视觉格式传入 image_url
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM Error] {e}")
            return ""

class TrajectoryPipeline:
    """主 Pipeline：调度数据、环境和模型，记录轨迹"""
    def __init__(self, env: TextBrowserEnv, llm: LLMClient, system_prompt_path: str, output_file: str):
        self.env = env
        self.llm = llm
        self.output_file = output_file
        
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        self.user_prompt_template = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

    def extract_command(self, text: str) -> str:
        # 支持多种标签：<action>、<cmd>、<command> 等
        tags = ['action']
        
        for tag in tags:
            pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
            blocks = re.findall(pattern, text, re.DOTALL)
            if blocks:
                return blocks[-1].strip()
        
        # 回退到代码块
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
        with open(self.output_file, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")

    def run_single_episode(self, question: str, ground_truth: str, max_steps: int = 30, model: str = "custom-llm"):
        tar_id = str(uuid.uuid4())
        history_actions = "\n"
        history_info = "\n"
        
        # 初始化环境，获取初始观测
        current_obs = self.env.step(tar_id, "", False)
        
        trajectory = {
            "id": tar_id,
            "question": question,
            "ground_truth": ground_truth,
            "steps": [],
            "final_conclusion": "",
            "success": False
        }

        for step in range(max_steps):
            # 1. 组装 Prompt (此处为未来接入人为干预提供切入点)
            real_prompt = self.user_prompt_template.format(
                question, current_obs, history_actions, history_info
            )
            
            # 2. 模型推理 (此处为未来接入 VTC 预留逻辑空间)
            response_text = self.llm.generate(self.system_prompt, real_prompt, model=model)
            
            # 3. 解析动作与总结
            action = self.extract_command(response_text)
            conclusion = self.extract_conclusion(response_text)
            
            # 记录轨迹
            step_data = {
                "step": step,
                "observation": current_obs,
                "prompt": real_prompt,
                "model_response": response_text,
                "action": action,
                "conclusion": conclusion
            }
            trajectory["steps"].append(step_data)

            # 更新历史
            if action: history_actions += action + "\n"
            if conclusion: history_info += conclusion + "\n"

            # 4. 判断终止条件并与环境交互
            is_stop = "stop" in action.lower()
            current_obs = self.env.step(tar_id, response_text, is_stop)

            if is_stop:
                # 使用正则表达式从 action 中提取 stop [answer] 的 answer 部分
                match = re.search(r'stop\s*\[(.*?)\]', action, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                else:
                    # 如果格式不标准但包含了 stop，做个 fallback
                    final_answer = action.replace("stop", "").strip(" []")
                
                trajectory["final_conclusion"] = final_answer
                
                # 简单正确率判断逻辑：如果是 N/A 直接视为失败，否则判断 ground_truth 是否在提取出的 answer 中
                if final_answer == "N/A" or final_answer == '"N/A"':
                    trajectory["success"] = False
                else:
                    trajectory["success"] = (ground_truth.lower() in final_answer.lower()) if ground_truth else False
                
                break
                
        # 保存完整轨迹
        self.save_trajectory(trajectory)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn Text Browser Agent Pipeline")
    parser.add_argument('--output_file', type=str, default='./results/test_results.jsonl')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--system_prompt', type=str, default='./system_prompt_with_history_info.txt')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--base_url', type=str, default='http://localhost:8008/v1/')
    parser.add_argument('--model', type=str, default='custom-llm')
    args = parser.parse_args()

    # 初始化各个组件
    data_loader = DataLoader(args.data_path)
    env = TextBrowserEnv(env_url="http://localhost:5000/get_observation")
    llm = LLMClient(base_url=args.base_url)
    # llm = LLMClient(base_url="https://api.deepseek.com", api_key="sk-5b242d486cf743029a466acd3924046c")
    pipeline = TrajectoryPipeline(env, llm, args.system_prompt, args.output_file)

    # 加载数据
    data_df = data_loader.load_and_sample()

    # 运行主循环
    for i, row in data_df.iterrows():
        if i >= args.max_samples:
            break
            
        question = row["extra_info"]["question"]
        gt = row["extra_info"]["selected_answer"]
        
        print(f"Processing sample {i+1}/{args.max_samples}...")
        pipeline.run_single_episode(question, gt, model=args.model)

if __name__ == "__main__":
    main()



"""
python pipeline.py \
    --output_file=./results/NQ_test_results.jsonl \
    --data_path=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/benchmark/v1/nq/test-00000-of-00001.parquet \
    --system_prompt=/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/system_prompt_with_history_info.txt
"""


# 启动本地 vLLM OpenAI 兼容服务器
# 如果是多模态模型，可加上 --enforce-eager 或针对特定模型的参数
"""
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python -m vllm.entrypoints.openai.api_server \
    --model /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/models/Qwen2.5 \
    --served-model-name custom-llm \
    --host 0.0.0.0 \
    --port 8008 \
    --trust-remote-code \
    --max-model-len 4096 \
    --tensor-parallel-size 4
"""