import json
import pandas as pd
from typing import List, Dict, Any
import requests
import time
import argparse
import re
import uuid
import base64
from io import BytesIO
from PIL import Image
import traceback
from concurrent.futures import ThreadPoolExecutor

# 替换为新的 Venus API 配置
VENUS_URL = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
VENUS_TOKEN = "YOUR_VENUS_TOKEN_HERE" # 填入你的实际 Token

with open("/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    env_url = "http://localhost:5000/get_observation"
    extra_fields = [
        {"url": "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"}
    ]
    data = {
        "trajectory_ids": trajectory_ids,
        "actions": actions,
        "finish": finish,
        "extra_fields": extra_fields
    }
    try:
        resp = requests.post(env_url, json=data, timeout=1200)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

def get_response(prompt, model="deepseek-reasoner", temperature=0.3, max_retries=5):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {VENUS_TOKEN}'
    }
    
    payload = {
        'model': model,
        'messages': [{"role": "user", "content": prompt}],
        'temperature': temperature,
        'max_tokens': 1024
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(VENUS_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                wait_time = (attempt + 1) * 5
                print(f"⚠️ 服务器拥塞或请求失败 (状态码: {response.status_code})，{wait_time}秒后重试... 报错: {response.text}")
                time.sleep(wait_time)
                continue
                
            resp_json = response.json()
            message = resp_json['choices'][0]['message']
            
            normal_content = message.get('content', "")
            if not normal_content:
                normal_content = ""
            
            reasoning_content = message.get('reasoning_content', "")
                
            if reasoning_content:
                model_answer = f"<think>\n{reasoning_content}\n</think>\n{normal_content}"
            else:
                model_answer = normal_content
                
            if not model_answer.strip():
                print("⚠️ 警告：模型本次请求返回了完全空的内容，以占位符替换。")
                model_answer = "<action>stop [N/A]</action>"
                
            usage = resp_json.get('usage', {})
            p_tokens = usage.get('prompt_tokens', 0)
            c_tokens = usage.get('completion_tokens', 0)
            t_tokens = usage.get('total_tokens', 0)
            
            return model_answer, p_tokens, c_tokens, t_tokens

        except requests.exceptions.RequestException as e:
            wait_time = (attempt + 1) * 5
            print(f"⚠️ 网络请求异常，{wait_time}秒后重试... 错误: {e}")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ 遇到不可恢复错误: {e}")
            break
            
    return "ERROR_STILL_OCCURRED", 0, 0, 0

def extract_command(text):
    new_style = re.findall(r'<action>\s*(.*?)\s*</action>', text, re.DOTALL)
    if new_style:
        return new_style[-1].strip()
    
    old_style = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    if old_style:
        return old_style[-1].strip().replace("```","").strip()
    
    return " "

def extract_conclusion(text):
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)
    if not blocks: return " "
    return blocks[-1].strip()

# 【修改1】将直接写文件改为组装字典并在内存中返回
def format_a_data(system_prompt_text, user_prompt_text, assistant_response, token_usage=None):
    written_data = {
        "messages": [
            {"role": "system", "content": system_prompt_text.strip()},
            {"role": "user", "content": user_prompt_text},
            {"role": "assistant", "content": assistant_response}
        ],
        "subset": "corr_hotpot_new1369q_swift",
        "stage": "sft"
    }
    if token_usage:
        written_data["token_usage"] = token_usage
    return written_data

# 【修改2】返回值调整，收集该任务产生的所有记录，交由主线程统一按序写入
def Get_multi_turn_response(question, answer):
    tar_id = str(uuid.uuid4())
    history = "\n"
    history_info = "\n"
    obj = question
    obs = ""
    
    task_prompt_tokens = 0
    task_completion_tokens = 0
    task_total_tokens = 0
    
    task_records = []  # 暂存当前任务的所有的步骤数据
    eval_record = None # 暂存评估结果

    try:
        jsoned_data = call_tool_server([tar_id], [''], [False])
        if 'observations' in jsoned_data:
            obs = jsoned_data['observations'][0]
        else:
            print(f"警告: 服务器未返回 observations。完整返回: {jsoned_data}")
            return task_records, eval_record, task_prompt_tokens, task_completion_tokens, task_total_tokens
    except Exception as e:
        print(f"获取初始 observation 失败: {e}")
        return task_records, eval_record, task_prompt_tokens, task_completion_tokens, task_total_tokens

    for i in range(10):
        # ================= Busy 状态拦截 =================
        wait_count = 0
        max_wait = 2 
        
        while re.search(r'RootWebArea.*?busy:\s*(1|True|true)', str(obs), re.IGNORECASE) and wait_count < max_wait:
            obs_str = str(obs)
            if len(obs_str) > 300 and "RootWebArea" in obs_str:
                obs = re.sub(r'busy:\s*(1|True|true)', 'busy: False', obs_str, flags=re.IGNORECASE)
                break
                
            time.sleep(2)
            try:
                temp_action = "<think></think>\n<action>scroll [down]</action>"
                refresh_data = call_tool_server([tar_id], [temp_action], [False])
                
                temp_action = "<think></think>\n<action>scroll [up]</action>"
                call_tool_server([tar_id], [temp_action], [False])
                
                if 'observations' in refresh_data:
                    obs = refresh_data['observations'][0]
            except Exception as e:
                pass
            wait_count += 1
            
        if re.search(r'RootWebArea.*?busy:\s*(1|True|true)', str(obs), re.IGNORECASE):
            call_tool_server([tar_id], [''], [True])  
            return task_records, eval_record, task_prompt_tokens, task_completion_tokens, task_total_tokens
        # ==========================================

        try:
            obs = obs.split('Observation:\n')[1].split('\nParsed Previous Action:')[0]
        except:
            pass
        
        real_prompt = user_prompt.format(obj, obs, history, history_info)
        prompt = system_prompt + "\n\n" + real_prompt
        
        response, p_tokens, c_tokens, t_tokens = get_response(prompt, temperature=0.3)
        
        task_prompt_tokens += p_tokens
        task_completion_tokens += c_tokens
        task_total_tokens += t_tokens

        last_command = extract_command(response)
        last_info = extract_conclusion(response)

        history = history + last_command + "\n"
        history_info = history_info + last_info + "\n"

        try:
            jsoned_data = call_tool_server([tar_id], [response], [False])
            obs = jsoned_data['observations'][0]
        except Exception as e:
            pass
            
        current_step_tokens = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": t_tokens
        }
        
        # 将数据追加入暂存列表
        task_records.append(format_a_data(
            system_prompt_text=system_prompt,
            user_prompt_text=real_prompt,
            assistant_response=response,
            token_usage=current_step_tokens
        ))

        time.sleep(1.5)

        if "stop" in last_command:
            match = re.search(r'stop\s*\[(.*?)\]', last_command, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
            else:
                final_answer = last_command.replace("stop", "").strip(" []")
            
            success = (str(answer).lower() in final_answer.lower()) if answer else False
            
            eval_record = {
                "id": tar_id,
                "question": question,
                "ground_truth": answer,
                "final_answer": final_answer,
                "success": success
            }
            
            call_tool_server([tar_id], [response], [True])
            return task_records, eval_record, task_prompt_tokens, task_completion_tokens, task_total_tokens
            
    return task_records, eval_record, task_prompt_tokens, task_completion_tokens, task_total_tokens

# 【修改3】包装一个工作函数供线程池调用
def process_single_task(task_args):
    task_id, question, gt = task_args
    print(f"⏳ 开始执行任务 [{task_id}]...")
    records, eval_record, p_tok, c_tok, t_tok = Get_multi_turn_response(question, gt)
    return task_id, records, eval_record, p_tok, c_tok, t_tok

number_to_process = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
    parser.add_argument('--output_file', type=str, default='./test.jsonl')
    parser.add_argument('--data_path', type=str, default='')
    # 【新增】并发度参数
    parser.add_argument('--workers', type=int, default=4, help="并发线程数")
    args = parser.parse_args()
    
    print(f"Output file: {args.output_file}\nData file: {args.data_path}\nWorkers: {args.workers}")

    data_df = pd.read_parquet(args.data_path).sample(frac=1, random_state=42).reset_index(drop=True)

    global_prompt_tokens = 0
    global_completion_tokens = 0
    global_total_tokens = 0

    # 提取需要的任务参数列表
    task_list = []
    for i, row in data_df.iterrows():
        question = row["extra_info"]["question"]
        gt = row["extra_info"]["selected_answer"]
        task_list.append((i + 1, question, gt))
        if len(task_list) == number_to_process:
            break

    result_file = args.output_file.replace(".jsonl", "_results.jsonl")
    if not result_file.endswith("_results.jsonl"): 
        result_file += "_results.jsonl"

    print(f"\n🚀 启动并发任务池，并发度 = {args.workers}")
    
    # 【修改4】使用 ThreadPoolExecutor 进行并发
    # 使用 executor.map 可以保证 yield 结果的顺序与 input 的 task_list 完全一致！
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_single_task, task_list):
            task_id, records, eval_record, p_tok, c_tok, t_tok = result
            
            # 1. 顺序写入当前任务产生的所有 SFT 轮次数据 (保证了单个任务多步数据不被穿插打断)
            if records:
                with open(args.output_file, "a", encoding="utf-8") as fw:
                    for r in records:
                        fw.write(json.dumps(r, ensure_ascii=False) + "\n")
            
            # 2. 顺序写入当前任务的评估结果
            if eval_record:
                with open(result_file, "a", encoding="utf-8") as fw:
                    fw.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                    
            global_prompt_tokens += p_tok
            global_completion_tokens += c_tok
            global_total_tokens += t_tok
            
            print(f"✅ 任务 [{task_id}/{number_to_process}] 处理并落盘完成！消耗 Token: {t_tok}")

    print("\n" + "="*40)
    print("✅ 所有并发任务处理完成！整体 Token 消耗统计：")
    print(f"🔹 提示词: {global_prompt_tokens}\n🔹 生成词: {global_completion_tokens}\n🔥 总计: {global_total_tokens}")
    print("="*40)