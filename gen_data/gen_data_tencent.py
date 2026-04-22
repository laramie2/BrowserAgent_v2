import json
import pandas as pd
from typing import List, Dict, Any
import requests
import time
import openai
from openai import OpenAI
import argparse
import re
import uuid
import base64
from io import BytesIO
from PIL import Image
import traceback


# 替换为新的 Venus API 配置
VENUS_URL = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
VENUS_TOKEN = "YOUR_VENUS_TOKEN_HERE" # 填入你的实际 Token

with open("/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    env_url = "http://localhost:5000/get_observation"
    extra_fields = [
        {"url": "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"}
        # {"url": "http://localhost:22015/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"}
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

# def get_response(prompt, model="gemini-2.5-pro", temperature=0.3, max_retries=5):
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
            # 增加 timeout 防止卡死
            response = requests.post(VENUS_URL, headers=headers, json=payload, timeout=60)
            
            # 替代原先的 openai.InternalServerError / RateLimitError 处理
            if response.status_code != 200:
                wait_time = (attempt + 1) * 5
                print(f"⚠️ 服务器拥塞或请求失败 (状态码: {response.status_code})，{wait_time}秒后重试... 报错: {response.text}")
                time.sleep(wait_time)
                continue
                
            resp_json = response.json()
            message = resp_json['choices'][0]['message']
            
            # 1. 安全提取 content (从字典取值)
            normal_content = message.get('content', "")
            if not normal_content:
                normal_content = ""
            
            # 2. 尝试提取 deepseek-reasoner 专属的 reasoning_content
            reasoning_content = message.get('reasoning_content', "")
                
            # 3. 按照你 SFT 数据集的格式要求重新拼装
            if reasoning_content:
                model_answer = f"<think>\n{reasoning_content}\n</think>\n{normal_content}"
            else:
                model_answer = normal_content
                
            # 4. 终极防空兜底
            if not model_answer.strip():
                print("⚠️ 警告：模型本次请求返回了完全空的内容，以占位符替换。")
                model_answer = "<action>stop [N/A]</action>"
                
            # 提取 Token 使用量
            usage = resp_json.get('usage', {})
            p_tokens = usage.get('prompt_tokens', 0)
            c_tokens = usage.get('completion_tokens', 0)
            t_tokens = usage.get('total_tokens', 0)
            
            return model_answer, p_tokens, c_tokens, t_tokens

        except requests.exceptions.RequestException as e:
            # 捕获网络层面的异常 (如超时、连接重置)
            wait_time = (attempt + 1) * 5
            print(f"⚠️ 网络请求异常，{wait_time}秒后重试... 错误: {e}")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ 遇到不可恢复错误: {e}")
            break
            
    return "ERROR_STILL_OCCURRED", 0, 0, 0

def extract_command(text):
    # 先尝试匹配新标签
    new_style = re.findall(r'<action>\s*(.*?)\s*</action>', text, re.DOTALL)
    if new_style:
        return new_style[-1].strip()
    
    # 如果没找到，尝试匹配旧的代码块格式
    old_style = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    if old_style:
        return old_style[-1].strip().replace("```","").strip()
    
    return " "

def extract_conclusion(text):
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)
    if not blocks: return " "
    return blocks[-1].strip()

def write_a_data(system_prompt_text, user_prompt_text, assistant_response, output_file, token_usage=None):
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
        
    with open(output_file, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(written_data, ensure_ascii=False) + "\n")

def Get_multi_turn_response(question, answer, output_file):
    tar_id = str(uuid.uuid4())
    history = "\n"
    history_info = "\n"
    obj = question
    obs = ""
    
    task_prompt_tokens = 0
    task_completion_tokens = 0
    task_total_tokens = 0

    try:
        jsoned_data = call_tool_server([tar_id], [''], [False])
        if 'observations' in jsoned_data:
            obs = jsoned_data['observations'][0]
        else:
            print(f"警告: 服务器未返回 observations。完整返回: {jsoned_data}")
            return task_prompt_tokens, task_completion_tokens, task_total_tokens
    except Exception as e:
        print(f"获取初始 observation 失败: {e}")
        return task_prompt_tokens, task_completion_tokens, task_total_tokens

    for i in range(10):
        # ==========================================
        #         Busy 状态拦截与轮询恢复机制
        # ==========================================
        wait_count = 0
        max_wait = 2  # 减少无意义的等待次数
        
        while re.search(r'RootWebArea.*?busy:\s*(1|True|true)', str(obs), re.IGNORECASE) and wait_count < max_wait:
            obs_str = str(obs)
            
            # 【核心优化】：判断页面是否已经有实质性内容 (比如超过 300 个字符)
            # 如果只是背景在转圈圈，但文字已经有了，我们就没必要死等
            if len(obs_str) > 300 and "RootWebArea" in obs_str:
                print("⚠️ 页面显示 busy，但检测到主体内容已加载，触发强制旁路 (Bypass)...")
                # 强行抹除 busy 状态，防止大模型看到 busy 后产生幻觉
                obs = re.sub(r'busy:\s*(1|True|true)', 'busy: False', obs_str, flags=re.IGNORECASE)
                break # 直接跳出等待，交给大模型处理
                
            print(f"⏳ 页面无内容且正忙，尝试发送无害动作刷新... ({wait_count+1}/{max_wait})")
            time.sleep(2)
            try:
                # 【核心优化】：不要发空字符串 ['']，有些环境会忽略空动作并返回缓存
                # 发送一个无害动作，比如原地的 scroll 或无效的 hover，强迫环境更新 DOM
                temp_action = "<think></think>\n<action>scroll [down]</action>"
                refresh_data = call_tool_server([tar_id], [temp_action], [False])
                
                # 紧接着再滚回来，抵消影响
                temp_action = "<think></think>\n<action>scroll [up]</action>"
                call_tool_server([tar_id], [temp_action], [False])
                
                if 'observations' in refresh_data:
                    obs = refresh_data['observations'][0]
            except Exception as e:
                print(f"刷新观测失败: {e}")
            wait_count += 1
            
        if re.search(r'RootWebArea.*?busy:\s*(1|True|true)', str(obs), re.IGNORECASE):
            print("❌ 页面彻底卡死且无内容，放弃当前轨迹。")
            call_tool_server([tar_id], [''], [True])  
            return task_prompt_tokens, task_completion_tokens, task_total_tokens
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
            print(e)
            
        current_step_tokens = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": t_tokens
        }
        
        write_a_data(
            system_prompt_text=system_prompt,
            user_prompt_text=real_prompt,
            assistant_response=response,
            output_file=output_file,
            token_usage=current_step_tokens
        )

        time.sleep(1.5)

        if "stop" in last_command:
            # ==========================================
            #           新增：答案提取与正确性判断
            # ==========================================
            match = re.search(r'stop\s*\[(.*?)\]', last_command, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
            else:
                final_answer = last_command.replace("stop", "").strip(" []")
            
            # 判断逻辑：ground_truth 是否包含在模型的最终回答中（忽略大小写）
            success = (str(answer).lower() in final_answer.lower()) if answer else False
            
            # 构建独立的结果指标记录
            eval_record = {
                "id": tar_id,
                "question": question,
                "ground_truth": answer,
                "final_answer": final_answer,
                "success": success
            }
            
            # 将同名后缀替换为 _results.jsonl 并写入
            result_file = output_file.replace(".jsonl", "_results.jsonl")
            if not result_file.endswith("_results.jsonl"):
                result_file += "_results.jsonl"  # 兜底：如果原文件没有.jsonl后缀
                
            with open(result_file, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            # ==========================================

            call_tool_server([tar_id], [response], [True])
            return task_prompt_tokens, task_completion_tokens, task_total_tokens
            
    return task_prompt_tokens, task_completion_tokens, task_total_tokens

number_to_process = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
    parser.add_argument('--output_file', type=str, default='./test.jsonl')
    parser.add_argument('--data_path', type=str, default='')
    args = parser.parse_args()
    print(f"Output file: {args.output_file}\nData file: {args.data_path}")

    data_df = pd.read_parquet(args.data_path).sample(frac=1, random_state=42).reset_index(drop=True)

    cnt = 0
    global_prompt_tokens = 0
    global_completion_tokens = 0
    global_total_tokens = 0

    for i, row in data_df.iterrows():
        question = row["extra_info"]["question"]
        gt = row["extra_info"]["selected_answer"]

        cnt += 1
        print(f"\n[{cnt}/{number_to_process}] 正在处理问题: {question[:50]}...")
        
        p_tok, c_tok, t_tok = Get_multi_turn_response(question, gt, args.output_file)
        
        global_prompt_tokens += p_tok
        global_completion_tokens += c_tok
        global_total_tokens += t_tok
        
        print(f"[*] 第 {cnt} 条任务消耗 Token -> 提示词: {p_tok}, 生成: {c_tok}, 总计: {t_tok}")
        print(f"[*] 当前全局累计消耗总 Token: {global_total_tokens}")

        if cnt == number_to_process:
            break

    print("\n" + "="*40)
    print("✅ 任务处理完成！整体 Token 消耗统计：")
    print(f"🔹 提示词: {global_prompt_tokens}\n🔹 生成词: {global_completion_tokens}\n🔥 总计: {global_total_tokens}")
    print("="*40)


"""
python -m gen_data.gen_data --data_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/benchmark/v2/nq/train-00000-of-00001.parquet
"""