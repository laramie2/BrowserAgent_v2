import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import requests
import time
import openai
from openai import OpenAI
import argparse
import re
import uuid
import os
import io
import base64
from VTC_tool.VTC_tool import VTCTool

api_key = "sk-5TOLjHJSn7uyRj2gXZLxYsRe9vxmr8N9XWK2lQHalvgXiBoc"
base_url = "https://open.xiaojingai.com/v1/"

client = OpenAI(api_key=api_key, base_url=base_url)

with open("/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

def generate_image_for_observation(vtc_tool: Any, ob_text: str, images_dir: str, step_id: str, compression_factor: float = 2.0) -> Tuple[str, str, float]:
    """调用 VTC 渲染文字为图像，压缩后在内存中转换为 Base64，同时保存图片，返回 (Base64, 相对路径, 耗时)"""
    start_render = time.perf_counter()
    img, char_count = vtc_tool.render_text_to_image_simple(
        ob_text, 
        width=1024,
        aspect_ratio="4:3"
    )
    
    # 嵌入图像压缩逻辑：如果 compression_factor > 1.0，则进行等比例缩放
    if compression_factor > 1.0:
        # compress_image_arrays 接收并返回 List，因此需要加 [0] 提取单张图片
        img = vtc_tool.compress_image_arrays([img], compression_factor=compression_factor)[0]

    render_time = time.perf_counter() - start_render

    # 内存中转 Base64 (供大模型推理用)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 物理保存图片 (落盘到 images 文件夹，保存压缩后的图像以对齐模型实际输入)
    img_filename = f"obs_{step_id}.png"
    img_path = os.path.join(images_dir, img_filename)
    img.save(img_path)
    
    # 构建相对路径供 JSONL 保存 (格式: images/obs_xxx.png)
    rel_img_path = f"images/{img_filename}"
    
    return img_base64, rel_img_path, render_time

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
HISTORY_ACTION: {}
HISTORY_info: {}

[Please refer to the attached image for the current environment observation.]
"""

def get_response(user_content: List[Dict], model: str = "gemini-2.5-pro", temperature: float = 0.3, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=1024,
                timeout=120
            )
            
            message = response.choices[0].message
            
            normal_content = message.content if message.content else ""
            reasoning_content = getattr(message, 'reasoning_content', "")
            if not reasoning_content:
                reasoning_content = ""
                
            if reasoning_content:
                model_answer = f"<think>\n{reasoning_content}\n</think>\n{normal_content}"
            else:
                model_answer = normal_content
                
            if not model_answer.strip():
                print("⚠️ 警告：模型本次请求返回了完全空的内容，以占位符替换。")
                model_answer = "<action>stop [N/A]</action>"
                
            usage = response.usage
            p_tokens = usage.prompt_tokens if usage else 0
            c_tokens = usage.completion_tokens if usage else 0
            t_tokens = usage.total_tokens if usage else 0
            
            return model_answer, p_tokens, c_tokens, t_tokens

        except (openai.InternalServerError, openai.RateLimitError) as e:
            wait_time = (attempt + 1) * 5
            print(f"⚠️ 服务器拥塞 (503/RateLimit)，{wait_time}秒后重试... 错误: {e}")
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

def write_a_data(system_prompt_text, save_user_content, assistant_response, output_file, token_usage=None):
    written_data = {
        "messages": [
            {"role": "system", "content": system_prompt_text.strip()},
            {"role": "user", "content": save_user_content},
            {"role": "assistant", "content": assistant_response}
        ],
        "subset": "corr_hotpot_new1369q_swift",
        "stage": "sft"
    }
    if token_usage:
        written_data["token_usage"] = token_usage
        
    with open(output_file, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(written_data, ensure_ascii=False) + "\n")

def Get_multi_turn_response(question, answer, output_file, images_dir, vtc_tool):
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
        wait_count = 0
        max_wait = 2 
        
        # 兜底转换保证 str(obs) 安全运行
        obs_str = str(obs) if obs is not None else ""
        
        while re.search(r'RootWebArea.*?busy:\s*(1|True|true)', obs_str, re.IGNORECASE) and wait_count < max_wait:
            if len(obs_str) > 300 and "RootWebArea" in obs_str:
                print("⚠️ 页面显示 busy，但检测到主体内容已加载，触发强制旁路 (Bypass)...")
                obs = re.sub(r'busy:\s*(1|True|true)', 'busy: False', obs_str, flags=re.IGNORECASE)
                obs_str = obs # 同步更新
                break 
                
            print(f"⏳ 页面无内容且正忙，尝试发送无害动作刷新... ({wait_count+1}/{max_wait})")
            time.sleep(2)
            try:
                temp_action = "<think></think>\n<action>scroll [down]</action>"
                refresh_data = call_tool_server([tar_id], [temp_action], [False])
                
                temp_action = "<think></think>\n<action>scroll [up]</action>"
                call_tool_server([tar_id], [temp_action], [False])
                
                if 'observations' in refresh_data:
                    obs = refresh_data['observations'][0]
                    obs_str = str(obs) if obs is not None else ""
            except Exception as e:
                print(f"刷新观测失败: {e}")
            wait_count += 1
            
        if re.search(r'RootWebArea.*?busy:\s*(1|True|true)', obs_str, re.IGNORECASE):
            print("❌ 页面彻底卡死且无内容，放弃当前轨迹。")
            call_tool_server([tar_id], [''], [True])  
            return task_prompt_tokens, task_completion_tokens, task_total_tokens

        # 使用处理好的字符串提取 clean_obs，杜绝正则类型报错
        try:
            clean_obs = obs_str.split('Observation:\n')[1].split('\nParsed Previous Action:')[0]
        except:
            clean_obs = obs_str
            
        step_id = f"{tar_id}_step{i}"
        
        img_base64, rel_img_path, _ = generate_image_for_observation(vtc_tool, clean_obs, images_dir, step_id)
        
        real_prompt_text = user_prompt.format(obj, history, history_info)
        
        # ==========================================
        # 1. 组装给 API 的 Content 
        # (包含 Base64，且不包含自定义 type 防止 API 报错)
        # ==========================================
        api_user_content = [
            {"type": "text", "text": real_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
        
        # ==========================================
        # 2. 组装存入 JSONL 的 Content 
        # (轻量化相对路径，并额外加入 original_text 字典供后续研究)
        # ==========================================
        save_user_content = [
            {"type": "text", "text": real_prompt_text},
            {"type": "image_url", "image_url": {"url": rel_img_path}},
            {"type": "original_text", "original_text": clean_obs}
        ]
        
        # 将合法的 api_user_content 发给大模型
        response, p_tokens, c_tokens, t_tokens = get_response(
            user_content=api_user_content,
            temperature=0.3
        )
        
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
        
        # 将携带 original_text 的 save_user_content 存入本地文件
        write_a_data(
            system_prompt_text=system_prompt,
            save_user_content=save_user_content,
            assistant_response=response,
            output_file=output_file,
            token_usage=current_step_tokens
        )

        time.sleep(1.5)

        if "stop" in last_command:
            call_tool_server([tar_id], [response], [True])
            return task_prompt_tokens, task_completion_tokens, task_total_tokens
            
    return task_prompt_tokens, task_completion_tokens, task_total_tokens

number_to_process = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
    parser.add_argument('--output_file', type=str, default='./test.jsonl')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--use_vlm', action='store_true', default=True, help="Whether to use Visual Language Model")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(args.output_file))
    images_dir = os.path.join(base_dir, "images_compress")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Output file: {args.output_file}")
    print(f"Images will be saved to: {images_dir}")
    print(f"Data file: {args.data_path}")

    vtc_tool = VTCTool() if args.use_vlm else None

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
        
        p_tok, c_tok, t_tok = Get_multi_turn_response(question, gt, args.output_file, images_dir, vtc_tool)
        
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
python -m gen_data.gen_data_vl \
    --output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/test_vl_compress.jsonl \
    --data_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_data/sft_seed/sft-hotpot2500-nq2500-seed.parquet \
    --use_vlm
"""