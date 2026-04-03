import json
import os
import re
import argparse
import uuid
from typing import List, Dict, Any
from tqdm import tqdm

from VTC_tool.VTC_tool import VTCTool

# ==========================================
# 解析工具函数
# ==========================================
def parse_user_content(content: str) -> Dict[str, str]:
    """解析出原始 user content 中的四部分：Objective, Observation, HISTORY_ACTION, HISTORY_info"""
    result = {
        "objective": "",
        "observation": "",
        "history_action": "",
        "history_info": ""
    }
    
    # 使用正则匹配截取各部分内容
    obj_match = re.search(r'Objective:\s*(.*?)\nObservation:', content, re.DOTALL)
    obs_match = re.search(r'Observation:\s*(.*?)\nHISTORY_ACTION:', content, re.DOTALL)
    ha_match = re.search(r'HISTORY_ACTION:\s*(.*?)\nHISTORY_info:', content, re.DOTALL)
    hi_match = re.search(r'HISTORY_info:\s*(.*)', content, re.DOTALL)

    if obj_match: result["objective"] = obj_match.group(1).strip()
    if obs_match: result["observation"] = obs_match.group(1).strip()
    if ha_match: result["history_action"] = ha_match.group(1).strip()
    if hi_match: result["history_info"] = hi_match.group(1).strip()
    
    return result

def generate_image_for_observation(vtc_tool: VTCTool, ob_text: str, output_dir: str, step_id: str) -> str:
    """调用 VTC 渲染文字为图像并保存，返回相对路径"""
    img, char_count = vtc_tool.render_text_to_image(
        ob_text, 
        use_compact_mode=True, 
        max_width=2048, 
        max_height=4096
    )
    os.makedirs(output_dir, exist_ok=True)
    img_filename = f"obs_{step_id}.png"
    img_path = os.path.join(output_dir, img_filename)
    img.save(img_path)
    return img_path

# ==========================================
# 核心转换逻辑
# ==========================================
def convert_dataset(input_file: str, output_file: str, image_output_dir: str, 
                    level: str, format_type: str):
    vtc = VTCTool()
    
    # 1. 读取原始数据并按任务分组
    print("Reading and grouping raw data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
        
    tasks = []
    current_task = []
    current_objective = None
    
    for item in raw_data:
        system_msg = next((m for m in item['messages'] if m['role'] == 'system'), None)
        user_msg = next((m for m in item['messages'] if m['role'] == 'user'), None)
        assistant_msg = next((m for m in item['messages'] if m['role'] == 'assistant'), None)
        
        parsed_user = parse_user_content(user_msg['content'])
        obj = parsed_user['objective']
        
        # 判断是否开启了新任务（根据 Objective 是否变化来判断，或者遇到空白的历史）
        if current_objective is None or obj != current_objective:
            if current_task:
                tasks.append(current_task)
            current_task = []
            current_objective = obj
            
        current_task.append({
            "system": system_msg['content'] if system_msg else "",
            "parsed_user": parsed_user,
            "assistant": assistant_msg['content'] if assistant_msg else "",
            "subset": item.get('subset', 'vision_dataset'),
            "stage": item.get('stage', 'sft')
        })
        
    if current_task:
        tasks.append(current_task)

    # 2. 流式处理与写入
    print(f"Converting and streaming {len(tasks)} tasks...")
    
    # 准备流式统计指标
    stats = {
        "total_converted_tasks": 0,
        "total_converted_steps": 0,
        "steps_per_task": []
    }
    
    # 提前打开输出文件
    with open(output_file, 'w', encoding='utf-8') as fw:
        for task_idx, task_steps in enumerate(tqdm(tasks, desc="Processing Tasks", unit="task")):
            task_images = []
            task_messages = []
            
            # 提取公共属性
            subset = task_steps[0]['subset'] + "_vision"
            stage = task_steps[0]['stage']
            system_content = task_steps[0]['system']
            
            if level == "task":
                task_messages.append({"role": "system", "content": system_content})
                
            for step_idx, step in enumerate(task_steps):
                step_id = f"task{task_idx}_step{step_idx}_{uuid.uuid4().hex[:6]}"
                parsed = step['parsed_user']
                
                # 渲染图像
                img_obs_path = generate_image_for_observation(vtc, parsed['observation'], image_output_dir, step_id)
                img_filename = os.path.basename(img_obs_path)
                img_path = os.path.join("images", img_filename)  # 相对路径
                
                task_images.append(img_path)
                
                # 构建 User 文本结构
                base_text = f"Objective: {parsed['objective']}\n"
                
                if format_type == "openai":
                    base_text += "Observation: Please refer to the provided webpage screenshot for the current UI state.\n"
                    if parsed['history_action']: base_text += f"HISTORY_ACTION: {parsed['history_action']}\n"
                    if parsed['history_info']: base_text += f"HISTORY_info: {parsed['history_info']}\n"
                    
                    user_content = [
                        {"type": "text", "text": base_text},
                        {"type": "image_url", "image_url": {"url": img_path}}
                    ]
                else: # opensource
                    base_text += "Observation: <image>\n"
                    if parsed['history_action']: base_text += f"HISTORY_ACTION: {parsed['history_action']}\n"
                    if parsed['history_info']: base_text += f"HISTORY_info: {parsed['history_info']}\n"
                    
                    user_content = base_text

                if level == "step":
                    # 单步级别：完成一步立即写入
                    step_data = {
                        "messages": [
                            {"role": "system", "content": step['system']},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": step['assistant']}
                        ],
                        "subset": subset,
                        "stage": stage
                    }
                    if format_type == "opensource":
                        step_data["images"] = [img_path]
                    
                    # 写入单条 Step 数据并更新统计
                    fw.write(json.dumps(step_data, ensure_ascii=False) + '\n')
                    stats["total_converted_steps"] += 1
                    
                elif level == "task":
                    # 任务级别：累加到 messages 列表中
                    task_messages.append({"role": "user", "content": user_content})
                    task_messages.append({"role": "assistant", "content": step['assistant']})
                    
            if level == "task":
                # 任务级别：完成一整个任务后立即写入
                task_data = {
                    "messages": task_messages,
                    "subset": subset,
                    "stage": stage
                }
                if format_type == "opensource":
                    task_data["images"] = task_images
                
                # 写入单条 Task 数据并更新统计
                fw.write(json.dumps(task_data, ensure_ascii=False) + '\n')
                stats["total_converted_tasks"] += 1
                user_turns = len([m for m in task_messages if m['role'] == 'user'])
                stats["steps_per_task"].append(user_turns)
            
            # 强制刷新缓冲区，确保实时落盘
            fw.flush()

    # 3. 打印统计数据
    print_statistics(len(raw_data), stats, level == "task")
    print(f"\n[Success] Converted dataset saved to {output_file}")


# ==========================================
# 统计功能函数（修改以适配流式统计变量）
# ==========================================
def print_statistics(original_steps_count: int, stats: Dict[str, Any], is_task_level: bool):
    print("\n" + "="*30)
    print("      Dataset Statistics")
    print("="*30)
    print(f"Total Original Steps: {original_steps_count}")
    
    if is_task_level:
        print(f"Total Converted Tasks: {stats['total_converted_tasks']}")
        steps_per_task = stats['steps_per_task']
        avg_steps = sum(steps_per_task) / len(steps_per_task) if steps_per_task else 0
        print(f"Avg Steps per Task: {avg_steps:.2f}")
        print(f"Max Steps in a Task: {max(steps_per_task) if steps_per_task else 0}")
        print(f"Min Steps in a Task: {min(steps_per_task) if steps_per_task else 0}")
    else:
        print(f"Total Converted Steps: {stats['total_converted_steps']}")
    print("="*30)

# ==========================================
# 命令行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Text-based Web Agent Dataset to Vision-based Dataset (Streaming Write)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to raw jsonl dataset")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output jsonl dataset")
    parser.add_argument("--img_dir", type=str, default="./dataset/images", help="Directory to save generated images")
    parser.add_argument("--level", type=str, choices=["step", "task"], default="task", help="Data granularity: 'step' or 'task'")
    parser.add_argument("--format", type=str, choices=["openai", "opensource"], default="opensource", help="Vision data format")
    
    args = parser.parse_args()
    
    print(f"Starting conversion...\nMode: {args.level}-level | Format: {args.format}")
    convert_dataset(
        input_file=args.input,
        output_file=args.output,
        image_output_dir=args.img_dir,
        level=args.level,
        format_type=args.format
    )


"""
python -m gen_seq.VTC_seq \
    --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft.jsonl \
    --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc.jsonl \
    --img_dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc_images \
    --level task \
    --format opensource

python -m gen_seq.VTC_seq \
    --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft.jsonl \
    --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_step-opsrc.jsonl \
    --img_dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_step-opsrc_images \
    --level step \
    --format opensource
"""