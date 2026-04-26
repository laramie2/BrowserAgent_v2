import json
import os
import re
import argparse
import hashlib
import concurrent.futures
import functools
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from VTC_tool.VTC_tool import VTCTool

# ==========================================
# 全局变量与多进程初始化
# ==========================================
# 仅在子进程中初始化的 VTC 工具实例，避免多进程间的序列化冲突
_local_vtc = None

def init_worker():
    """每个工作进程独立初始化自己的 VTC_tool"""
    global _local_vtc
    _local_vtc = VTCTool()

# ==========================================
# 解析与渲染工具函数
# ==========================================
def parse_user_content(content: str) -> Dict[str, str]:
    """解析出原始 user content 中的四部分：Objective, Observation, HISTORY_ACTION, HISTORY_info"""
    result = {
        "objective": "",
        "observation": "",
        "history_action": "",
        "history_info": ""
    }
    
    obj_match = re.search(r'Objective:\s*(.*?)\nObservation:', content, re.DOTALL)
    obs_match = re.search(r'Observation:\s*(.*?)\nHISTORY_ACTION:', content, re.DOTALL)
    ha_match = re.search(r'HISTORY_ACTION:\s*(.*?)\nHISTORY_info:', content, re.DOTALL)
    hi_match = re.search(r'HISTORY_info:\s*(.*)', content, re.DOTALL)

    if obj_match: result["objective"] = obj_match.group(1).strip()
    if obs_match: result["observation"] = obs_match.group(1).strip()
    if ha_match: result["history_action"] = ha_match.group(1).strip()
    if hi_match: result["history_info"] = hi_match.group(1).strip()
    
    return result

def generate_image_for_observation(ob_text: str, output_dir: str, is_simple: bool = False) -> str:
    """使用哈希值作为文件名，避免重复渲染相同的 Observation"""
    global _local_vtc
    
    # 核心优化 1：计算文本的 MD5 哈希值
    text_hash = hashlib.md5(ob_text.encode('utf-8')).hexdigest()
    img_filename = f"obs_{text_hash}.png"
    img_path = os.path.join(output_dir, img_filename)
    
    # 如果图片已存在，直接返回路径（大幅节省渲染时间）
    if not os.path.exists(img_path):
        os.makedirs(output_dir, exist_ok=True)
        if not is_simple:
            img, char_count = _local_vtc.render_text_to_image(
                ob_text, 
                use_compact_mode=True, 
                max_width=2048, 
                max_height=2048
            )
        else:
            img, char_count = _local_vtc.render_text_to_image_simple(
                ob_text, 
                width=1024,
                aspect_ratio="1:1"
            )

        img.save(img_path)
        
    return img_path

# ==========================================
# 数据处理核心逻辑
# ==========================================
def task_generator(input_file: str, local_system_msg: str = None):
    """核心优化 2：流式读取数据，避免将超大文件一次性载入内存"""
    with open(input_file, 'r', encoding='utf-8') as f:
        current_task = []
        current_objective = None
        
        for line in f:
            item = json.loads(line)
            system_msg = next((m for m in item['messages'] if m['role'] == 'system'), None) if local_system_msg is None else {"role": "system", "content": local_system_msg}
            user_msg = next((m for m in item['messages'] if m['role'] == 'user'), None)
            assistant_msg = next((m for m in item['messages'] if m['role'] == 'assistant'), None)
            
            parsed_user = parse_user_content(user_msg['content'])
            obj = parsed_user['objective']
            
            if current_objective is None or obj != current_objective:
                if current_task:
                    yield current_task
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
            yield current_task

def process_single_task(task_steps: List[Dict], image_output_dir: str, level: str, format_type: str, is_simple: bool = False) -> Tuple[List[Dict], int]:
    """核心优化 3：将单个任务的处理逻辑纯粹化，以便分配给多进程执行"""
    task_images = []
    task_messages = []
    
    subset = task_steps[0]['subset'] + "_vision"
    stage = task_steps[0]['stage']
    system_content = task_steps[0]['system']
    
    if level == "task":
        task_messages.append({"role": "system", "content": system_content})
        
    step_outputs = []
    
    for step in task_steps:
        parsed = step['parsed_user']
        
        # 调用基于哈希的渲染函数
        img_obs_path = generate_image_for_observation(parsed['observation'], image_output_dir, is_simple=is_simple)
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
            step_outputs.append(step_data)
            
        elif level == "task":
            task_messages.append({"role": "user", "content": user_content})
            task_messages.append({"role": "assistant", "content": step['assistant']})
            
    if level == "task":
        task_data = {
            "messages": task_messages,
            "subset": subset,
            "stage": stage
        }
        if format_type == "opensource":
            task_data["images"] = task_images
        return [task_data], len(task_steps)
    else:
        return step_outputs, len(task_steps)

def convert_dataset(input_file: str, output_file: str, image_output_dir: str, 
                    level: str, format_type: str, max_workers: int = None, local_system_msg: str = None, is_simple: bool = False):
    print("Initializing task generator...")
    # 获取任务迭代器
    tasks = task_generator(input_file, local_system_msg)
    
    stats = {
        "total_converted_tasks": 0,
        "total_converted_steps": 0,
        "original_steps_count": 0,
        "steps_per_task": []
    }

    # 提前打开输出文件（核心优化 4：移除 flush，依靠系统级缓存进行高效 I/O）
    with open(output_file, 'w', encoding='utf-8') as fw:
        # 使用多进程池进行并行处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            # 偏函数绑定固定参数
            process_func = functools.partial(
                process_single_task, 
                image_output_dir=image_output_dir, 
                level=level, 
                format_type=format_type,
                is_simple=is_simple
            )
            
            print(f"Starting multiprocessing conversion (Workers: {executor._max_workers})...")
            # executor.map 会保证产出的结果顺序与输入数据顺序一致
            for data_to_write, num_steps in tqdm(executor.map(process_func, tasks), desc="Processing Tasks"):
                
                # 写入结果
                for item in data_to_write:
                    fw.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                # 统计数据
                stats["original_steps_count"] += num_steps
                stats["total_converted_tasks"] += 1
                stats["steps_per_task"].append(num_steps)
                if level == "step":
                    stats["total_converted_steps"] += len(data_to_write)

    print_statistics(stats, level == "task")
    print(f"\n[Success] Converted dataset saved to {output_file}")


# ==========================================
# 统计功能函数
# ==========================================
def print_statistics(stats: Dict[str, Any], is_task_level: bool):
    print("\n" + "="*30)
    print("      Dataset Statistics")
    print("="*30)
    print(f"Total Original Steps: {stats['original_steps_count']}")
    
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
    parser = argparse.ArgumentParser(description="Convert Text-based Web Agent Dataset to Vision-based Dataset (Parallel & Streaming)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to raw jsonl dataset")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output jsonl dataset")
    parser.add_argument("--img_dir", type=str, default="./dataset/images", help="Directory to save generated images")
    parser.add_argument("--level", type=str, choices=["step", "task"], default="task", help="Data granularity: 'step' or 'task'")
    parser.add_argument("--format", type=str, choices=["openai", "opensource"], default="opensource", help="Vision data format")
    # 可以手动指定使用的进程数，默认会自动根据系统 CPU 核心数分配
    parser.add_argument("--workers", type=int, default=None, help="Number of multi-processing workers")
    parser.add_argument("--system_msg_path", type=str, default=None, help="Optional fixed system message to override original system prompts")
    parser.add_argument("--is_simple", type=bool, default=False, help="Whether to use simplified VTC rendering ")

    args = parser.parse_args()

    local_system_msg = None
    
    if args.system_msg_path:
        with open(args.system_msg_path, 'r', encoding='utf-8') as f:
            local_system_msg = f.read().strip()

    print(f"Starting conversion...\nMode: {args.level}-level | Format: {args.format}")
    convert_dataset(
        input_file=args.input,
        output_file=args.output,
        image_output_dir=args.img_dir,
        level=args.level,
        format_type=args.format,
        max_workers=args.workers,
        local_system_msg=local_system_msg,  # 可以在这里传入一个固定的 system prompt 来覆盖原数据中的 system 信息
        is_simple=args.is_simple  # 是否使用简化版 VTC 渲染逻辑
    )



"""
python -m gen_seq.VTC_seq_para \
    --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft.jsonl \
    --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc-enhanced_format_without_content.jsonl \
    --img_dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc-enhanced_format_without_content_images \
    --level task \
    --format opensource \
    --system_msg_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance_without_content.txt \
    --is_simple False

python -m gen_seq.VTC_seq_para \
    --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft.jsonl \
    --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc-simplified-2.jsonl \
    --img_dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc-simplified-2_images \
    --level task \
    --format opensource \
    --system_msg_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance.txt \
    --is_simple True

python -m gen_seq.VTC_seq_para \
    --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_new_add2720.jsonl \
    --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc-new_add2720.jsonl \
    --img_dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_task-opsrc-new_add2720_images \
    --level task \
    --format opensource \
    --system_msg_path /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/prompt/system_prompt_with_history_info_enhance_yt_and_action.txt
    
python -m gen_seq.VTC_seq_para \
    --input /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft.jsonl \
    --output /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_step-opsrc-simplified.jsonl \
    --img_dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/data/v2/sft_step-opsrc-simplified_images \
    --level step \
    --format opensource
"""