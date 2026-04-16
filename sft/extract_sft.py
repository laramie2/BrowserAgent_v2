import re
import os
import json
import argparse
import shutil
import random
from pathlib import Path

def print_summary(filepath, mode, tasks, steps, is_analysis_only=False, copied_imgs=None, missing_imgs=None, img_refs=None):
    """打印详细的统计面板"""
    print("\n" + "="*45)
    title = "📊 输出文件详细信息 (-v / 独立分析)" if is_analysis_only else "📊 提取完成详细信息 (-v)"
    print(title)
    print("="*45)
    print(f"📄 文件路径: {filepath}")
    print(f"🎯 数据模式: {'步骤级 (Step-level)' if mode == 'step' else '任务级 (Task-level)'}")
    print(f"📦 任务数 (行数): {tasks}")
    print(f"👣 步骤数: {steps}")
    
    # 纯分析模式下只统计引用的图片数
    if is_analysis_only and img_refs is not None:
        print("-" * 45)
        print(f"🖼️  文件中引用的图片总数: {img_refs} 张")
    
    # 提取模式下统计复制情况
    if not is_analysis_only and copied_imgs is not None:
        print("-" * 45)
        print(f"🖼️  成功复制图片: {copied_imgs} 张")
        if missing_imgs > 0:
            print(f"⚠️  缺失图片: {missing_imgs} 张")
    print("="*45 + "\n")

def analyze_jsonl(filepath, mode):
    """独立扫描和分析已有的 JSONL 文件，给出详细信息"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"要分析的文件不存在: {filepath}")
        
    tasks = 0
    steps = 0
    img_refs = 0
    
    print(f"正在分析文件: {filepath} ...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                tasks += 1
                
                # 计算步骤数和图片引用数
                if mode == 'step':
                    steps += 1
                    if "images" in data:
                        img_refs += len(data["images"])
                else:  # mode == 'task'
                    if "images" in data:
                        steps += len(data["images"])
                        img_refs += len(data["images"])
                    else:
                        st = sum(1 for m in data.get('messages', []) if m.get('role') == 'user')
                        steps += max(1, st)
                        
            except json.JSONDecodeError:
                pass
                
    print_summary(filepath, mode, tasks, steps, is_analysis_only=True, img_refs=img_refs)

def extract_jsonl_lines(input_file, output_file, mode, num_tasks=None, num_steps=None, 
                        src_img_dir=None, dst_img_dir=None, is_random=False, seed=None, verbose=False):
    """
    从JSONL文件中提取数据
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 图片目录校验
    copy_images = False
    if src_img_dir and dst_img_dir:
        src_img_path = Path(src_img_dir)
        dst_img_path = Path(dst_img_dir)
        if not src_img_path.exists():
            raise FileNotFoundError(f"源图片目录不存在: {src_img_dir}")
        copy_images = True
    elif src_img_dir or dst_img_dir:
        raise ValueError("src_img_dir 和 dst_img_dir 必须同时提供才能进行图片提取")
    
    if is_random and seed is not None:
        random.seed(seed)
        
    if verbose:
        print("第一阶段：正在扫描数据并建立轻量级索引...")
        
    valid_indices = []
    
    with open(input_path, 'rb') as f:
        while True:
            offset = f.tell()
            line_bytes = f.readline()
            if not line_bytes:
                break
                
            line_str = line_bytes.decode('utf-8').strip()
            if not line_str:
                continue
                
            try:
                data = json.loads(line_str)
                if mode == 'step':
                    steps = 1
                else:
                    if "images" in data:
                        steps = len(data["images"])
                    else:
                        steps = sum(1 for m in data.get('messages', []) if m.get('role') == 'user')
                        steps = max(1, steps)
                        
                valid_indices.append((offset, steps))
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"警告: 偏移量 {offset} 处的行不是有效的JSON格式，已跳过。")
                continue

    iter_indices = valid_indices.copy()
    if is_random:
        random.shuffle(iter_indices)

    selected_items = []
    current_tasks = 0
    current_steps = 0
    
    for offset, steps in iter_indices:
        if num_tasks is not None and current_tasks >= num_tasks:
            break
        if num_steps is not None and current_steps >= num_steps:
            break
            
        selected_items.append((offset, steps))
        current_tasks += 1
        current_steps += steps

    if is_random:
        selected_items.sort(key=lambda x: x[0])
        
    if verbose:
        print(f"索引建立完毕。计划提取 {current_tasks} 个任务(行)，共计 {current_steps} 个步骤。")
        print("第二阶段：开始写入数据并处理图片...")
    
    extracted_tasks = 0
    extracted_steps = 0
    copied_img_count = 0
    missing_img_count = 0
    
    with open(input_path, 'rb') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for offset, steps in selected_items:
            infile.seek(offset)
            line_str = infile.readline().decode('utf-8').strip()
            data = json.loads(line_str)
            
            if copy_images and "images" in data:
                for img_rel_path in data["images"]:
                    src_file = src_img_path / img_rel_path
                    dst_file = dst_img_path / img_rel_path
                    
                    if src_file.exists():
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        copied_img_count += 1
                    else:
                        if verbose:
                            print(f"警告: 任务(行)提取中发现图片未找到 -> {src_file}")
                        missing_img_count += 1
            
            outfile.write(line_str + '\n')
            extracted_tasks += 1
            extracted_steps += steps
            
    if verbose:
        # 调用解耦出来的格式化打印函数
        print_summary(output_file, mode, extracted_tasks, extracted_steps, 
                      is_analysis_only=False, 
                      copied_imgs=copied_img_count if copy_images else None, 
                      missing_imgs=missing_img_count if copy_images else None)
    else:
        print(f"✅ 成功提取 {extracted_tasks} 行数据到 {output_file}")
            
    return extracted_tasks

def extract_objectives(input_file, output_json_file):
    """
    从JSONL文件中提取user message中的Objective内容，去重后写入JSON文件。
    """
    unique_objectives = set()
    # 使用正则匹配 "Objective: " 和 "\nObservation:" 之间的内容
    objective_pattern = re.compile(r"Objective:\s*(.*?)\nObservation:", re.DOTALL)
    
    print(f"正在从 {input_file} 提取 Objectives...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 遍历 messages 寻找 role 为 user 的节点
                    for msg in data.get("messages", []):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            match = objective_pattern.search(content)
                            if match:
                                objective = match.group(1).strip()
                                unique_objectives.add(objective)
                except json.JSONDecodeError:
                    print(f"警告：跳过第 {line_num} 行，JSON格式解析错误。")
                    continue
                    
        # 将去重后的集合转换为列表并写入JSON
        with open(output_json_file, 'w', encoding='utf-8') as out_f:
            json.dump(list(unique_objectives), out_f, ensure_ascii=False, indent=4)
            
        print(f"提取完成！共提取了 {len(unique_objectives)} 个不重复的 Objective，已保存至 {output_json_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except Exception as e:
        print(f"提取过程中发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='从JSONL文件中提取数据，或分析已有文件的详细信息')
    
    # input_file 改为可选参数
    parser.add_argument('--input_file', type=str, help='输入JSONL文件路径 (提取时必填，仅分析输出文件时可省略)')
    
    # 将 output_file 的 required 改为 False，以兼容独立运行 Objective 提取
    parser.add_argument('--output_file', type=str, help='输出JSONL文件路径 (提取的目标文件，或分析的目标文件，或提取问题的目标文件)')
    
    parser.add_argument('--mode', choices=['step', 'task'], default='step', 
                        help='数据层级: step(步骤级，每行当做1步计算) 或 task(任务级，基于内容计算步骤数)')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--num-steps', type=int, help='要提取的总步骤数')
    group.add_argument('--num-tasks', type=int, help='要提取的总任务数 (仅在 --mode task 下可用)')
    
    parser.add_argument('--src-img-dir', type=str, help='源图片根目录')
    parser.add_argument('--dst-img-dir', type=str, help='目标图片根目录')
    
    parser.add_argument('-r', '--random', action='store_true', help='启用随机抽样')
    parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子，固定随机抽样的结果')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示输出文件的详细统计信息')
    
    parser.add_argument('--extract_obj', action='store_true', help='开启Objective提取模式 (需结合 --output_file 和 --obj_output_file)')
    parser.add_argument('--obj_output_file', type=str, help='提取Objective后保存的JSON文件路径')
    # ====================================================

    args = parser.parse_args()
    
    try:
        # ================= 如果是提取 Objective 模式 =================
        if args.extract_obj:
            if not args.output_file or not args.obj_output_file:
                raise ValueError("开启 --extract_obj 时，必须同时提供 --output_file (读取源) 和 --obj_output_file (保存JSON的位置)")
            
            extract_objectives(args.output_file, args.obj_output_file)
            return 0  # 执行完毕后直接退出，避免走入原本的流程
        # =========================================================================

        # 在非 extract_obj 模式下，强制要求 output_file
        if not args.output_file:
            raise ValueError("常规模式下，必须提供 --output_file 参数")

        # 分支 1：如果是提取模式 (提供了 input_file)
        if args.input_file:
            # 校验提取参数
            if args.mode == 'step' and args.num_tasks is not None:
                raise ValueError("在 步骤级(step) 模式下，请使用 --num-steps 指定提取数量，不能使用 --num-tasks")
            if args.num_tasks is None and args.num_steps is None:
                raise ValueError("提取模式下，必须指定 --num-tasks 或 --num-steps 中的一个")
                
            extract_jsonl_lines(
                input_file=args.input_file,
                output_file=args.output_file,
                mode=args.mode,
                num_tasks=args.num_tasks,
                num_steps=args.num_steps,
                src_img_dir=args.src_img_dir,
                dst_img_dir=args.dst_img_dir,
                is_random=args.random,
                seed=args.seed,
                verbose=args.verbose
            )
            
        # 分支 2：如果是纯分析模式 (没有提供 input_file)
        else:
            if not args.verbose:
                raise ValueError("未指定 --input_file 进行提取时，必须使用 -v 参数来分析指定的 --output_file")
            
            # 直接调用分析逻辑
            analyze_jsonl(args.output_file, args.mode)
            
    except Exception as e:
        print(f"程序出错终止: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()


"""
终端使用示例
1. 任务级模式，随机抽取 5000 个完整的任务（行），并附带提取对应的多张图片
python extract_sft.py \
    --input_file data.jsonl --output_file small.jsonl \
    --mode task --num-tasks 500 \
    --src-img-dir /data/img --dst-img-dir ./small_img -r

python extract_sft.py \
    --input_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/data.jsonl \
    --output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc-5000stp/data.jsonl \
    --src-img-dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/ \
    --dst-img-dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc-5000stp/ \
    --mode task --num-task 5000 \
    -r -s 42 -v

2.任务级模式，不断随机抽取完整任务，直到累计的“步骤数量”达到 1000 步
(例如有些任务含有 4 个步骤，有些含有 2 个，脚本会自动累加计算直到触碰 1000 步阈值)
python extract_sft.py \
    --input_file data.jsonl --output_file small.jsonl \
    --mode task --num-steps 1000 \
    --src-img-dir /data/img --dst-img-dir ./small_img -r

python extract_sft.py \
    --input_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/data.jsonl \
    --output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc-2500stp/data.jsonl \
    --src-img-dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/ \
    --dst-img-dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc-2500stp/ \
    --mode task --num-steps 2500 \
    -r -s 42 -v

3.步骤级模式，从步骤数据中随机抽取一定数量的数据
python extract_sft.py \
    --input_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/step-opsrc/data.jsonl \
    --output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/step-opsrc-2500/data.jsonl \
    --src-img-dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/step-opsrc/ \
    --dst-img-dir /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/step-opsrc-2500/ \
    --mode step --num-steps 2500 \
    -r -s 42

3.退回到原本的步骤级模式（1 行就是 1 步），顺序提取前 100 行
python extract_sft.py \
    --input_file data.jsonl --output_file small.jsonl \
    --mode step --num-steps 100

4.查看详细数据    
python extract_sft.py --output_file out.jsonl --mode task -v

python extract_sft.py --output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/data.jsonl --mode step -v

5.提取问题
python extract_sft.py \
    --extract_obj \
    --output_file out.jsonl \
    --obj_output_file obj.json

python extract_sft.py \
    --extract_obj \
    --output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/data.jsonl \
    --obj_output_file /DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/sft/dataset/task-opsrc/obj.json
"""
