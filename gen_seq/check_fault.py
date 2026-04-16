import json
import re
from collections import defaultdict, Counter

def analyze_trajectory_errors(file_path):
    # ================= 基础统计 =================
    total_tasks = 0
    total_failed_tasks = 0      
    total_actions = 0           # 全局总动作数(Step数)
    
    # ================= 互斥的失败主因统计 =================
    # 改用 defaultdict，彻底杜绝 KeyError 报错！
    failure_reasons = defaultdict(int)
    
    # 细节诊断统计 
    format_err_actions_count = 0  # 记录格式错误动作总数
    format_err_details = defaultdict(int)
    missing_backticks_details = defaultdict(int)
    
    action_patterns = [
        r"^click \[\d+\] \[.*?\]$",
        r"^type \[\d+\] \[.*?\](?: \[[01]\])?$",
        r"^hover \[\d+\] \[.*?\]$",
        r"^press \[.*?\]$",
        r"^scroll \[(?:down|up)\]$",
        r"^new_tab$",
        r"^tab_focus \[\d+\]$",
        r"^close_tab$",
        r"^goto \[.*?\]$",
        r"^go_back$",
        r"^go_forward$",
        r"^stop \[.*?\]$"
    ]
    valid_verbs = ['click', 'type', 'hover', 'press', 'scroll', 'new_tab', 'tab_focus', 'close_tab', 'goto', 'go_back', 'go_forward', 'stop']

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                task_success = data.get('success', False) 
                total_tasks += 1
                
                if task_success:
                    # 对于成功任务，我们依然需要把它的 step 数加到全局总动作数里
                    total_actions += len(data.get('steps', []))
                    continue # 成功任务跳过后续错误分析
                
                total_failed_tasks += 1
                
                # 当前任务的状态标记
                task_has_loop = False
                task_has_format_error = False
                task_called_stop = False
                
                action_counts = Counter()
                
                for step in data.get('steps', []):
                    total_actions += 1  # 累加全局动作数
                    
                    model_response = step.get('model_response', '')
                    match = re.search(r'</think>\s*```(.*?)```', model_response, re.DOTALL)
                    extracted_action = ""
                    
                    # 1. 检查格式错误
                    if not match:
                        task_has_format_error = True
                        format_err_actions_count += 1  # 累加动作错误数
                        
                        # 记录细节诊断
                        if '</think>' not in model_response:
                            format_err_details["缺少 </think> 闭合标签"] += 1
                        elif '`'*3 not in model_response.split('</think>')[-1]:
                            format_err_details["</think> 后缺少反引号包裹动作"] += 1
                            
                            text_after_think = model_response.split('</think>')[-1].strip()
                            if not text_after_think:
                                missing_backticks_details["1. 完全空白 (无输出)"] += 1
                            else:
                                first_word = text_after_think.split()[0].lower()
                                clean_first_word = first_word.replace('`', '').strip(':')
                                if text_after_think.startswith('`') and not text_after_think.startswith('`'*3):
                                    missing_backticks_details["2. 反引号数量不足"] += 1
                                elif clean_first_word in valid_verbs:
                                    missing_backticks_details["3. 纯文本直接输出动作"] += 1
                                elif "action" in clean_first_word or "step" in clean_first_word:
                                    missing_backticks_details["4. 包含多余的引导词"] += 1
                                else:
                                    missing_backticks_details["5. 输出其他非标准文本"] += 1
                        else:
                            format_err_details["标签结构混乱"] += 1
                            
                    else:
                        extracted_action = match.group(1).strip()
                        is_valid_format = any(re.match(pattern, extracted_action) for pattern in action_patterns)
                        if not is_valid_format:
                            task_has_format_error = True
                            format_err_actions_count += 1  # 累加动作错误数
                            action_verb = extracted_action.split(' ')[0] if extracted_action else "EMPTY"
                            format_err_details[f"动作语法错误 ({action_verb})"] += 1
                            
                        # 检查是否调用了 stop
                        if extracted_action.startswith("stop"):
                            task_called_stop = True

                    # 2. 检查死循环
                    action_for_loop = extracted_action if extracted_action else step.get('action', '').strip()
                    if action_for_loop:
                        action_counts[action_for_loop] += 1
                        if action_counts[action_for_loop] > 5:
                            task_has_loop = True
                            
                # ====================================================
                # 为失败任务进行互斥归因 (严格要求格式版)
                # ====================================================
                # # 优先级 1：对格式零容忍，只要轨迹中包含哪怕一次格式错误，就归为格式崩溃
                # if task_has_format_error:
                #     failure_reasons["1. 严重格式崩溃 (轨迹中包含非法格式输出)"] += 1
                # # 优先级 2：格式没问题的情况下，看有没有死循环
                # elif task_has_loop:
                #     failure_reasons["2. 陷入动作死循环 (重复动作>5次)"] += 1
                # # 优先级 3：格式没问题、没死循环，但最后给出了错误答案
                # elif task_called_stop:
                #     failure_reasons["3. 最终答案错误 (执行了stop但答案错)"] += 1
                # # 优先级 4：啥毛病没有，就是没找到答案耗尽了步数
                # else:
                #     failure_reasons["4. 步数耗尽/迷失 (没找到答案)"] += 1

                # 优先级 1：对格式零容忍，只要轨迹中包含哪怕一次格式错误，就归为格式崩溃
                if task_called_stop:
                    failure_reasons["1. 最终答案错误 (执行了stop但答案错)"] += 1
                # 优先级 2：格式没问题的情况下，看有没有死循环
                elif task_has_loop:
                    failure_reasons["2. 陷入动作死循环 (重复动作>5次)"] += 1
                # 优先级 3：格式没问题、没死循环，但最后给出了错误答案
                elif task_has_format_error:
                    failure_reasons["3. 严重格式崩溃 (轨迹中包含非法格式输出)"] += 1
                # 优先级 4：啥毛病没有，就是没找到答案耗尽了步数
                else:
                    failure_reasons["4. 步数耗尽/迷失 (没找到答案)"] += 1

    except Exception as e:
        import traceback
        print(f"解析出错: {e}")
        traceback.print_exc() 
        return

    # ================= 打印报告 =================
    print("="*70)
    print(" 轨迹失败原因深度提炼报告 (严格格式零容忍版) ")
    print("="*70)
    print(f"总任务数: {total_tasks} | 失败任务数: {total_failed_tasks} (失败率: {total_failed_tasks/max(total_tasks,1)*100:.2f}%)")
    print("-" * 70)
    
    print("【主成分分析：为什么任务会失败？】(互斥分类，合计 100%)")
    for reason, count in sorted(failure_reasons.items()):
        fault_percentage = count / max(total_failed_tasks, 1) * 100
        all_percentage = count / max(total_tasks, 1) * 100
        print(f"  {reason}: {count} 个任务 ({fault_percentage:.2f}%, 占总任务 {all_percentage:.2f}%)")
        
    print("-" * 70)
    print(f"【附加诊断：格式错误细节追踪】(不影响主分类)")
    # 这里添加了你需要的全局动作数打印！
    print(f"  -> 全局总动作数(Steps): {total_actions} | 其中格式错误动作数: {format_err_actions_count} | 错误率: {(format_err_actions_count/total_actions*100):.2f}%")
    
    if format_err_details:
        for err_type, count in sorted(format_err_details.items(), key=lambda x: x[1], reverse=True):
            print(f"  * {err_type}: {count} 次")
            
    if missing_backticks_details:
        print("\n  [</think> 后缺少反引号 细分]:")
        for sub_err, count in sorted(missing_backticks_details.items()):
            print(f"    -> {sub_err}: {count} 次")
    print("="*70)


if __name__ == "__main__":
    TEST_FILE_PATH = "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_task-opsrc-sft-1e-5lr-freeze_true-2epoch/nq_test_results.jsonl" 
    analyze_trajectory_errors(TEST_FILE_PATH)