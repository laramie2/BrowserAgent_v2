import json
import re
import os
from collections import defaultdict, Counter

def analyze_trajectory_errors(file_path):
    # ================= 基础统计 =================
    total_tasks = 0
    total_failed_tasks = 0      
    total_actions = 0           # 全局总动作数(Step数)
    
    # ================= 双轨统计字典 =================
    # 1. 致命主因 (互斥，加起来等于 total_failed_tasks)
    primary_failure_reasons = defaultdict(int)
    
    # 2. 缺陷感染率 (非互斥，记录有多少任务"至少触发过一次"该错误)
    defect_prevalence = defaultdict(int)
    
    # ================= 细节诊断统计 =================
    format_err_actions_count = 0  
    hallucinated_id_actions_count = 0  
    element_mismatch_actions_count = 0 
    
    format_err_details = defaultdict(int)
    missing_backticks_details = defaultdict(int)
    action_tag_err_details = defaultdict(int) 
    bad_action_samples = defaultdict(int) 
    hallucinated_id_details = defaultdict(int)
    element_mismatch_details = defaultdict(int)
    
    action_patterns = [
        r"^click \[\d+\] \[.*?\]$",
        r"^click \[\d+\]$",
        r"^type \[\d+\] \[.*?\](?: \[[01]\])?$",
        r"^hover \[\d+\] \[.*?\]$",
        r"^hover \[\d+\]",
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

    # ================= 设定文件输出路径 =================
    base_dir = os.path.dirname(os.path.abspath(file_path))
    output_dir = os.path.join(base_dir, "error_extractions")
    os.makedirs(output_dir, exist_ok=True) 
    
    # 打开五个用于写入的 jsonl 文件 (对应5大互斥主因)
    f_format_crash = open(os.path.join(output_dir, "1_format_crash_tasks.jsonl"), 'w', encoding='utf-8')
    f_hallucination = open(os.path.join(output_dir, "2_hallucination_tasks.jsonl"), 'w', encoding='utf-8')
    f_loop = open(os.path.join(output_dir, "3_loop_tasks.jsonl"), 'w', encoding='utf-8')
    f_wrong_answer = open(os.path.join(output_dir, "4_wrong_answer_tasks.jsonl"), 'w', encoding='utf-8')
    f_max_steps = open(os.path.join(output_dir, "5_max_steps_tasks.jsonl"), 'w', encoding='utf-8')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                task_success = data.get('success', False) 
                total_tasks += 1
                
                if task_success:
                    total_actions += len(data.get('steps', []))
                    continue 
                
                total_failed_tasks += 1
                
                # 当前任务的状态标记 (用于判定非互斥感染率和互斥主因)
                task_has_loop = False
                task_has_format_error = False
                task_has_hallucinated_id = False
                task_has_element_mismatch = False
                task_called_stop = False
                
                action_counts = Counter()
                
                for step in data.get('steps', []):
                    total_actions += 1  
                    
                    # === 1. 解析 Observation 中的有效元素 ===
                    obs_elements = {}
                    obs_text = step.get('observation', '')
                    for match in re.finditer(r'^[\t ]*\[(\d+)\]\s+([a-zA-Z]+)', obs_text, re.MULTILINE):
                        el_id = match.group(1)
                        el_type = match.group(2).lower()
                        obs_elements[el_id] = el_type
                    
                    # === 2. 提取 Action ===
                    model_response = step.get('model_response', '')
                    match = re.search(r'</think>\s*(?:```(.*?)```|<action>(.*?)</action>)', model_response, re.DOTALL)
                    extracted_action = ""
                    is_valid_format = False
                    
                    # 检查格式错误
                    if not match:
                        task_has_format_error = True
                        format_err_actions_count += 1
                        
                        if '</think>' not in model_response:
                            format_err_details["缺少 </think> 闭合标签"] += 1
                        else:
                            text_after_think = model_response.split('</think>')[-1].strip()
                            text_after_think_lower = text_after_think.lower()
                            
                            if not text_after_think:
                                format_err_details["</think> 后完全空白 (无输出)"] += 1
                            elif '<action>' in text_after_think_lower or '</action>' in text_after_think_lower:
                                format_err_details["<action> 格式规范错误"] += 1
                                if '<action>' in text_after_think_lower and '</action>' not in text_after_think_lower:
                                    action_tag_err_details["1. 有 <action> 但缺少 </action> 闭合标签"] += 1
                                elif '</action>' in text_after_think_lower and '<action>' not in text_after_think_lower:
                                    action_tag_err_details["2. 有 </action> 但缺少 <action> 起始标签"] += 1
                                elif text_after_think_lower.find('<action>') > 0:
                                    action_tag_err_details["3. </think> 与 <action> 之间包含了多余的废话文本"] += 1
                                else:
                                    action_tag_err_details["4. 标签结构错乱/多重嵌套/包含非法不可见字符"] += 1
                            elif '`' in text_after_think:
                                format_err_details["Markdown 反引号代码块错误"] += 1
                                if text_after_think.startswith('`') and not text_after_think.startswith('`'*3):
                                    missing_backticks_details["1. 反引号数量不足"] += 1
                                else:
                                    missing_backticks_details["2. Markdown 标签结构混乱"] += 1
                            else:
                                first_word = text_after_think.split()[0].lower().strip(':')
                                if first_word in valid_verbs:
                                    format_err_details["纯文本直接输出动作 (无标签包裹)"] += 1
                                elif "action" in first_word or "step" in first_word:
                                    format_err_details["包含多余的引导词 (如 Action: , Step: )"] += 1
                                else:
                                    format_err_details["输出其他非标准文本"] += 1
                    else:
                        extracted_action = (match.group(1) or match.group(2)).strip()
                        is_valid_format = any(re.match(pattern, extracted_action) for pattern in action_patterns)
                        if not is_valid_format:
                            task_has_format_error = True
                            format_err_actions_count += 1
                            action_verb = extracted_action.split(' ')[0] if extracted_action else "EMPTY"
                            format_err_details[f"动作语法内容错误 (如动作名不支持/参数错) -> {action_verb}"] += 1

                            short_action = extracted_action[:80].replace('\n', ' ')
                            bad_action_samples[f"[{action_verb}] {short_action}"] += 1
                            
                    if extracted_action.startswith("stop"):
                        task_called_stop = True

                    # === 3. 检查死循环 ===
                    action_for_loop = extracted_action if extracted_action else step.get('action', '').strip()
                    if action_for_loop:
                        action_counts[action_for_loop] += 1
                        if action_counts[action_for_loop] > 5:
                            task_has_loop = True

                    # === 4. 检查操作合法性 ===
                    if is_valid_format and extracted_action:
                        action_match = re.match(r'^(click|type|hover|tab_focus)\s+\[(\d+)\]', extracted_action)
                        if action_match:
                            verb = action_match.group(1)
                            target_id = action_match.group(2)
                            
                            if target_id not in obs_elements:
                                task_has_hallucinated_id = True
                                hallucinated_id_actions_count += 1
                                hallucinated_id_details[f"动作 {verb} 操作了不存在的ID [{target_id}]"] += 1
                            else:
                                target_type = obs_elements[target_id]
                                is_mismatch = False
                                
                                if verb == 'type':
                                    if target_type not in ['textbox', 'searchbox', 'textarea', 'combobox', 'input']:
                                        is_mismatch = True
                                elif verb in ['click', 'hover']:
                                    invalid_click_types = ['statictext', 'heading', 'rootwebarea', 'row', 'cell', 'table', 'group', 'paragraph', 'text']
                                    if target_type in invalid_click_types:
                                        is_mismatch = True
                                        
                                if is_mismatch:
                                    task_has_element_mismatch = True
                                    element_mismatch_actions_count += 1
                                    element_mismatch_details[f"动作 {verb} 操作了不支持的元素 <{target_type}>"] += 1

                # ====================================================
                # 【新策略】A：非互斥统计 (缺陷感染率) - 只要发生过就+1
                # ====================================================
                if task_called_stop:          defect_prevalence["[至少一次] 提交了错误答案 (Stop)"] += 1
                if task_has_loop:             defect_prevalence["[至少一次] 陷入动作死循环"] += 1
                if task_has_format_error:     defect_prevalence["[至少一次] 发生格式错误"] += 1
                if task_has_hallucinated_id:  defect_prevalence["[至少一次] 操作不存在的幻觉ID"] += 1
                if task_has_element_mismatch: defect_prevalence["[至少一次] 操作错误类型的元素"] += 1

                # ====================================================
                # 【新策略】B：互斥统计 (致命主因) - 按照严重性优先级拦截并写入文件
                # ====================================================
                if task_has_format_error:
                    primary_failure_reasons["1. 格式崩溃 (格式错误导致阻断)"] += 1
                    f_format_crash.write(line)
                elif task_has_hallucinated_id or task_has_element_mismatch:
                    primary_failure_reasons["2. 视觉/交互幻觉 (操作无效元素导致脱轨)"] += 1
                    f_hallucination.write(line)
                elif task_has_loop:
                    primary_failure_reasons["3. 决策死锁 (陷入死循环耗尽步数)"] += 1
                    f_loop.write(line)
                elif task_called_stop:
                    primary_failure_reasons["4. 逻辑错判 (执行了stop但答案错)"] += 1
                    f_wrong_answer.write(line)
                else:
                    primary_failure_reasons["5. 迷失/步数耗尽 (未发现代码级异常)"] += 1
                    f_max_steps.write(line)

    except Exception as e:
        import traceback
        print(f"解析出错: {e}")
        traceback.print_exc() 
        return
    finally:
        # 确保安全关闭所有5个文件句柄
        f_format_crash.close()
        f_hallucination.close()
        f_loop.close()
        f_wrong_answer.close()
        f_max_steps.close()

    # ================= 打印报告 =================
    print("="*80)
    print(" 轨迹失败原因深度提炼报告 (双轨分析版) ")
    print("="*80)
    print(f"总任务数: {total_tasks} | 失败任务数: {total_failed_tasks} (失败率: {total_failed_tasks/max(total_tasks,1)*100:.2f}%  |  成功率: {(total_tasks-total_failed_tasks)/max(total_tasks,1)*100:.2f}%)")
    print("-" * 80)
    
    print("【一、 致命主因分析】 (互斥，合计 100% —— 到底是谁杀死了任务？)")
    for reason, count in sorted(primary_failure_reasons.items()):
        fault_percentage = count / max(total_failed_tasks, 1) * 100
        print(f"  {reason}: {count} 个任务 ({fault_percentage:.2f}%)")
        
    print("\n【二、 缺陷感染率】 (非互斥，可超 100% —— 这些坏毛病有多普遍？)")
    for reason, count in sorted(defect_prevalence.items(), key=lambda x: x[1], reverse=True):
        infect_percentage = count / max(total_failed_tasks, 1) * 100
        print(f"  {reason}: {count} 个任务感染 (覆盖 {infect_percentage:.2f}% 的失败任务)")

    print("-" * 80)
    print(f"【三、 附加诊断：核心动作错误细节追踪】")
    print(f"  -> 全局总动作数(Steps): {total_actions}")
    
    print(f"\n  [一] 格式错误分析... | 触发动作数: {format_err_actions_count} ({format_err_actions_count/max(total_actions, 1) * 100:.2f}%)")
    if format_err_details:
        for err_type, count in sorted(format_err_details.items(), key=lambda x: x[1], reverse=True):
            print(f"    * {err_type}: {count} 次")
            if err_type == "<action> 格式规范错误" and action_tag_err_details:
                for sub_err, sub_count in sorted(action_tag_err_details.items()):
                    print(f"      - {sub_err}: {sub_count} 次")
            if err_type == "Markdown 反引号代码块错误" and missing_backticks_details:
                for sub_err, sub_count in sorted(missing_backticks_details.items()):
                    print(f"      - {sub_err}: {sub_count} 次")
        if bad_action_samples:
            print(f"\n    [!] 附：模型输出的非法动作指令采样 (Top 10)")
            for bad_str, count in sorted(bad_action_samples.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      - {bad_str} ({count} 次)")
                
    print(f"\n  [二] 幻觉ID分析... | 触发动作数: {hallucinated_id_actions_count} ({hallucinated_id_actions_count/max(total_actions, 1) * 100:.2f}%)")
    if hallucinated_id_details:
        for err_type, count in sorted(hallucinated_id_details.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    * {err_type}: {count} 次")

    print(f"\n  [三] 交互失配分析... | 触发动作数: {element_mismatch_actions_count} ({element_mismatch_actions_count/max(total_actions, 1) * 100:.2f}%)")
    if element_mismatch_details:
        for err_type, count in sorted(element_mismatch_details.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    * {err_type}: {count} 次")

    print("="*80)
    print(f"\n✅ 错误样本提取完成！")
    print(f"相关文件已保存至目录: {output_dir}/")
    print(f"包含以下文件: \n - 1_format_crash_tasks.jsonl\n - 2_hallucination_tasks.jsonl\n - 3_loop_tasks.jsonl\n - 4_wrong_answer_tasks.jsonl\n - 5_max_steps_tasks.jsonl")


if __name__ == "__main__":
    TEST_FILE_PATH = "/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/gen_seq/results/Qwen2.5-VL-7B-Instruct_task-opsrc-sft-5e-5lr-freeze_false-2epoch/nq_test_results.jsonl" 
    analyze_trajectory_errors(TEST_FILE_PATH)