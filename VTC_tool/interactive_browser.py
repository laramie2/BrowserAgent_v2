#!/usr/bin/env python
"""
Interactive smoke-test for the Text-Browser tool server with VLM Agent.
"""

import os
import re
import json
import uuid
import logging
import requests
import fire
import base64
from openai import OpenAI

from .VTC_tool import VTCTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. 客户端初始化函数
# ==========================================
API_KEY = ""
# API_KEY = ""
BASE_URL = "https://api.deepseek.com"
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

def init_llm_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """
    初始化 LLM 客户端。
    默认指向本地 vLLM/SGLang 服务，如果在云端请替换对应的 api_key 和 base_url。
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", None)

    logger.info(f"Initializing LLM client with base_url: {base_url}")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client

# ==========================================
# 2. 图像 Base64 编码辅助函数
# ==========================================
def encode_image_to_base64(image_path: str) -> str:
    """将保存的图片转换为 Base64 格式，用于 API 传输"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# 3. 核心交互循环函数
# ==========================================
def interactive_browser(
    client: OpenAI,
    model_name: str = "deepseek-reasoner", # 替换为你的具体模型名称
    max_turns: int = 15,
    url: str = "http://localhost:5000/get_observation",
    trajectory_prefix: str = "agent",
    use_image_obs: bool = True,
    log_dir: str = "./logs"  # 新增：指定日志保存的目录
):
    traj_id = f"{trajectory_prefix}-{uuid.uuid4().hex[:6]}"
    logger.info(f"Started session. ID: {traj_id}")

    # ==========================================
    # 初始化日志文件
    # ==========================================
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"traj_{traj_id}.txt")
    
    extra_field = {
        "question": "who is the director of iron man?",
        "url": "https://www.wikipedia.org/"
    }

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Session Started. ID: {traj_id}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Objective: {extra_field['question']}\n")
        f.write(f"Target URL: {extra_field['url']}\n")
        f.write("="*80 + "\n\n")

    # ==========================================
    # 系统 Prompt 设置
    # ==========================================
    system_prompt = """
                "Here's the information you'll have:\n"
                "The user's objective: This is the task you're trying to complete.\n"
                "The current web page's accessibility tree: This is a simplified representation of the webpage,\n"
                "  providing key information.\n"
                "The current web page's URL: This is the page you're currently navigating.\n"
                "The open tabs: These are the tabs you have open.\n"
                "The previous action: This is the action you just performed.\n\n"
                "The actions you can perform fall into several categories:\n\n"

                "Page Operation Actions:\n"
                "`click [id]`: This action clicks on an element with a specific id on the webpage.\n"
                "`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id.\n"
                "  By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n"
                "`hover [id]`: Hover over an element with id.\n"
                "`press [key_comb]`: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
                "`scroll [down|up]`: Scroll the page up or down.\n\n"

                "Tab Management Actions:\n"
                "`new_tab`: Open a new, empty browser tab.\n"
                "`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n"
                "`close_tab`: Close the currently active tab.\n\n"

                "URL Navigation Actions:\n"
                "`goto [url]`: Navigate to a specific URL.\n"
                "`go_back`: Navigate to the previously viewed page.\n"
                "`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"

                "Completion Action:\n"
                "`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a\n"
                "text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete,\n"
                "provide the answer as \"N/A\" in the bracket.\n\n"

                "To be successful, it is very important to follow the following rules:\n"
                "1. You should only issue an action that is valid given the current observation.\n"
                "2. You should only issue one action at a time.\n"
                "3. You should follow the examples to reason step by step and then issue the next action.\n"
                "4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.\n"
                "5. After `<think></think>`, only the action should be generated in the correct format, enclosed in '<action></action>'. You MUST USE `<action></action>` to wrap the action. Tags like `[action>` or `$action>` are FORBIDDEN.\n"
                "   For example:\n"
                "   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>\n"
                "   <action>click [1234]</action>\n"
                "\n"
                "   <think>I need to type the question into the search box and press enter to get the answer.</think>\n"
                "   <action>type [21] [death row inmates in the US] [press_enter_after=1]</action>\n"
                "\n"
                "   <think>There is no useful information available. I will scroll down to see more content.</think>\n"
                "   <action>scroll [down]</action>\n"
                "6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.\n"
                "7. If current web page is loading and you cannot get any useful information from the observation, output "<think></think>\n<action>None</action>".\n"
"""
    
    # 维护上下文消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Your objective is: {extra_field['question']}. Start browsing."}
    ]

    current_action = "" # 初始为空，获取首页
    step = 0

    logger.info(f"Target URL is: {extra_field['url']}")
    
    while step < max_turns:
        payload = {
            "trajectory_ids": [traj_id],
            "actions": [current_action],
            "extra_fields": [extra_field],
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            print("\n" + "="*60)
            print(f"OBSERVATION (Step {step})")
            print("="*60)
            obs = data[0] if isinstance(data, list) else data
            ob_text = json.dumps(obs, indent=2, ensure_ascii=False)
            print(ob_text)
            
            # ==========================================
            # 记录工具观测日志
            # ==========================================
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*20} [Step {step}] 🔍 网页观测 {'='*20}\n")
                f.write(ob_text + "\n")

            # 判断工具是否返回了任务完成信号
            dones_list = obs.get("dones", [False])
            # 取列表的第一个元素进行判断
            if dones_list and dones_list[0]:
                logger.info("Tool signaled that the task is finished.")
                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(f"\n✅ [任务结束] 工具返回完成信号。\n")
                break

            # ==========================================
            # 分支逻辑：图像观测 vs 纯文本观测
            # ==========================================
            if use_image_obs:
                save_dir = "/data/yutao/lzt/BrowserAgent_v2/VTC_tool/saved_img"
                os.makedirs(save_dir, exist_ok=True) 

                highlights = [
                    {"context": "action:", "color": (255, 150, 150)},
                    {"context": "observation:", "color": (150, 255, 150)},
                    {"context": "url:", "color": (255, 50, 50)},
                    {"context": "\\n", "color": (100, 100, 255)}
                ]
                vtc = VTCTool(font_size=18, bg_color=(20, 20, 20), text_color=(200, 200, 200), title_color=(80, 180, 80), highlight_configs=highlights)
                img, char_count = vtc.render_text_to_image(ob_text, use_compact_mode=True, max_width=2048, max_height=4096)
                
                save_path = os.path.join(save_dir, f"observation_step_{step}.png")
                img.save(save_path)
                logger.info(f"Observation image saved as {save_path}")

                base64_image = encode_image_to_base64(save_path)
                
                # 构造多模态 List 格式的内容
                step_content = [
                    {
                        "type": "text",
                        "text": f"Step {step} Observation:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            else:
                logger.info(f"Skipping image rendering, using raw text observation for Step {step}.")
                # 构造纯文本格式的内容
                step_content = f"Step {step} Observation:\n{ob_text}"

            # 追加当前轮次的观测到历史记录
            messages.append({"role": "user", "content": step_content})

            logger.info(f"Calling LLM API (Turn {step})...")
            
            # 调用 API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.2, # 较低的温度以保持动作的确定性
            )

            llm_reply = response.choices[0].message.content
            print("\n" + "="*60)
            print(f"🤖 LLM RESPONSE (Step {step})")
            print("="*60)
            print(llm_reply)

            # ==========================================
            # 记录 LLM 输出日志
            # ==========================================
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*20} [Step {step}] 🤖 LLM 输出 {'='*20}\n")
                f.write(llm_reply + "\n")
            
            # 将助手的回复加入历史记录，维持上下文闭环
            messages.append({"role": "assistant", "content": llm_reply})

            # ==========================================
            # 格式校验与自动修补 (Patching)
            # ==========================================
            patched_reply = llm_reply.strip()
            
            # 模拟工具服务器的正则验证逻辑
            pattern = r"<think>.*?</think>\s*(?:```.*?```|<action>.*?</action>)"
            matched = re.search(pattern, patched_reply, re.DOTALL)
            
            if not matched:
                logger.warning("LLM response failed format validation. Attempting to auto-patch...")
                
                has_think = "<think>" in patched_reply and "</think>" in patched_reply
                has_action = "<action>" in patched_reply and "</action>" in patched_reply
                has_codeblock = "```" in patched_reply
                
                if has_think and not (has_action or has_codeblock):
                    # 情况 1：有思考过程，但忘记写 <action> 标签
                    # 我们把 </think> 后面的所有内容强行塞进 <action> 里
                    parts = patched_reply.split("</think>", 1)
                    think_part = parts[0] + "</think>"
                    action_part = parts[1].strip() if len(parts) > 1 else ""
                    patched_reply = f"{think_part}\n<action>{action_part}</action>"
                    
                elif (has_action or has_codeblock) and not has_think:
                    # 情况 2：直接输出了动作，忘记写 <think> 标签
                    # 我们在最前面强行补一个空的思考过程
                    patched_reply = f"<think>Auto-patched missing thought process.</think>\n{patched_reply}"
                    
                else:
                    # 情况 3：彻底放飞自我，啥标签都没写
                    # 我们捏造一个假的 think，然后把整个大模型的输出当成 action 包起来
                    patched_reply = f"<think>Auto-patched missing thought process.</think>\n<action>{patched_reply}</action>"
                
                logger.info("Patched LLM reply successfully.")
                
                # 可选：将修补后的内容也记录到日志里，方便排查
                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*20} 🔧 自动修补后的 Payload {'='*20}\n")
                    f.write(patched_reply + "\n")

            # 将经过校验或修补后的字符串传给工具服务器
            current_action = patched_reply
            logger.info(f"Action payload prepared (length: {len(current_action)})")

            step += 1

        except Exception as e:
            logger.error("Agent Loop Failed: %s", e, exc_info=True)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"\n❌ [ERROR] 运行中断: {e}\n")
            break

    if step >= max_turns:
        logger.warning(f"Reached max turns ({max_turns}). Terminating session.")
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"\n⚠️ [警告] 达到最大交互轮数 ({max_turns})，终止执行。\n")


# ==========================================
# 4. 入口包装
# ==========================================
def main(base_url: str = None, max_turns: int = 15, use_image_obs: bool = True, log_dir: str = "./VTC_tool/logs"):
    """
    通过 fire 暴露给命令行的入口。
    --use_image_obs: 指示是否渲染为图片并发送
    --log_dir: 轨迹文件保存的位置
    """
    client = init_llm_client(base_url=base_url)
    interactive_browser(client=client, max_turns=max_turns, use_image_obs=use_image_obs, log_dir=log_dir)

if __name__ == "__main__":
    fire.Fire(main)