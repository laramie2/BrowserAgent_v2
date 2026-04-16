#!/usr/bin/env python
"""
Interactive smoke-test for the Text-Browser tool server.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type text_browser \
        --url=http://localhost:5000/get_observation

Then execute this script:
    python -m VTC_tool.interactive_browser_demo run --url=http://localhost:5000/get_observation

"""

import os
import json
import uuid
import logging
import requests
import fire

from pathlib import Path
from .VTC_tool import VTCTool


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def interactive_browser(
    url: str = "http://localhost:5000/get_observation",
    trajectory_prefix: str = "interactive-agent"

):
    traj_id = f"{trajectory_prefix}-{uuid.uuid4()}"
    logger.info(f"Started session. ID: {traj_id}")

    extra_field = {
        "question": "who plays the wildling woman in game of thrones",
        "url": "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        # "url": "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        # "url": "https://www.wikipedia.org/"
        # "url": "https://www.baidu.com"
    }

    current_action = "" # 初始为空，获取首页
    step = 0
    logger.info(f"Target URL is: {extra_field['url']}")

    while True:
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
            # 假设返回的是列表，取第一个
            obs = data[0] if isinstance(data, list) else data
            ob_text = json.dumps(obs, indent=2, ensure_ascii=False)
            print(ob_text)
            print("="*60 + "\n")

            project_root = Path(__file__).resolve().parents[1]
            save_dir = project_root / "VTC_tool" / "saved_img_demo"
            os.makedirs(save_dir, exist_ok=True) # 如果目录不存在则创建，存在则忽略

            # ========================================================================
            # 将工具返回观测文本渲染为图像并保存
            # ========================================================================
            highlight_keywords = ["Observation", "Action", "observation", "action", "url:"] # 需要高亮的关键词列表
            highlights = [
                {"context": "action:", "color": (255, 150, 150)},      # 动作标红
                {"context": "observation:", "color": (150, 255, 150)}, # 观测标绿
                {"context": "url:", "color": (255, 50, 50)},         # 错误标深红
                {"context": "\\n", "color": (100, 100, 255)}           # 紧凑模式下的转义换行符标蓝
            ]

            # vtc = VTCTool(
            #     font_size=18, 
            #     bg_color=(20, 20, 20), 
            #     text_color=(200, 200, 200), 
            #     title_color=(80, 180, 80), 
            #     highlight_configs=highlights)
            # img, char_count = vtc.render_text_to_image(
            #     ob_text, use_compact_mode=True, 
            #     max_width=2048, 
            #     max_height=4096
            # )
            
            vtc = VTCTool(
                font_size=18, 
                bg_color=(20, 20, 20),           # 深灰底色
                text_color=(200, 200, 200),      # 浅灰文字
                title_color=(80, 180, 80),       # 绿色标题
                id_color=(255, 215, 0)           # 【新增】金黄色，专门用于高亮 [343] 这种可交互 ID
            )
            img, char_count = vtc.render_text_to_image_simple(
                ob_text, 
                width=1024,                      # 设定基准宽度，1024 对 ViT 来说通常足够清晰
                aspect_ratio="4:3"               # 生成 1024 x 768 的稳定比例图像
            )
            
            img = vtc.compress_image_arrays([img], compression_factor=3.0)[0] # 进一步压缩以节省空间         

            # 使用 os.path.join 拼接路径更安全
            save_path = os.path.join(save_dir, f"observation_step_{step}.png")
            img.save(save_path)
          
            logger.info(f"Observation image saved as {save_path} with {char_count} characters.")
            # ========================================================================

        except Exception as e:
            # 这里是关键：打印详细的错误堆栈信息，看看究竟是 Permission Denied 还是 FileNotFoundError
            logger.error("Request or Save failed: %s", e, exc_info=True)
            break

        # 交互输入
        print(f"💡 指令示例: click [10] | type [5] [text] [1] | scroll [down]")
        user_input = input(f"Step {step} Action > ").strip()

        if user_input.lower() in ['quit', 'exit']:
            break     

        # 核心逻辑：自动包裹标签
        if user_input == "":
            current_action = "" # 允许发送空 action 刷新页面
        else:
            current_action = f"<think>\nManual\n</think>\n<action>{user_input}</action>"      

        step += 1



def main():
    fire.Fire({"run": interactive_browser})

if __name__ == "__main__":
    main()