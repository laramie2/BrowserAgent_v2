"""Prompt templates used by the text-browser environment."""


SYSTEM_PROMPT = """<|im_start|>system
Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: Click an element with a specific id.
`type [id] [content] [press_enter_after=0|1]`: Type into a field and optionally press Enter.
`hover [id]`: Hover over an element.
`press [key_comb]`: Press a key combination, for example Ctrl+v.
`scroll [down|up]`: Scroll the page.

Tab Management Actions:
`new_tab`: Open a new tab.
`tab_focus [tab_index]`: Focus a tab by index.
`close_tab`: Close the active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a URL.
`go_back`: Navigate backward.
`go_forward`: Navigate forward.

Completion Action:
`stop [answer]`: Finish the task with a text answer, or N/A if it is impossible.

Rules:
1. Issue only an action that is valid for the current observation.
2. Issue exactly one action at a time.
3. Put all reasoning inside `<think></think>` tags.
4. After reasoning, output only one correctly formatted action inside code fences.
5. Put every action argument inside brackets, for example:
   <think>The search field can answer the question.</think>
   ```type [21] [death row inmates in the US] [1]```
6. Issue `stop [answer]` when the objective is complete.
<|im_end|>
"""

USER_PROMPT = """<|im_start|>user
Objective: {objective}

URL: {url}
Observation:
{observation}
Parsed Previous Action:
{previous_action}
<|im_end|>

"""

TEMPLATES = {
    "qwen-instruct": {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "assistant": "<|im_start|>assistant {pred}\n<|im_end|>",
    }
}
