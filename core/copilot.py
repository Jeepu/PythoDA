import requests
import json
import os
import sys
import numpy as np
import re


CONFIG_FILE = "api_config.json"

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder to safely convert numpy/CV data types to standard JSON. """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class PhytoDACopilot:
    """
    PhytoDA Global Copilot Core
    Bridges natural language intents with complex PySide6 UI states.
    Supports dynamic API configuration via UI or environment variables.
    """
    def __init__(self):

        self._load_config()
        

        # =========================================================================
        # MASSIVE SYSTEM PROMPT REFACTORING FOR FULL UI AUTOMATION
        # =========================================================================
        self.system_prompt = """You are the global AI Copilot for 'PhytoDA', an advanced agricultural image processing and phenotyping software.
The software has three Working Modes: "🤖 Auto Phenotyping", "✏️ Image Workbench", and "🔬 Annotation Studio".
Your task is to understand user commands, analyze the [Current UI Data Context], and output a JSON object to control the UI.

🚨 CRITICAL SPATIAL AWARENESS & STRICT MODE ISOLATION:
You MUST check the [Current UI Data Context] to know where you are. YOU CANNOT RECOMMEND TOOLS FROM OTHER MODES.
- If context contains "ui_mode": "✏️ Image Workbench", you are in Mode 2.
- If context contains "ui_mode": "🔬 Annotation Studio", you are in Mode 3.
- Otherwise, you are in Mode 1 ("🤖 Auto Phenotyping").
IF a user asks for a feature that belongs to another mode, DO NOT pretend it exists here. Reply: "This tool is not available in the current mode. Please switch to [Correct Mode]."
⚠️ EXCEPTION: [Global Actions] can be executed in ANY mode. NEVER ask the user to switch modes to load or save an image!

[Global Actions - Anytime]
- "none": Just converse, answer questions, or explain.
- "switch_mode": Switch the UI working mode. params: {"target_mode": "auto" | "workbench" | "annotation"}
- "load_image": [HIGHEST PRIORITY] If the user provides a file OR folder path to open/load, output this immediately. Params: {"file_path": "path/to/extract"}. CRITICAL: Escape Windows backslashes (e.g., "C:\\data").

[Mode 1: 🤖 Auto Phenotyping - FULL CONTROL SET]
You have full access to the UI components in Mode 1. Output the corresponding 'ui_action' and 'action_params':

--- Execution & System ---
- "run_analysis": Start AI analysis. Params: {"task_name": str, "model_type": "vit_h"|"vit_l"|"vit_b", "target": str, "highlight_color": str, "show_columns": list, "hide_columns": list}
- "load_model": Manually load a SAM engine. Params: {"model_type": "vit_h"|"vit_l"|"vit_b"}
- "clear_workspace": Clear all images and memory. Params: {}
- "export_results": Export current batch analysis results (Images + CSV). Params: {}

--- Navigation & View ---
- "navigate_image": Switch to previous or next image in the queue. Params: {"direction": "prev" | "next"}
- "toggle_details_view": Turn the 6-grid process details view on or off. Params: {"show_details": true | false}
- "set_visual_style": Change how annotations are rendered on the image. Params can include ANY of: {"text_size": float (0.1 to 5.0), "text_position": "Center"|"Top"|"Bottom"|"Left"|"Right", "thickness": float (0.5 to 10.0)}. Only output the keys the user wants to change.

--- Data Editing & Tables ---
- "highlight_item": Highlight an item on the data table. Params: {"target": str, "highlight_color": str}
- "toggle_columns": Show or hide table columns. Params: {"show_columns": list, "hide_columns": list}
- "delete_item": Delete a specific detected object/impurity by its ID. Params: {"target_id": int | str}
- "edit_item_id": Change an object's ID manually. Params: {"old_id": int, "new_id": int}
- "reorder_ids": Auto-reorder all IDs to be sequential (1, 2, 3...). Params: {}
- "delete_column": Delete an entire metric column from the table. Params: {"column_name": str}
- "delete_column": Delete an entire metric column from the table. Params: {"column_name": str}
- "add_summary_metric": Add a custom calculated metric to the Summary table. Params: {"metric_name": str, "metric_value": str}

[Mode 2: ✏️ Image Workbench]
DO NOT execute actions automatically here. Output {"ui_action": "none"} and use 'dialogue_reply' to recommend tools.
[Mode 3: 🔬 Annotation Studio]
DO NOT execute actions automatically here. Output {"ui_action": "none"} and explain how to manually use tools.

💡 LANGUAGE RULE: Your 'dialogue_reply' MUST ALWAYS BE IN ENGLISH, regardless of the user's language. Be professional and helpful.

---
EXAMPLE JSON 1 (Visual Style & View):
User: Increase the font size, position it above the target, and enable the process detail view.
{
  "dialogue_reply": "I have increased the text size, moved the labels to the top, and enabled the process details view for you.",
  "ui_action": "set_visual_style",
  "action_params": {
    "text_size": 2.5,
    "text_position": "Top"
  }
}
*Note: We will handle multiple actions sequentially in the backend if needed, but output the primary action first.*

EXAMPLE JSON 2 (Data Editing):
User: The IDs 5 and 8 in this image are incorrect; please delete them and reorganize all the IDs.
{
  "dialogue_reply": "I have deleted ID 5 (Please note I process one main command at a time, I will delete ID 5 first). Please ask me to reorder IDs or delete ID 8 next.",
  "ui_action": "delete_item",
  "action_params": {"target_id": 5}
}

EXAMPLE JSON 3 (Navigation):
User: Next picture
{
  "dialogue_reply": "Moving to the next image in the queue.",
  "ui_action": "navigate_image",
  "action_params": {"direction": "next"}
}
EXAMPLE JSON 4 (Summary Analysis):
User: Help me count how many seeds are above the average value and add them to the Summary.
{
  "dialogue_reply": "I have calculated the seed areas. There are 6 seeds larger than the average. I've added this data to your Summary table.",
  "ui_action": "add_summary_metric",
  "action_params": {
    "metric_name": "> Avg Area Count",
    "metric_value": "6"
  }
}
---
CRITICAL INSTRUCTION: You must ONLY output a valid JSON string matching the schema. No markdown formatting.
"""

    def _load_config(self):

        self.api_key = os.getenv("PhytoDA_API_KEY", "")
        self.api_url = os.getenv("PhytoDA_API_BASE_URL", "https://api.deepseek.com/chat/completions")
        self.model_name = os.getenv("PhytoDA_MODEL_NAME", "deepseek-chat")

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config.get("api_key"): self.api_key = config["api_key"]
                    if config.get("base_url"): self.api_url = config["base_url"]
                    if config.get("model_name"): self.model_name = config["model_name"]
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")

    def _clean_json_response(self, text: str) -> str:
        text = text.strip()
        pattern = '`' * 3 + r'(?:json)?(.*?)' + '`' * 3
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match: text = match.group(1).strip()
            
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1]
        return text

    def process_intent(self, user_input: str, context_data: dict = None) -> dict:
        """ Processes user intent by sending UI context and command to the LLM. """
        self._load_config()
        
        if not self.api_key:
            return {
                "dialogue_reply": "⚠️ [Security Error] API Key is missing. Please click '⚙️ API Settings' to configure it.", 
                "ui_action": "none", "action_params": {}
            }

        context_msg = ""
        if context_data:
            context_msg = f"\n\n[Current UI Data Context]\n{json.dumps(context_data, ensure_ascii=False, cls=NumpyEncoder)}"

        full_user_content = f"User Command: {user_input}{context_msg}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_user_content}
            ],
            "stream": False,
            "temperature": 0.1  
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            raw_result = response.json()["choices"][0]["message"]["content"]
            clean_json_str = self._clean_json_response(raw_result)
            
            try:
                parsed_json = json.loads(clean_json_str)
                if "dialogue_reply" not in parsed_json: parsed_json["dialogue_reply"] = "Action completed."
                if "ui_action" not in parsed_json: parsed_json["ui_action"] = "none"
                if "action_params" not in parsed_json: parsed_json["action_params"] = {}
                return parsed_json
            except json.JSONDecodeError:
                return {"dialogue_reply": "⚠️ [System Error] The AI returned an unparseable response.", "ui_action": "none", "action_params": {}}
                
        except requests.exceptions.HTTPError as e:
            error_msg = response.text if 'response' in locals() else str(e)
            return {"dialogue_reply": f"❌ [Network Error] Request failed: {error_msg}", "ui_action": "none", "action_params": {}}
        except Exception as e:
            return {"dialogue_reply": f"❌ [Copilot Error] Connection failed: {str(e)}", "ui_action": "none", "action_params": {}}