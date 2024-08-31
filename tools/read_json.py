import os
import json

def read_robot_json(robot_name: str) -> dict:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "actions", f"{robot_name}_action.json")
    with open(path, "r") as f:
        robot_info = json.load(f)
    return robot_info

def read_llm_prompt_json(llm_name: str) -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_prompt.json")
    with open(path, "r") as f:
        prompt_json = json.load(f)
    try:
        llm_prompt = prompt_json[llm_name]
    except KeyError:
        llm_prompt = prompt_json["default"]
        print(f"LLM prompt for {llm_name} not found. Using default prompt.")
    return llm_prompt