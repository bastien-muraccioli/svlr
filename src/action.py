import time
from actions.call_actions import call_robot_class
from src.entity import Entity
from src.perception import Perception

from sentence_transformers import SentenceTransformer, util
import json
import re
import os

from typing import List

class ActionManager:
    def __init__(self, robot_info: dict, perception: Perception):
        self.robot_info = robot_info
        self.robot_actions = self.robot_info["actions"]
        self.robot_actions_name = [action["name"] for action in self.robot_actions]
        self.robot_action_class = call_robot_class(self.robot_info["robot_name"])
        self.perception = perception

        # Initialize the similarity model
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "similarity_model",
            "all-MiniLM-L6-v2",
        )
        self.similarity_model = SentenceTransformer(model_path)
        # Embed the robot actions
        self.robot_actions_embedding = self.similarity_model.encode(
            self.robot_actions_name, convert_to_tensor=True
        ).cpu()

    def run(
        self,
        action_text: str,
        environment_description_list: List[Entity],
    ):
        action_list = []
        llm_output_action_list = parse_action_text(action_text=action_text)
        print(f"Actions found in the LLM Response:\n{llm_output_action_list}")

        # Use sentence similarity to ensure that LLM output matches actions defined in robot_action.json and parameters defined by the VLM (environment_description_list)
        for llm_action in llm_output_action_list:
            parameters = [] # List of Entity objects corresponding to the parameters
            action_name = self.most_similar(
                target=llm_action["action"],
                compare_list=self.robot_actions_name,
                embedded_compare_list=self.robot_actions_embedding,
            )

            if llm_action["param"] != "None":
                for llm_parameter in llm_action["param"]:
                    # parameter_text = name of an entity in the environment_description_list that is the most similar to the llm_parameter
                    parameter_text = self.most_similar(
                        target=llm_parameter, compare_list=[entity.name for entity in environment_description_list]
                    )
                    # Save the entity corresponding to the parameter text
                    p = next((ent for ent in environment_description_list if ent.name == parameter_text), None)
                    if p is not None:
                        parameters.append(p)
                    else:
                        parameters.append(None)
            else:
                parameters = None

            print(f"Formatted Action: {action_name}, Parameters: {parameters}")
            action_list.append({"action": action_name, "param": parameters})

        for executable_action in action_list:
            if executable_action["param"] is None:
                param = (None,)
            else:
                param = executable_action["param"]

            self.robot_action_class.add_actions(
                executable_action["action"],
                *param
            )
        
        self.robot_action_class.action_finished = False
    
    def action_tracking(self,
                        environment_description_list: List[Entity]):
        """ Update current action_dict_list with the robot coordinates of the objects in the environment based on the latest perception results."""

        current_low_level_action = self.robot_action_class.current_low_level_action()
        if current_low_level_action is None:
            print("No current action to track.")
            return False, environment_description_list
        entity_involved = current_low_level_action["entity"]
        if(entity_involved is None):
            print("No entity involved in the current action.")
            return False, environment_description_list
        if current_low_level_action["tracking"] is False:
            return False, environment_description_list
        
        # Set the need_to_be_tracked flag for the entity involved in the low level action
        # Reset tracking flags for all others entities
        matched_entity = None
        for entity in environment_description_list:
            if entity.name == entity_involved.name:
                matched_entity = entity
                if not entity.tracked:
                    entity.need_to_be_tracked = True
            else:
                entity.reset_tracking()

        # # Check if the involved entity is still tracked
        # if matched_entity is None or not matched_entity.found:
        #     entity_involved.found = False
        #     try_count = 0
        #     while not entity_involved.found and try_count < self.max_try_to_find_entity:
        #         print(f"Entity {entity_involved.name} lost, trying to find it again ({try_count+1}/{self.max_try_to_find_entity})...")
        #         if self.perception.segment_one_entity(entity_involved.name):
        #             print(f"Entity {entity_involved.name} found again.")
        #         else:
        #             print(f"Entity {entity_involved.name} not found, please adjust the camera.")
        #             time.sleep(1)  # Wait before trying again
        #         try_count += 1
        #     if not entity_involved.found:
        #         print(f"Failed to find entity {entity_involved.name} after {self.max_try_to_find_entity} attempts. Resetting action.")
        #         self.robot_action_class.reset_action()
        #         return False, environment_description_list
        #     else:
        #         matched_entity = next((ent for ent in environment_description_list if ent.name == entity_involved.name), None)
        
        # print(f'Entity {entity_involved.name}, pos: {matched_entity.robot_frame_pos}, found: {matched_entity.found}')
        # Update the entity involved in the action with the latest perception results
        self.robot_action_class.sync_action(matched_entity)

        return True, environment_description_list

    def most_similar(self, target: str, compare_list: list, embedded_compare_list=None):

        embedded_target = self.similarity_model.encode(
            target, convert_to_tensor=True
        ).cpu()

        if embedded_compare_list == None:
            embedded_compare_list = self.similarity_model.encode(
                compare_list, convert_to_tensor=True
            ).cpu()

        # Find the most similar element of target in the compare_list
        similarities = util.cos_sim(embedded_target, embedded_compare_list)[0]
        most_similar_index = similarities.argmax().item()

        return compare_list[most_similar_index]

def parse_action_text(action_text: str):
    """
    Parse the JSON output from the VLM/LLM that describes robot actions.

    Expected format:
    [
      { "action": "pick and place", "parameters": ["toothpaste tube", "cup"] },
      { "action": "open", "parameters": ["cup lid"] }
    ]

    Returns:
        list[dict]: A list of dicts with keys 'action' and 'param'.
    """
    # Try to extract valid JSON (in case the model includes markdown formatting)
    try:
        # Remove markdown-style code fences if present
        clean_text = re.sub(r"```(?:json)?", "", action_text).strip()
        action_list_raw = json.loads(clean_text)
    except json.JSONDecodeError:
        print("⚠️ Warning: JSON decoding failed. Attempting fallback parsing.")
        action_list_raw = []

    # Normalize the data to your expected internal format
    action_list = []
    for entry in action_list_raw:
        if not isinstance(entry, dict):
            continue
        action_name = entry.get("action", "").strip()
        parameters = entry.get("parameters", [])
        # Ensure parameters is always a list
        if isinstance(parameters, str):
            parameters = [parameters]
        elif not isinstance(parameters, list):
            parameters = []
        action_list.append({
            "action": action_name,
            "param": [p.strip() for p in parameters if isinstance(p, str)]
        })

    return action_list