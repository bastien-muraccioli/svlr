from tools.robot_tool import pixel_to_robot
from actions.call_actions import call_robot_function

from sentence_transformers import SentenceTransformer, util
import re
import os

def parse_action_text(action_text: str):
    action_text_lines = action_text.split('\n')
    first_action_found = False
    # Define a list of action dicts of the format: {'action': 'action_name', param: ['param1', 'param2'...]}
    action_list = [] 
    # Define a regex pattern to match the action_name and optional parameters
    pattern = r'^(?P<action_name>[^\:]+)(?:\:\s*\[(?P<parameters>[^\]]*)\])?$'
    
    # Search in action_text all the actions and the parameters
    for lines in action_text_lines:
        # Use re.match to find the pattern in the action_text
        match = re.match(pattern, lines)
        if match:
            if not first_action_found:
                first_action_found = True
                
            # Extract action_name and parameters from match groups
            action_name = match.group('action_name')
            parameters_str = match.group('parameters')
            
            if parameters_str:
                # Split parameters by comma and strip whitespace
                parameters = [param.strip() for param in parameters_str.split(',') if param.strip()]
            else:
                parameters = 'None'
            action_list.append({'action': action_name, 'param': parameters})
        
        # After have found one action, if no more action is found stop            
        if not match and first_action_found:
            break
        
    return action_list

class ActionManager:
    def __init__(self, robot_info: dict):
        self.robot_info = robot_info
        self.robot_actions = self.robot_info["actions"]
        self.robot_actions_name = [action["name"] for action in self.robot_actions]
         # Initialize the similarity model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "similarity_model", "all-MiniLM-L6-v2")
        self.similarity_model = SentenceTransformer(model_path)
        # Embed the robot actions
        self.robot_actions_embedding = self.similarity_model.encode(self.robot_actions_name, convert_to_tensor=True).cpu()

    def run(self, action_text: str, environment_description_list: list, environment_pos: dict):
        action_dict_list = []
        action_list = []
        llm_output_action_list = parse_action_text(action_text=action_text)
        print(f"Actions found in the LLM Response:\n{llm_output_action_list}")
        
        # Use sentence similarity to ensure that LLM output matches actions defined in robot_action.json and parameters defined by the VLM (environment_description_list)
        
        for llm_action in llm_output_action_list:
            parameters = []
            action_name = self.most_similar(target=llm_action['action'], 
                                            compare_list=self.robot_actions_name, 
                                            embedded_compare_list=self.robot_actions_embedding)
            if llm_action['param'] != 'None':
                for llm_parameter in llm_action['param']:
                    # parameter_text = name of an object
                    parameter_text = self.most_similar(target=llm_parameter, 
                                                compare_list=environment_description_list)
                    # based on the name, find the pixel coordinates
                    parameter_pixel = environment_pos[parameter_text]
                    # convert it to robot coordinates and add it to the list of the final parameters
                    parameters.append(pixel_to_robot(self.robot_info['robot_name'], parameter_pixel))
            else:
                parameters = 'None'
            print(f"Formatted Action: {action_name}, Parameters: {parameters}")
            action_list.append({'action': action_name, 'param': parameters})
        
        for executable_action in action_list:
            if executable_action['param'] == 'None':
                param = (None, )
            else:
                param = (executable_action['param'])
            action_dict_list += call_robot_function(self.robot_info['robot_name'], executable_action['action'], *param)
        
        return action_dict_list
    
    def most_similar(self, target: str, compare_list: list, embedded_compare_list=None):
        
        embedded_target = self.similarity_model.encode(target, convert_to_tensor=True).cpu()
        
        if embedded_compare_list == None:
            embedded_compare_list = self.similarity_model.encode(compare_list, convert_to_tensor=True).cpu()
        
        # Find the most similar element of target in the compare_list
        similarities = util.cos_sim(embedded_target, embedded_compare_list)[0]
        most_similar_index = similarities.argmax().item()
        
        return compare_list[most_similar_index]
        
        