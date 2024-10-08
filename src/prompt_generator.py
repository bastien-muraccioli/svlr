from src.perception import Perception

class PromptGenerator:
    def __init__(self, robot_info: dict):
        self.robot_info = robot_info
        self.environment_prompt = "" # Generated by the perception module (VLM)
        self.robot_prompt = f"Description of the robot:\n{self.robot_info['description']}\n\nList of Actions that the robot can do:\n{self.robot_action_to_readable_format()}"
        self.user_command = ""
        self.environment_description_list = None
        #Initialize perception
        self.perception = Perception()

    def run(self, user_input: str, image):
        self.user_command = f"User Command:\n{user_input}"
        #Run perception
        self.environment_description_list = self.perception.run(image)
        environment_description = ", ".join(self.environment_description_list)
        self.environment_prompt = f"Environment Description:\n{environment_description}"
        prompt = "\n\n".join([self.environment_prompt, self.robot_prompt, self.user_command, "Solution:"])
        return prompt

    def robot_action_to_readable_format(self):
        actions = self.robot_info["actions"]
        result = "\n"
        for action in actions:
            result += f"- Action_name: {action['name']}\n"
            result += f"\tAction_description: {action['description']}\n"
            try:
                result += f"\tAction_parameters: {action['parameters']}\n\n"
            except:
                result += "\n"
        # Remove the last two newlines
        result = result[:-2]
        return result