from src.llm import LLM
from src.perception import Perception
from src.prompt_generator import PromptGenerator
from src.action import ActionManager
from tools.read_json import read_robot_json
from tools.robot_tool import RobotCamera
from src.entity import Entity
from typing import List

import argparse
import torch


class ControlLoop:
    def __init__(self, args: argparse.Namespace, node=None):

        # Initialize the robot information
        self.robot_name = args.robot_name
        self.robot_info = read_robot_json(self.robot_name)

        # Save LLM parameters
        self.llm_temperature = args.llm_temperature
        self.llm_name = args.llm_name
        self.llm_provider = args.llm_provider
        self.llm_is_chat = args.llm_is_chat
        self.llm = None

        self.vlm_name = args.vlm_name
        self.vlm_provider = args.vlm_provider

        self.llm_is_vlm = True if self.llm_name == self.vlm_name else False
        if self.llm_is_vlm:
            print("LLM is VLM")

        self.node = node  # ROS node will be initialized in main.py

        robot_camera = RobotCamera(args, node=self.node)

        # Initialize perception
        self.perception = Perception(robot_camera=robot_camera, vlm_name=self.vlm_name, vlm_provider=self.vlm_provider)

        # Initialize the prompt generator
        self.prompt_generator = PromptGenerator(robot_info=self.robot_info, perception=self.perception)

        # Initialize action
        self.action = ActionManager(robot_info=self.robot_info, perception=self.perception)
        print("Control loop initialized")
    
    def perception_run(self, image):
        environment_description_list, vlm_raw_output, frame_with_masks_and_centers = self.perception.run(image)
        return environment_description_list, vlm_raw_output, frame_with_masks_and_centers
    
    def language_run(self, user_input, environment_description_list: List[Entity]):
        prompt = self.prompt_generator.run(user_input, environment_description_list)
        # Initialize the LLM model
        print("Starting LLM")
        self.llm = LLM(
            model_name=self.llm_name,
            temperature=self.llm_temperature,
            provider=self.llm_provider,
            is_chat=self.llm_is_chat,
            llm_is_vlm=self.llm_is_vlm,
        )
        print(f"Generated Prompt:\n{self.llm.prompt_system.format(content=prompt)}")
        llm_output = self.llm.run(prompt)
        # Reset the LLM model to free up GPU memory
        self.llm = None
        torch.cuda.empty_cache()
        print(f"LLM Response:\n{llm_output}")
        self.action_run(llm_output, environment_description_list)
        print(f"Generated actions:\n{self.get_actions()}")
        return self.get_actions(), llm_output
    
    def action_run(self, llm_output: str, environment_description_list: List[Entity]):
        self.action.run(
            llm_output,
            environment_description_list,
        )

    def action_tracking(self, environment_description_list: List[Entity]):
        return self.action.action_tracking(environment_description_list)
    
    def get_current_action(self):
        return self.action.robot_action_class.current_action()
    
    def action_step_success(self):
        return self.action.robot_action_class.action_step_success()
    
    def get_actions(self):
        return self.action.robot_action_class.action_names
    
    def all_actions_finished(self):
        return self.action.robot_action_class.all_actions_finished
    
    def action_counter(self):
        return self.action.robot_action_class.action_step, self.action.robot_action_class.current_action_step
    
    def set_robot_pose(self, pose: List[float]):
        self.perception.robot_camera.robot_last_pose = pose

    def get_readable_actions(self):
        return self.action.robot_action_class.get_readable_actions()
    
    def get_readable_current_low_level_action(self):
        return self.action.robot_action_class.get_readable_current_low_level_action()
