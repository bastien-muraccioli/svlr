import json
import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch


from platform import node
import time
import cv2
import gradio as gr

import argparse

from control_loop import ControlLoop
from tools.read_camera import get_camera_image, get_camera_image_ros
from tools.read_json import read_robot_json

from transformers import logging
logging.set_verbosity_error()


class SVLR:
    """Scalable Visual Language Robotics (SVLR) Interface"""

    def __init__(self, args):
        self.args = args
        self.camera_device = args.camera_device

        self.perception_pipeline_is_running = False
        self.perception_pipeline_has_run = False
        self.language_pipeline_has_run = False
        self.end_action_received = True
        self.frequency = 40.0  # Hz
        self.time_step = 1.0 / self.frequency
        # self.action_counter = 0
        # self.action_size = 0
        self.user_entered_prompt = False
        self.robot_is_idle = True
        self.robot_searching_for_entity = False
        self.max_try_to_find_entity = 5
        self.counter_try_to_find_entity = 0
        self.entity_to_find = None

        self.user_prompt = ""
        self.last_user_prompt = ""
        self.llm_output = ""
        self.final_action = []
        self.last_final_action = {}
        self.vlm_output = ""
        self.objects_found = ""
        self.camera_frame = None
        self.frame_with_masks_and_centers = None
        self.action_progress = ""

        self.simulation_mode = False
        self.ros_controller_mode = False

        if self.args.simulation:
            self.simulation_mode = True
        elif self.args.ros_publisher and self.args.ros_subscriber:
            self.ros_controller_mode = True
        else:
            print("""
            Please provide either:
            --simulation to run in simulation mode, or
            --ros_publisher and --ros_subscriber to run with ROS2.
            """)
            return
        
        self.node = None
        if self.ros_controller_mode:
            self.node = self.init_ros_node()
        
        # Init ControlLoop
        self.controller = ControlLoop(self.args, node=self.node)

    # -------------------------
    # Backend functions
    # -------------------------

    def init_ros_node(self):
        import rclpy
        from tools.ros_pubsub import RosPubSub
        # Init rclpy
        rclpy.init()

        # Initial pose of the robot
        init_pose = [read_robot_json(self.args.robot_name)["init_pose"]]
        print(f"Initial pose: {init_pose}")

        # Folder to save captured images
        image_folder_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "captured_image"
        )
        os.makedirs(image_folder_path, exist_ok=True)

        camera_topic = self.args.camera_topic
        if self.args.use_camera_without_ros:
            camera_topic = ""

        # Initialize ROS pub-sub node
        node = RosPubSub(self.args.ros_publisher, self.args.ros_subscriber, camera_topic)

        print(f"Publisher topic: {self.args.ros_publisher}")
        print(f"Subscriber topic: {self.args.ros_subscriber}")
        print(f"Camera topic: {camera_topic}")
        print("ROS2 node initialized")
        return node

    def process_llm_command(self, prompt):
        self.user_prompt = prompt
        skip_action_generation = False

        if not self.robot_is_idle:
            print("Robot is busy, please wait until the current actions are done")
            self.user_entered_prompt = False
            return self.llm_output, self.controller.get_current_action()

        if self.user_prompt.strip() == "":
            self.user_entered_prompt = False
            return "", ""
        
        if self.perception_pipeline_has_run is False:
            print("Please run the perception pipeline (VLM) first")
            self.user_entered_prompt = False
            return "", ""
        
        if self.user_prompt == self.last_user_prompt and self.user_entered_prompt:
            print("User prompt unchanged, skipping LLM processing")
            skip_action_generation = True

        self.last_user_prompt = self.user_prompt
        self.user_entered_prompt = True

        if not skip_action_generation:
            self.final_action, self.llm_output = self.controller.language_run(self.user_prompt, self.objects_found)
    
        self.language_pipeline_has_run = True
        print("LLM processing done")
        return self.llm_output, self.controller.get_readable_actions()
    
    def process_by_pass_llm_command(self, prompt):
        self.user_prompt = prompt
        self.llm_output = prompt

        self.controller.action_run(self.llm_output, self.objects_found)
        self.final_action = self.controller.get_actions()  
        self.language_pipeline_has_run = True
        return self.llm_output, self.controller.get_readable_actions()

    def process_vlm(self):
        self.perception_pipeline_is_running = True
        self.objects_found, self.vlm_output, self.frame_with_masks_and_centers = self.controller.perception_run(self.camera_frame)
        self.perception_pipeline_has_run = True
        print("VLM processing done")
        self.perception_pipeline_is_running = False
        return self.vlm_output, self.objects_found

    # -------------------------
    # Camera generator
    # -------------------------
    def generate_frames(self):
        if self.simulation_mode and not self.args.use_camera_in_simulation:
            # Path to the image used for simulation
            simulation_image_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pictures",
                self.args.simulation_image_file,
            )
            self.camera_frame = cv2.imread(simulation_image_path)

        elif self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation):
            cap = cv2.VideoCapture(self.camera_device)

        while True:
            if self.args.camera_topic and self.ros_controller_mode and not self.args.use_camera_without_ros:
                self.camera_frame = self.node.get_camera_image_ros()
            elif self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation):
                ret, self.camera_frame  = cap.read()
                if not ret:
                    break
            elif self.simulation_mode:
                # self.camera_frame = cv2.imread(simulation_image_path)
                time.sleep(0.1)
            else:
                print("No camera source available")
                break

            if self.camera_frame is None:
                print("Failed to get image. Skipping iteration.")
                continue

            if self.perception_pipeline_has_run is False or self.perception_pipeline_is_running:
                self.frame_with_masks_and_centers = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
            elif self.perception_pipeline_has_run is True: # and not (self.simulation_mode and not self.args.use_camera_in_simulation):
                self.objects_found, self.frame_with_masks_and_centers = self.controller.perception.update_trackers(self.camera_frame)

            yield self.frame_with_masks_and_centers
        
        if self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation):
            cap.release()
                

    def get_robot_state(self):
        if self.end_action_received and not self.perception_pipeline_has_run:
            return "Waiting for running perception pipeline..."
        if not self.language_pipeline_has_run and self.perception_pipeline_has_run:
            return "Waiting for user command..."
        if not self.controller.all_actions_finished():
            return f"Robot action in progress: {self.action_progress}"
        if not self.end_action_received:
            return f"Robot executing actions = {self.final_action}"
        return "Idle"
    

    # -------------------------
    # Build Gradio layout
    # -------------------------
    def gradio_interface(self):
        with gr.Blocks(title="Scalable Visual Language Robotics (SVLR)") as demo:
            timer = gr.Timer()
            
            gr.Markdown("# Scalable Visual Language Robotics (SVLR)")

            with gr.Row():
                # --- LEFT SIDE (LLM) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Language Reasoning")
                    self.user_prompt = gr.Textbox(
                        label="User Command",
                        placeholder="e.g., Pick up the red cup / [{\"action\": \"move_to\", \"parameters\": [\"person\"]}]"
                    )
                    run_llm_button = gr.Button("Run LLM 🔄")
                    by_pass_button = gr.Button("Bypass LLM ⏭️ (user command = llm output)")
                    llm_output = gr.Textbox(label="LLM Output", interactive=False, lines=3)
                    final_command = gr.Textbox(label="System Final Command", interactive=False, lines=3)
                    gr.Markdown("### 🤖 Robot Control")
                    self.robot_state = gr.Textbox(label="robot action state", interactive=False, value=self.get_robot_state, every=timer, lines=3)
                    

                # --- RIGHT SIDE (VLM) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 👁️ Visual Perception")
                    video_display = gr.Image(label="Live Camera", streaming=True)
                    run_vlm_button = gr.Button("Run VLM 🔍")
                    vlm_output = gr.Textbox(label="VLM Output", interactive=False, lines=3)
                    env_objects = gr.Textbox(label="Detected Objects", interactive=False, lines=3)

            # ---- FUNCTIONAL CONNECTIONS ----
            run_llm_button.click(
                fn=self.process_llm_command,
                inputs=self.user_prompt,
                outputs=[llm_output, final_command],
                show_progress=True,
            )

            by_pass_button.click(
                fn=self.process_by_pass_llm_command,
                inputs=self.user_prompt,
                outputs=[llm_output, final_command],
                show_progress=True,
            )

            run_vlm_button.click(
                fn=self.process_vlm,
                inputs=None,
                outputs=[vlm_output, env_objects],
                show_progress=True,
            )
            # timer = gr.Timer(0.5, True, True)
            # demo.load(fn=self.get_robot_state, inputs=None, outputs=self.robot_state, every=timer)
            # Stream camera
            demo.load(self.generate_frames, None, video_display)

        demo.launch(prevent_thread_lock=True)

    def simulation_controller(self):
        while True:
            time.sleep(0.05)
            if self.user_prompt == "stop":
                # Quit the program
                return
            if self.language_pipeline_has_run and self.final_action != self.last_final_action and self.end_action_received:
                print(f"Actions: {self.final_action}")
                self.last_final_action = self.final_action
                self.end_action_received = False
                self.language_pipeline_has_run = False
                self.robot_is_idle = False
                time.sleep(1) # Simulate time taken to execute actions
                self.robot_is_idle = True
                self.end_action_received = True
                print("Actions executed")

    def ros_controller(self):
        import rclpy
        try:
            while rclpy.ok():
                # rate.sleep()
                time.sleep(self.time_step)

                if self.user_prompt == "stop":
                    # Quit the program
                    return

                if not self.language_pipeline_has_run:
                    continue

                #  Initialize the action sending if we received end of action and the robot is idle
                if  self.end_action_received and self.robot_is_idle:
                    print("Ros2 controller ready to synchronize and send actions")
                    self.robot_is_idle = False
                    self.action_size, self.action_counter = self.controller.action_counter()
                    print(f"Action size: {self.action_size}, Action counter: {self.action_counter+1}")
                    self.node.reset_end_action()
                    self.end_action_received = False

                #  Check if we received end of action from ROS2 subscriber
                # if not self.end_action_received:
                if not self.robot_searching_for_entity and not self.end_action_received:
                    rclpy.spin_once(self.node, timeout_sec=self.time_step)
                    if self.node.end_action_received():
                        print("End of action received from ROS2 subscriber")
                        self.end_action_received = True
                        self.node.reset_end_action()
                        self.action_size, self.action_counter = self.controller.action_counter()
                        print(f"Action size: {self.action_size}, Action counter: {self.action_counter+1}")
                        # Tell the controller that the current action is done
                        # This increments the action step and
                        # returns True if all actions are done
                        all_steps_successful = self.controller.action_step_success()
                        print("Ready for next action")
                        if all_steps_successful:
                            print("All actions are done")
                            self.controller.perception.reset_all_session()                                
                            self.language_pipeline_has_run = False
                            self.robot_is_idle = True
                            self.end_action_received = True
                            self.perception_pipeline_has_run = False
                            continue

                # Update robot pose if not idle
                if not self.robot_is_idle:
                    robot_pose = self.node.get_robot_pose(timeout_sec=0.1)
                    # print(f"Robot current pose: {robot_pose}")
                    if robot_pose:
                        self.controller.set_robot_pose(robot_pose)

                if not self.robot_searching_for_entity:
                    action_is_tracked, self.objects_found = self.controller.action_tracking(self.objects_found)
                    for entity in self.objects_found:
                        if entity.need_to_be_tracked and not entity.tracked:
                            print(f"Entity {entity.name} needs to be tracked but is not, trying to find it again...")
                            self.entity_to_find = entity
                            self.counter_try_to_find_entity = 0
                            self.robot_searching_for_entity = True
                            self.node.reset_end_action()
                            self.end_action_received = True
                            self.controller.action.robot_action_class.current_action_step -= 1  # Retry the current action
                            break
                    if not self.robot_searching_for_entity:
                        # Update the current action only if we are not searching for an entity
                        self.final_action = self.controller.get_current_action()
                        self.end_action_received = False
                else:
                    self.node.reset_end_action()
                    self.end_action_received = True
                    if self.counter_try_to_find_entity < self.max_try_to_find_entity:
                        print(f"Trying to find entity {self.entity_to_find.name} ({self.counter_try_to_find_entity+1}/{self.max_try_to_find_entity})...")
                        entity_found = self.controller.perception.segment_one_entity(entity.name)
                        if entity_found:
                            self.objects_found = self.controller.perception.environment_description_list
                            print(f"Entity {entity.name} found again.")
                            self.robot_searching_for_entity = False
                            self.entity_to_find = None
                            self.counter_try_to_find_entity = 0
                        else:
                            self.counter_try_to_find_entity += 1

               
                self.action_size, self.action_counter = self.controller.action_counter()
                self.action_progress = f"Sending action {self.action_counter + 1}/{self.action_size}:\n {self.controller.get_readable_current_low_level_action()}"
                self.node.send_actions(self.final_action)                

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Shutting down.")
        finally:
            print("Shutting down ROS2 node.")
            self.node.destroy_node()
            rclpy.shutdown()

    def run(self):
        print("Starting SVLR...")
        if self.simulation_mode:
            print("Running in simulation mode")
            self.simulation_controller()
        elif self.ros_controller_mode:
            print("Running in ROS2 controller mode")
            self.ros_controller()
        else:
            print("Please provide either --simulation or --ros_publisher and --ros_subscriber arguments")
            return

def parser_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the SVLR control loop")

    parser.add_argument(
        "--robot_name", type=str, default="UR10", help="Name of the robot"
    )

    # ROS topics
    parser.add_argument(
        "--ros_publisher",
        type=str,
        default="",
        help="Name of the ros2 topic to publish SVLR actions",
    )
    parser.add_argument(
        "--ros_subscriber",
        type=str,
        default="",
        help="Name of the subscriber ros2 topic to handle of actions",
    )

    # Camera
    parser.add_argument("--camera_topic", type=str, default="/camera/camera/color/image_raw", help="Camera ros2 topic")
    parser.add_argument(
        "--camera_device", type=str, default="/dev/video0", help="Camera device, you can also use video file path"
    )
    parser.add_argument("--camera_width", type=int, default=640, help="Camera width")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera height")

    # LLM
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gemma3n:e4b",
        help="LLM name",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="Ollama",
        help="LLM provider: Ollama, HuggingFace or OpenAI",
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.1,
        help="LLM temperature: float between 0.1 and 1.0",
    )
    parser.add_argument(
        "--llm_is_chat", action="store_true", help="The LLM is a Chat model"
    )

    # VLM
    parser.add_argument(
        "--vlm_name",
        type=str,
        default="granite3.2-vision",
        help="VLM name",
    )
    parser.add_argument(
        "--vlm_provider",
        type=str,
        default="Ollama",
        help="VLM provider: Only Ollama for now",
    )

    # Simulation
    parser.add_argument(
        "--simulation", action="store_true", help="Run in simulation mode"
    )

    parser.add_argument(
        "--use_camera_in_simulation", action="store_true", help="Use camera in simulation mode instead of a static image"
    )
    parser.add_argument(
        "--use_camera_without_ros", action="store_true", help="Force using camera device instead of ROS2 topic"
    )
    parser.add_argument(
        "--use_depth_camera", action="store_true", help="Use depth camera info instead of fixed depth"
    )

    parser.add_argument(
        "--simulation_image_file",
        type=str,
        default="test.png",
        help="Simulation image file",
    )

    return parser.parse_args()

def main():
    args = parser_args()

    svlr = SVLR(args)
    svlr.gradio_interface()
    svlr.run()

if __name__ == "__main__":
    main()

    
