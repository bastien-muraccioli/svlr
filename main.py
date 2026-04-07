import json
import os

from controller.controller import Controller
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
        self.frequency = 40.0  # Hz
        self.time_step = 1.0 / self.frequency
        # self.action_counter = 0
        # self.action_size = 0

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
        self.web_controller_mode = False

        if self.args.simulation:
            self.simulation_mode = True
        elif self.args.ros_publisher and self.args.ros_subscriber:
            self.ros_controller_mode = True
        elif self.args.http_server is not None:
            self.web_controller_mode = True
        else:
            print("""
            Please provide either:
            --simulation to run in simulation mode, or
            --server to run in web server mode, or
            --ros_publisher and --ros_subscriber to run with ROS2.
            """)
            return
        
        self.node = None
        if self.ros_controller_mode:
            self.node = self.init_ros_node()
        

        if self.simulation_mode:
            from controller.simulation_controller import SimulationController
            print("Running in simulation mode")
            robot_controller = SimulationController()
        elif self.ros_controller_mode:
            from controller.ros_controller import RosRobotController
            print("Running in ROS2 controller mode")
            robot_controller = RosRobotController(self.node, camera_topic="" if self.args.use_camera_without_ros else self.args.camera_topic)
        elif self.web_controller_mode:
            from controller.web_controller import WebRobotController
            robot_controller = WebRobotController(base_url=f"http://{self.args.http_server}:{self.args.port}")
        else:
            print("Please provide either --simulation or --server or --ros_publisher and --ros_subscriber arguments")
            return
        
        # Init ControlLoop
        self.controller = ControlLoop(self.args, node=self.node)
        
        self.robot_controller = Controller(self.controller, robot_controller=robot_controller)

    # -------------------------
    # Backend functions
    # -------------------------

    def process_llm_command(self, prompt):
        self.user_prompt = prompt
        skip_action_generation = False

        if not self.robot_controller.robot_is_idle:
            print("Robot is busy, please wait until the current actions are done")
            self.robot_controller.user_entered_prompt = False
            return self.llm_output, self.controller.get_current_action()

        if self.user_prompt.strip() == "":
            self.robot_controller.user_entered_prompt = False
            return "", ""
        
        if self.robot_controller.perception_pipeline_has_run is False:
            print("Please run the perception pipeline (VLM) first")
            self.robot_controller.user_entered_prompt = False
            return "", ""
        
        if self.user_prompt == self.last_user_prompt and self.robot_controller.user_entered_prompt:
            print("User prompt unchanged, skipping LLM processing")
            skip_action_generation = True

        self.last_user_prompt = self.user_prompt
        self.robot_controller.user_entered_prompt = True

        if not skip_action_generation:
            self.final_action, self.llm_output = self.controller.language_run(self.user_prompt, self.objects_found)
    
        self.robot_controller.language_pipeline_has_run = True
        print("LLM processing done")
        return self.llm_output, self.controller.get_readable_actions()
    
    def process_by_pass_llm_command(self, prompt):
        self.user_prompt = prompt
        self.llm_output = prompt

        self.controller.action_run(self.llm_output, self.objects_found)
        self.final_action = self.controller.get_actions()  
        self.robot_controller.language_pipeline_has_run = True
        return self.llm_output, self.controller.get_readable_actions()

    def process_vlm(self):
        self.perception_pipeline_is_running = True
        self.objects_found, self.vlm_output, self.frame_with_masks_and_centers = self.controller.perception_run(self.camera_frame)
        self.robot_controller.perception_pipeline_has_run = True
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

        elif self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation) or self.web_controller_mode:
            cap = cv2.VideoCapture(self.camera_device)

        while True:
            if self.args.camera_topic and self.ros_controller_mode and not self.args.use_camera_without_ros:
                self.camera_frame = self.node.get_camera_image_ros()
            elif self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation) or self.web_controller_mode:
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

            if self.robot_controller.perception_pipeline_has_run is False or self.perception_pipeline_is_running:
                self.frame_with_masks_and_centers = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
            elif self.robot_controller.perception_pipeline_has_run is True: # and not (self.simulation_mode and not self.args.use_camera_in_simulation):
                self.objects_found, self.frame_with_masks_and_centers = self.controller.perception.update_trackers(self.camera_frame)

            yield self.frame_with_masks_and_centers
        
        if self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation):
            cap.release()
                

    def get_robot_state(self):
        if self.robot_controller.end_action_received and not self.robot_controller.perception_pipeline_has_run:
            return "Waiting for running perception pipeline..."
        if not self.robot_controller.language_pipeline_has_run and self.robot_controller.perception_pipeline_has_run:
            return "Waiting for user command..."
        if not self.controller.all_actions_finished():
            return f"Robot action in progress: {self.action_progress}"
        if not self.robot_controller.end_action_received:
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

    def run(self):
        print("Starting SVLR...")
        self.robot_controller.control()

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
    parser.add_argument(
        "--http_server", type=str, default=None, help="Robot server address"
    )
    parser.add_argument("--port", type=int, default=65500, help="Robot server port")

    return parser.parse_args()

def main():
    args = parser_args()

    svlr = SVLR(args)
    svlr.gradio_interface()
    svlr.run()

if __name__ == "__main__":
    main()

    
