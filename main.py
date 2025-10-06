import json
from platform import node
import time
import cv2
import gradio as gr

import argparse

from control_loop import ControlLoop
from tools.read_camera import get_camera_image, get_camera_image_ros
from tools.read_json import read_robot_json

import cv2
import time
import os





# from transformers import logging

# logging.set_verbosity_error()

import time
import cv2
import gradio as gr

class SVLR:
    """Scalable Visual Language Robotics (SVLR) Interface"""

    def __init__(self, args):
        self.args = args
        self.camera_device = args.camera_device

        self.perception_pipeline_has_run = False
        self.language_pipeline_has_run = False
        self.user_entered_prompt = False

        self.user_prompt = ""
        self.last_user_prompt = ""
        self.llm_output = ""
        self.final_action = {}
        self.last_final_action = {}
        self.vlm_output = ""
        self.objects_found = ""
        self.camera_frame = None
        self.frame_with_masks_and_centers = None

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
        self.controller = ControlLoop(self.args)

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

        # Initialize ROS pub-sub node
        node = RosPubSub(self.args.ros_publisher, self.args.ros_subscriber)

        print(f"Publisher topic: {self.args.ros_publisher}")
        print(f"Subscriber topic: {self.args.ros_subscriber}")
        print(f"Camera topic: {self.args.camera_topic}")
        print("ROS2 node initialized")
        return node

    def process_llm_command(self, prompt):
        self.user_prompt = prompt

        if self.user_prompt.strip() == "":
            self.user_entered_prompt = False
            return "", ""

        if self.user_prompt == self.last_user_prompt and self.user_entered_prompt:
            print("User prompt unchanged, skipping LLM processing")
            self.user_entered_prompt = False
            return self.llm_output, self.controller.action.action_and_parameters_in_semantic
        
        if self.perception_pipeline_has_run is False:
            print("Please run the perception pipeline (VLM) first")
            self.user_entered_prompt = False
            return "", ""

        self.last_user_prompt = self.user_prompt
        self.user_entered_prompt = True

        self.final_action, self.llm_output = self.controller.language_run(self.user_prompt, self.objects_found)
        self.language_pipeline_has_run = True
        print("LLM processing done")
        return self.llm_output, json.dumps(self.controller.action.action_and_parameters_in_semantic)


    def process_vlm(self):
        self.objects_found, self.vlm_output, self.frame_with_masks_and_centers = self.controller.perception_run(self.camera_frame)
        self.perception_pipeline_has_run = True
        print("VLM processing done")
        return self.vlm_output, ", ".join(self.objects_found)

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
            # time.sleep(0.033) # ~30 FPS

            if self.args.camera_topic and self.ros_controller_mode:
                self.camera_frame = self.node.get_camera_image_ros(topic=self.args.camera_topic)
            elif self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation):
                ret, self.camera_frame  = cap.read()
                if not ret:
                    break

            if self.camera_frame is None:
                print("Failed to get image. Skipping iteration.")
                continue

            if self.perception_pipeline_has_run is False:
                self.frame_with_masks_and_centers = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
            elif self.perception_pipeline_has_run is True and not (self.simulation_mode and not self.args.use_camera_in_simulation):
                self.objects_found, self.frame_with_masks_and_centers = self.controller.perception.update_trackers(self.camera_frame)

            yield self.frame_with_masks_and_centers
        
        if self.args.use_camera_without_ros or (self.simulation_mode and self.args.use_camera_in_simulation):
            cap.release()
                

    # -------------------------
    # Build Gradio layout
    # -------------------------
    def gradio_interface(self):
        with gr.Blocks(title="Scalable Visual Language Robotics (SVLR)") as demo:
            gr.Markdown("# Scalable Visual Language Robotics (SVLR)")

            with gr.Row():
                # --- LEFT SIDE (LLM) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Language Reasoning")
                    self.user_prompt = gr.Textbox(
                        label="User Command",
                        placeholder="e.g., Pick up the red cup"
                    )
                    run_llm_button = gr.Button("Run LLM 🔄")
                    llm_output = gr.Textbox(label="LLM Output", interactive=False)
                    final_command = gr.Textbox(label="System Final Command", interactive=False)

                # --- RIGHT SIDE (VLM) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 👁️ Visual Perception")
                    video_display = gr.Image(label="Live Camera", streaming=True)
                    run_vlm_button = gr.Button("Run VLM 🔍")
                    vlm_output = gr.Textbox(label="VLM Output", interactive=False)
                    env_objects = gr.Textbox(label="Detected Objects", interactive=False)

            # ---- FUNCTIONAL CONNECTIONS ----
            run_llm_button.click(
                fn=self.process_llm_command,
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

            # Stream camera
            demo.load(self.generate_frames, None, video_display)

        demo.launch(prevent_thread_lock=True)

    def simulation_controller(self):
        while True:
            time.sleep(0.05)
            if self.user_prompt == "stop":
                # Quit the program
                return
            if self.language_pipeline_has_run and self.final_action != self.last_final_action:
                print(f"Actions: {self.final_action}")
                self.last_final_action = self.final_action
                self.language_pipeline_has_run = False

    def ros_controller(self):
        try:
            import rclpy
            while rclpy.ok():
                time.sleep(0.05)
                if self.user_prompt == "stop":
                    # Quit the program
                    return
            
                # Send actions
                if self.language_pipeline_has_run and self.final_action != self.last_final_action:
                    print(f"Sending actions to ROS2 topic {self.args.ros_publisher}")
                    self.last_final_action = self.final_action
                    self.node.send_actions(self.final_action)
                    self.language_pipeline_has_run = False

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Shutting down.")
        finally:
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

    
