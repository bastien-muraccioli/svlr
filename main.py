from control_loop import ControlLoop
from tools.read_camera import get_camera_image
from tools.read_json import read_robot_json

import argparse
import json
import socket
import cv2
import time
from datetime import datetime
import os

from transformers import logging
logging.set_verbosity_error()



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the robot control loop')
    
    # Robot and server info
    parser.add_argument('--robot_name', type=str, default="UR10", help='Name of the robot')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='Robot server address')
    parser.add_argument('--port', type=int, default=65500, help='Robot server port')
    parser.add_argument('--buffer', type=int, default=1024, help='Server buffer size')

    # Camera
    parser.add_argument('--camera_device', type=str, default='/dev/video2', help='Camera device')
    parser.add_argument('--camera_width', type=int, default=640, help='Camera width')
    parser.add_argument('--camera_height', type=int, default=480, help='Camera height')

    # LLM
    parser.add_argument('--llm_name', type=str, default='microsoft/Phi-3-mini-4k-instruct', help='LLM name')
    parser.add_argument('--llm_provider', type=str, default='HuggingFace', help='LLM provider: HuggingFace or OpenAI')
    parser.add_argument('--llm_temperature', type=float, default=0.1, help='LLM temperature: float between 0.1 and 1.0')
    parser.add_argument('--llm_is_chat', action='store_true', help='The LLM is a Chat model')

    # Simulation
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--simulation_image_file', type=str, default='test.png', help='Simulation image file')

    # Image
    parser.add_argument('--show_image', action='store_true', help='Show the captured image')
    parser.add_argument('--save_image', action='store_true', help='Save the captured image')
    

    args = parser.parse_args()
    
    if not args.simulation:
        real_controller(args)
    else:
        simulation_controller(args)
        

def real_controller(args: argparse.Namespace):

    # Initial pose of the robot, expressed in the camera's frame
    # pos = [*pos, *quaternion]
    # pos = [x, y, z, rx, ry, rz ,rw]
    init_pose = [read_robot_json(args.robot_name)["init_pose"]]
    print(f"Initial pose: {init_pose}")
    init_pose_json = json.dumps(init_pose)
    print(f"Initial pose json: {init_pose_json}")
    
    # Create a folder to save the captured images
    image_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captured_image')

    # Init ControlLoop
    controller = ControlLoop(args)

    # Define the server address, port and buffer
    server_address = (args.server, args.port)
    buffer_size = args.buffer

    #Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        # Connect to the server
        client_socket.connect(server_address)
        
        # Move the robot to the initial position
        robot_init_position(client_socket, init_pose_json, buffer_size)
    
        while True:
            # Get image from camera
            image_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            image = get_camera_image(device=args.camera_device, width=args.camera_width, height=args.camera_height)
            if args.show_image:
                # Display the captured image
                cv2.imshow('Captured Image', image)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()

            # Optionally save the image to a file
            if args.save_image:
                cv2.imwrite(os.path.join(image_folder_path, f'captured_image_{image_name}.png'), image)

            print("Write 'stop' if you want to stop the program")
            user_input = input("User input: ")
            if user_input == "stop":
                # Quit the program
                return
            # Get a more recent image
            image = get_camera_image(device=args.camera_device, width=args.camera_width, height=args.camera_height)
            # Generate robot action based on the image
            action_dict_list = controller.run(image, user_input)
            if action_dict_list is None:
                print("No action generated")
                continue

            # Send the action to the server
            robot_make_action(client_socket, action_dict_list, buffer_size)

            # Go back to the initial position
            robot_init_position(client_socket, init_pose_json, buffer_size)

def simulation_controller(args: argparse.Namespace):
    # Path to the image used for simulation
    simulation_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pictures', args.simulation_image_file)
    image = cv2.imread(simulation_image_path)

    # Init ControlLoop
    controller = ControlLoop(args)
    while True:
        # Generate robot action based on the image
        if args.show_image:
            cv2.imshow("Loaded Image", image)
            cv2.waitKey(2000)  # 0 means wait indefinitely
            cv2.destroyAllWindows()
        print("Write 'stop' if you want to stop the program")
        user_input = input("User input: ")
        if user_input == "stop":
            # Quit the program
            return
        action_dict = controller.run(image, user_input)
        if action_dict is None:
            print("No action generated")
            break
        print(f"Action: {action_dict}")

def send_data(client_socket: socket.socket, data: list):
    json_data = json.dumps(data)
    client_socket.sendall(json_data.encode('utf-8'))
    print(f"Sent: {json_data}")

def receive_data(client_socket: socket.socket, buffer: int) -> list:
    data = client_socket.recv(buffer).decode('utf-8')
    print(f"Received: {data}")
    return data

def robot_init_position(client_socket: socket.socket, init_pose_json: str, buffer: int):
    print(f"Robot moving to init pose")
    client_socket.sendall(init_pose_json.encode('utf-8'))
    time.sleep(0.1)
    # Wait for the server's response
    receive_data(client_socket, buffer)

def robot_make_action(client_socket: socket.socket, action: list, buffer: int):
    print(f"Robot is doing the action")
    send_data(client_socket, action)
    # Wait for the server's response
    receive_data(client_socket, buffer)

if __name__ == '__main__':
    main()
