from tools.read_json import read_robot_json

import yaml
import numpy as np
import os

def pixel_to_camera_coordinates(robot_name: str, pixel_pose:list) -> list:
    robot_info = read_robot_json(robot_name)
    # Depth of the object in meters
    depth = robot_info["eye_to_hand"]["depth"]
    
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #[x, y, z, rx, ry, rz, rw]
    data = {
        "camera_matrix": {
            "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.yaml")
    # Check if the file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"File {yaml_path} not found, please calibrate the camera, and add the calibration.yaml file at the root of the project.")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
 
    camera_matrix = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)

    # Example pixel coordinates of the object in the image
    u = pixel_pose[0] # example x pixel coordinates
    v = pixel_pose[1] # example y pixel coordinates 

    # Convert pixel coordinates to normalized image coordinates
    x_norm = (u - camera_matrix[0][2]) / camera_matrix[0][0]
    y_norm = (v - camera_matrix[1][2]) / camera_matrix[1][1]

    # Direction vector in camera coordinates
    direction_vector = np.array([x_norm, y_norm, 1], dtype=float)

    # Convert direction vector to 3D position in camera coordinates
    position_camera = direction_vector * depth
    pose[:len(position_camera)] = position_camera

    return pose

def camera_to_robot(robot_name: str, camera_pose: list) -> list:
    robot_info = read_robot_json(robot_name)
    robot_init_pose = robot_info["init_pose"]["pos_end_effector"]            
    robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #[x, y, z, rx, ry, rz, rw]
    # Convert the camera pose to robot pose
    robot_pose[0] = camera_pose[0] + robot_init_pose[0] + robot_info["eye_to_hand"]["dx"] # x
    robot_pose[1] = -camera_pose[1] + robot_init_pose[1] + robot_info["eye_to_hand"]["dy"] # y
    # For the moment we fix the Z manually (no depth with camera), so we keep the same Z
    robot_pose[2] = camera_pose[2] + robot_info["eye_to_hand"]["dz"] # z
    
    # Copy the orientation from the robot initial pose
    robot_pose[3:] = robot_init_pose[3:]
    
    return robot_pose

def pixel_to_robot(robot_name: str, pixel_pose: list) -> list:
    camera_pose = pixel_to_camera_coordinates(robot_name, pixel_pose)
    robot_pose = camera_to_robot(robot_name, camera_pose)
    return robot_pose