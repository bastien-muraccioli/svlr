import argparse
from tools.read_json import read_robot_json

import yaml
import numpy as np
import os

class RobotCamera:
    def __init__(self, args: argparse.Namespace, node=None):
        self.robot_name = args.robot_name
        self.use_camera_without_ros = args.use_camera_without_ros
        self.depth_camera_flag = args.use_depth_camera 
        self.ros_depth = 0.0
        self.robot_info = read_robot_json(self.robot_name)
        self.robot_last_pose = self.robot_info["init_pose"]["pos_end_effector"]
        if not self.use_camera_without_ros:
            self.node = node  # ROS node will be set in main.py if using ROS
        else:
            self.node = None

        data = {"camera_matrix": {"data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.yaml"
        )
        # Check if the file exists
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"File {yaml_path} not found, please calibrate the camera, and add the calibration.yaml file at the root of the project."
            )

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.camera_matrix = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)
        self.dist_coeffs = np.array(
            data["distortion_coefficients"]["data"], dtype=float
        ).reshape(5)

    def pixel_to_camera_coordinates(self, pixel_pose: list) -> list:
        if self.node:
            # print("Getting 3D point from ROS pointcloud...")
            pc_msg = self.node.get_pointcloud_message(timeout_sec=0.1)
            if pc_msg:
                # print("3D point from ROS pointcloud:")
                point = self.node.get_3d_point_from_pixel(pc_msg, pixel_pose[0], pixel_pose[1])
                if point:
                    camera_pose = point
                    self.ros_depth = point[2]
                    # print(f"3D point from ROS pointcloud: {camera_pose}")
                    self.depth_camera_flag = True
                    return camera_pose

        # Depth of the object in meters
        if self.depth_camera_flag:
            depth = self.ros_depth
        else:
            depth = self.robot_info["eye_to_hand"]["depth"]

        k1, k2, p1, p2, k3 = self.dist_coeffs

        # Example pixel coordinates of the object in the image
        u = pixel_pose[0]  # example x pixel coordinates
        v = pixel_pose[1]  # example y pixel coordinates

        # Convert pixel coordinates to normalized image coordinates
        x_norm = (u - self.camera_matrix[0][2]) / self.camera_matrix[0][0]
        y_norm = (v - self.camera_matrix[1][2]) / self.camera_matrix[1][1]

        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2 * r4

        # Radial distortion
        distortion = 1 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        delta_x = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
        delta_y = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm

        # Apply distortion
        x_undistorted = (x_norm - delta_x) / distortion
        y_undistorted = (y_norm - delta_y) / distortion

        # Convert back to pixel coordinates
        u_undistorted = self.camera_matrix[0][0] * x_undistorted + self.camera_matrix[0][2]
        v_undistorted = self.camera_matrix[1][1] * y_undistorted + self.camera_matrix[1][2]

        x = (u_undistorted - self.camera_matrix[0][2]) / self.camera_matrix[0][0]
        y = (v_undistorted - self.camera_matrix[1][2]) / self.camera_matrix[1][1]

        x_3d = x * depth
        y_3d = y * depth
        z_3d = depth

        return np.array([x_3d, y_3d, z_3d])


    def camera_to_robot(self, camera_pose: list) -> list:
        robot_init_pose = self.robot_info["init_pose"]["pos_end_effector"]
        robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, rx, ry, rz, rw]
        
        # Retrieve the rotation matrix from the robot info, or use identity if not provided
        robot_rotation_matrix = np.array(self.robot_info.get("rot_mat", np.eye(3)))  # Default to identity if not provided
        old_coords = np.array([camera_pose[0], camera_pose[1], camera_pose[2]])  # [X, Y, Z] in old frame
        camera_pose = robot_rotation_matrix @ old_coords
        
        # Convert the camera pose to robot pose
        x_camera_pose = camera_pose[0]
        x_robot_last_pose = self.robot_last_pose[0]
        robot_pose[0] = (
            x_camera_pose + x_robot_last_pose + self.robot_info["eye_to_hand"]["dx"]
        )  # x
        
        y_camera_pose = camera_pose[1]
        y_robot_last_pose = self.robot_last_pose[1]
        robot_pose[1] = (
            -y_camera_pose + y_robot_last_pose + self.robot_info["eye_to_hand"]["dy"]
        )  # y
        if self.depth_camera_flag:
            robot_pose[2] = (
                    camera_pose[2] - self.robot_last_pose[2]
                )  # z
        else:
            robot_pose[2] = - camera_pose[2] + self.robot_last_pose[2]  # z
            
        # For the moment we fix the Z manually (no depth with camera), so we keep the same Z
        # robot_pose[2] = camera_pose[2] + self.robot_info["eye_to_hand"]["dz"]  # z

        # Copy the orientation from the robot initial pose
        robot_pose[3:] = robot_init_pose[3:]

        return robot_pose


    def pixel_to_robot(self, pixel_pose: list) -> list:
        self.depth_camera_flag = False
        if self.node:
            print("Getting 3D point from ROS pointcloud...")
            pc_msg = self.node.get_pointcloud_message(timeout_sec=0.1)
            if pc_msg:
                print("3D point from ROS pointcloud:")
                point = self.node.get_3d_point_from_pixel(pc_msg, pixel_pose[0], pixel_pose[1])
                if point:
                    camera_pose = point
                    self.ros_depth = point[2]
                    print(f"3D point from ROS pointcloud: {camera_pose}")
                    self.depth_camera_flag = True
                else:
                    camera_pose = self.pixel_to_camera_coordinates(pixel_pose)
            else:
                    camera_pose = self.pixel_to_camera_coordinates(pixel_pose)
        else:
            print("No ROS node available, using camera model for 3D point computation.")
            camera_pose = self.pixel_to_camera_coordinates(pixel_pose)

        robot_pose = self.camera_to_robot(camera_pose)
        return robot_pose
