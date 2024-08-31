from tools.read_json import read_robot_json

def open_gripper():
    robot_info = read_robot_json("UR10")
    action_dict = [
        {
            "pos_end_effector": robot_info["init_pose"]["pos_end_effector"],
            "gripper": robot_info["gripper"]["open"]
        }]
    return action_dict

def close_gripper():
    robot_info = read_robot_json("UR10")
    action_dict = [
        {
            "pos_end_effector": robot_info["init_pose"]["pos_end_effector"],
            "gripper": robot_info["gripper"]["close"]
        }]
    return action_dict