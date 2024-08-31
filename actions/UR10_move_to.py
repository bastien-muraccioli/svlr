from tools.read_json import read_robot_json

def move_to(pos: list):
    robot_info = read_robot_json("UR10")
    robot_init_pose = robot_info["init_pose"]["pos_end_effector"]
    quaternion = robot_init_pose[3:]

    
    move_pos = pos[:]
    move_pos[2] = robot_init_pose[2] # z high


    action_dict = [
        {
            "pos_end_effector": [*move_pos, *quaternion],
            "gripper": robot_info["gripper"]["close"]
        }]
    
    return action_dict