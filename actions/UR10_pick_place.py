from tools.read_json import read_robot_json

def pick_and_place(pick_pos: list, place_pos: list):
    robot_info = read_robot_json("UR10")
    init_pose = robot_info["init_pose"]["pos_end_effector"]
    gripper_open = robot_info["gripper"]["open"]
    gripper_close = robot_info["gripper"]["close"]
    quaternion = init_pose[3:]
    z_max = init_pose[2]
    z_min = -0.2673
    
    pick_pos_high = pick_pos[:]
    pick_pos_high[2] = z_max
    pick_pos_low = pick_pos[:]
    pick_pos_low[2] = z_min

    place_pos_high = place_pos[:]
    place_pos_high[2] = z_max
    place_pos_low = place_pos[:]
    place_pos_low[2] = z_min


    action_dict = [
        {
            "pos_end_effector": [*pick_pos_high, *quaternion],
            "gripper": gripper_open
        },{
            "pos_end_effector": [*pick_pos_low, *quaternion],
            "gripper": gripper_close
        },{
            "pos_end_effector": [*pick_pos_high, *quaternion],
            "gripper": gripper_close
        },{
            "pos_end_effector": [*place_pos_high, *quaternion],
            "gripper": gripper_close
        },{
            "pos_end_effector": [*place_pos_low, *quaternion],
            "gripper": gripper_open
        },{
            "pos_end_effector": [*place_pos_high, *quaternion],
            "gripper": gripper_open
        },
        ]
    
    return action_dict