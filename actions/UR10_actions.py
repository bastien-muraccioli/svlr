from tools.read_json import read_robot_json
from src.entity import Entity
from actions.robot_actions import RobotActions

class UR10Actions(RobotActions):
    def __init__(self):
        super().__init__()
        self.robot_name = "UR10"
        self.robot_info = read_robot_json("UR10")
        self.init_pose = self.robot_info["init_pose"]["pos_end_effector"] # [x, y, z, qx, qy, qz, qw]
        self.gripper_open = self.robot_info["gripper"]["open"]
        self.gripper_close = self.robot_info["gripper"]["close"]
        self.quaternion = self.init_pose[3:]
        self.z_max = self.init_pose[2]
        self.z_min = 0.07  # Minimum height to avoid collision with the table
        print(f"UR10Actions initialized with init_pose: {self.init_pose}, gripper_open: {self.gripper_open}, gripper_close: {self.gripper_close}")

    #----------------------------------------------------------------------------------------
    # All low-level actions return action_dict
    # Target position is relative to the robot base frame
    def low_level_go_high(self, entity: Entity, gripper_state):
        """Move the robot to a high position above the entity target."""
        # print("UR10 go_high")
        target_pos = entity.robot_frame_pos[:]
        target_pos[2] = self.z_max
        action_dict = [
            {"pos_end_effector": [*target_pos, *self.quaternion], "gripper": gripper_state}
        ]
        return action_dict

    def low_level_go_low(self, entity: Entity, gripper_state):
        """Move the robot to a low position above the entity target (x, y)."""
        # print("UR10 go_low")
        target_pos = entity.robot_frame_pos[:]
        if target_pos[2] < self.z_min:
            # print(f"Warning: target z {target_pos[2]} is below minimum {self.z_min}, adjusting to minimum.")
            target_pos[2] = self.z_min
        action_dict = [
            {"pos_end_effector": [*target_pos, *self.quaternion], "gripper": gripper_state}
        ]
        return action_dict
    
    def low_level_open_gripper(self):
        action_dict = [
            {
                "pos_end_effector": [0,0,0, *self.quaternion], 
                "gripper": self.gripper_open,
            }
        ]
        return action_dict

    def low_level_close_gripper(self):
        action_dict = [
            {
                "pos_end_effector": [0,0,0, *self.quaternion],
                "gripper": self.gripper_close,
            }
        ]
        return action_dict
    #----------------------------------------------------------------------------------------
    # High-level actions return a list of {"action": action_dict, "entity": entity}

    def pick_and_place(self, pick_entity: Entity, place_entity: Entity):
        low_level_actions = [
            {"function": "go_high", "params": [pick_entity, self.gripper_open], "entity": pick_entity, "tracking": True},
            {"function": "go_low", "params": [pick_entity, self.gripper_close], "entity": pick_entity, "tracking": False},
            {"function": "go_high", "params": [pick_entity, self.gripper_close], "entity": pick_entity, "tracking": False},
            {"function": "go_high", "params": [place_entity, self.gripper_close], "entity": place_entity, "tracking": False},
            {"function": "go_low", "params": [place_entity, self.gripper_open], "entity": place_entity, "tracking": False},
            {"function": "go_high", "params": [place_entity, self.gripper_open], "entity": place_entity, "tracking": False},
        ]
        return low_level_actions
    
    def pick(self, entity: Entity):
        low_level_actions = [
            {"function": "go_high", "params": [entity, self.gripper_open], "entity": entity, "tracking": True},
            {"function": "go_low", "params": [entity, self.gripper_close], "entity": entity, "tracking": False},
            {"function": "go_high", "params": [entity, self.gripper_close], "entity": entity, "tracking": False},
        ]
        return low_level_actions
    
    def place(self, entity: Entity):
        low_level_actions = [
            {"function": "go_high", "params": [entity, self.gripper_close], "entity": entity, "tracking": True},
            {"function": "go_low", "params": [entity, self.gripper_open], "entity": entity, "tracking": False},
            {"function": "go_high", "params": [entity, self.gripper_open], "entity": entity, "tracking": False},
        ]
        return low_level_actions
    
    def move_to(self, entity: Entity):
        low_level_actions = [
            {"function": "go_high", "params": [entity, self.gripper_open], "entity": entity, "tracking": True},
        ]
        return low_level_actions

    def open_gripper(self):
        low_level_actions = [
            {"function": "open_gripper", "params": [], "entity": None, "tracking": False},
        ]
        return low_level_actions

    def close_gripper(self):
        low_level_actions = [
            {"function": "close_gripper", "params": [], "entity": None, "tracking": False},
        ]
        return low_level_actions
