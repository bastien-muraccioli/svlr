
from src.entity import Entity


class RobotActions():
    def __init__(self):
        self.robot_name = ""
        self.robot_info = {}
        self.all_actions_finished = True
        self.current_action_step = 0
        self.action_step = 0
        self.robot_actions = [] # List of all low-level action results
        self.action_names = []  # List of action names for quick reference
        self.entities_involved = []  # List of arrays of entity names involved in actions
        self.low_level_actions = []  # List of dicts with "function" (low level action name), "params" and "entity" (entities involved in the low level action)

    def execute(self, function_name: str, *params):
        """
        Calls an action method defined in the robot subclass, 
        validating against the JSON definition.
        """
        # Check if action is defined in robot JSON
        action_entry = next((a for a in self.robot_info["actions"] if a["name"] == function_name), None)
        if not action_entry:
            print(f"❌ {self.robot_name}'s function '{function_name}' not found in JSON.")
            return None, None

        # Check that the method actually exists
        if not hasattr(self, function_name):
            print(f"❌ Function '{function_name}' not found in class '{type(self).__name__}'.")
            return None, None

        func = getattr(self, function_name)

        try:
            low_level_actions = func(*params) if params else func()
            actions = [self.call_low_level_action(act["function"], *act["params"]) for act in low_level_actions]
            self.action_names.append(function_name)
            self.entities_involved.append([param.name if param else "" for param in params])
            self.low_level_actions.extend(low_level_actions)
            return actions
        except Exception as e:
            print(f"⚠️ Error executing '{function_name}': {e}")
            return None, None
        
    def call_low_level_action(self, function_name: str, *params):
        """
        Calls a low level action method defined in the robot subclass.
        """
        function_name = "low_level_" + function_name
        # Check that the method actually exists
        if not hasattr(self, function_name):
            print(f"❌ Function '{function_name}' not found in class '{type(self).__name__}'.")
            return None

        func = getattr(self, function_name)

        try:
            low_level_action = func(*params) if params else func()
            return low_level_action
        except Exception as e:
            print(f"⚠️ Error executing '{function_name}': {e}")
            return None

    def sync_action(self, entity: Entity):
        """
        Synchronizes all the low level actions with the provided entity.
        This is useful when the entity's position has been updated and we want to ensure
        that all actions involving this entity use the latest position.
        """
        
        # Update parameters with the provided entity if applicable
        for i in range(len(self.low_level_actions)):
            low_level_action = self.low_level_actions[i]
            low_level_action_name = low_level_action["function"]
            low_level_action_params = low_level_action["params"]
            for j, param in enumerate(low_level_action_params):
                if isinstance(param, Entity) and entity is not None:
                    if param.name == entity.name:
                        low_level_action_params[j] = entity
            self.low_level_actions[i]["params"] = low_level_action_params
            self.robot_actions[i] = self.call_low_level_action(low_level_action_name, *low_level_action_params)
    
    def action_step_success(self):
        """
        Call this method after successfully completing the current action step.
        Returns True if the entire action sequence is finished.
        """
        self.current_action_step += 1
        if self.current_action_step >= self.action_step:
            self.reset_actions()
            return True
        return False

    def reset_actions(self):
        """
        Resets the action state.
        """
        self.all_actions_finished = True
        self.current_action_step = 0
        self.action_step = 0
        self.action_names = []
        self.robot_actions = []
        self.entities_involved = []
        self.low_level_actions = []

    def add_actions(self, function_name: str, *params):
        """
        Adds an action to the robot's action list.
        """
        actions = self.execute(function_name, *params)

        # if actions:
        #     self.low_level_actions.extend(actions)
        # else:
        #     print(f"⚠️ Warning: No low-level actions returned from '{function_name}' with params {params}.")

        if actions:
            if self.all_actions_finished:
                self.all_actions_finished = False
                self.current_action_step = 0
                self.action_step = len(actions)
            for action in actions:
                self.robot_actions.append(action)
        else:
            print(f"⚠️ Warning: No actions returned from '{function_name}' with params {params}.")

    def current_action(self):
        """
        Returns the current action to be executed.
        """
        return self.robot_actions[self.current_action_step]
    
    def current_low_level_action(self):
        """
        Returns the current low level action to be executed.
        """
        return self.low_level_actions[self.current_action_step]
    
    def get_readable_actions(self):
        result = ""
        # print(f'action names: {self.action_names}')
        # print(f'entities involved: {self.entities_involved}')
        for i in range(len(self.action_names)):
            result += f"Action {i + 1}: {self.action_names[i]} with params: [{','.join(self.entities_involved[i])}]\n"
        return result.strip()
    
    def get_readable_current_low_level_action(self):
        action = self.current_low_level_action()
        if action is None:
            return "No current low level action."
        params = [param.name if isinstance(param, Entity) else "" for param in action["params"]]
        return f"Current low level action: {action['function']} with entity: [{', '.join(params)}]"