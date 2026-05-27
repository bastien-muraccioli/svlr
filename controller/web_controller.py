import json

import requests

from controller.controller import RobotController


class WebRobotController(RobotController):
    """
    HTTP client that mirrors the ROS2 node interface.
    All node.* calls in the original ros_controller are replaced
    by HTTP requests to the robot web server.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._end_action = False

    # ------------------------------------------------------------------
    # Mirrored node interface
    # ------------------------------------------------------------------

    def end_action_received(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/end_action", timeout=1.0)
            r.raise_for_status()
            return r.json().get("end_action", False)
        except requests.RequestException:
            return False

    def reset_end_action(self) -> None:
        try:
            requests.post(f"{self.base_url}/reset_end_action", timeout=1.0)
        except requests.RequestException:
            pass

    def get_robot_pose(self, timeout_sec: float = 0.1):
        try:
            r = requests.get(
                f"{self.base_url}/robot_pose", timeout=max(timeout_sec, 0.2)
            )
            r.raise_for_status()
            data = r.json()
            # Return None if the server has no pose yet
            return data.get("pose")
        except requests.RequestException:
            return None

    def send_actions(self, action) -> None:
        if action is None:
            return
        try:
            # Handle list of actions - take the first one
            if isinstance(action, list):
                if len(action) == 0:
                    return
                payload = action[0]  # Extract the dict from the list
            elif isinstance(action, dict):
                payload = action
            elif isinstance(action, str):
                # Try to parse JSON string
                try:
                    payload = json.loads(action)
                    if isinstance(payload, list) and len(payload) > 0:
                        payload = payload[0]
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse action string as JSON: {action}")
                    return
            else:
                print(f"Warning: Unexpected action type: {type(action)}")
                return

            requests.post(f"{self.base_url}/send_action", json=payload, timeout=1.0)
        except requests.RequestException as e:
            print(f"Failed to send action: {e}")
