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
            return data if data.get("pose") is not None else None
        except requests.RequestException:
            return None

    def send_actions(self, action) -> None:
        if action is None:
            return
        try:
            payload = action if isinstance(action, dict) else {"action": str(action)}
            requests.post(
                f"{self.base_url}/send_action", json=payload, timeout=1.0
            )
        except requests.RequestException:
            pass