import threading

from controller.controller import RobotController


class SimulationController(RobotController):
    """
    HTTP client that mirrors the ROS2 node interface.
    All node.* calls in the original ros_controller are replaced
    by HTTP requests to the robot web server.
    """

    def __init__(self):
        self._end_action = False

    # ------------------------------------------------------------------
    # Mirrored node interface
    # ------------------------------------------------------------------

    def end_action_received(self) -> bool:
        return self._end_action

    def reset_end_action(self) -> None:
        self._end_action = False

    def get_robot_pose(self, timeout_sec: float = 0.1):
        return [0, 0, 0, 1.0, 0.0, 0.0, 0.0]  # Dummy pose (x, y, z, qw, qx, qy, qz)

    def send_actions(self, _action) -> None:
        def set_done():
            self._end_action = True

        timer = threading.Timer(1.0, set_done)
        timer.start()
