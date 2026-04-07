import time
import requests


class WebRobotNode:
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


# ---------------------------------------------------------------------------
# Controller class (replaces the ROS-based one)
# ---------------------------------------------------------------------------

class WebController:
    """
    Replacement for the ROS2 controller loop.
    Uses HTTP polling instead of rclpy.spin_once / subscriber callbacks.
    """

    def __init__(self, controller, base_url: str = "http://localhost:8000"):
        self.controller = controller
        self.node = WebRobotNode(base_url)

        self.time_step: float = 0.05          # seconds between loop ticks
        self.max_try_to_find_entity: int = 3

        # Shared state (same names as the original)
        self.user_prompt: str = ""
        self.language_pipeline_has_run: bool = False
        self.perception_pipeline_has_run: bool = False
        self.end_action_received: bool = False
        self.robot_is_idle: bool = True
        self.robot_searching_for_entity: bool = False
        self.entity_to_find = None
        self.counter_try_to_find_entity: int = 0
        self.objects_found: list = []
        self.final_action = None
        self.action_size: int = 0
        self.action_counter: int = 0
        self.action_progress: str = ""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def web_controller(self):
        try:
            while True:
                time.sleep(self.time_step)

                if self.user_prompt == "stop":
                    return

                if not self.language_pipeline_has_run:
                    continue

                # --- Synchronise: end-of-action + robot idle ---
                if self.end_action_received and self.robot_is_idle:
                    print("Web controller ready to synchronise and send actions")
                    self.robot_is_idle = False
                    self.action_size, self.action_counter = (
                        self.controller.action_counter()
                    )
                    print(
                        f"Action size: {self.action_size}, "
                        f"Action counter: {self.action_counter + 1}"
                    )
                    self.node.reset_end_action()
                    self.end_action_received = False

                # --- Poll for end-of-action from the robot ---
                if (
                    not self.robot_searching_for_entity
                    and not self.end_action_received
                ):
                    if self.node.end_action_received():
                        print("End of action received from web server")
                        self.end_action_received = True
                        self.node.reset_end_action()
                        self.action_size, self.action_counter = (
                            self.controller.action_counter()
                        )
                        print(
                            f"Action size: {self.action_size}, "
                            f"Action counter: {self.action_counter + 1}"
                        )
                        all_steps_successful = self.controller.action_step_success()
                        print("Ready for next action")
                        if all_steps_successful:
                            print("All actions are done")
                            self.controller.perception.reset_all_session()
                            self.language_pipeline_has_run = False
                            self.robot_is_idle = True
                            self.end_action_received = True
                            self.perception_pipeline_has_run = False
                            continue

                # --- Update robot pose ---
                if not self.robot_is_idle:
                    robot_pose = self.node.get_robot_pose(timeout_sec=0.1)
                    if robot_pose:
                        self.controller.set_robot_pose(robot_pose)

                # --- Action tracking / entity search ---
                if not self.robot_searching_for_entity:
                    action_is_tracked, self.objects_found = (
                        self.controller.action_tracking(self.objects_found)
                    )
                    for entity in self.objects_found:
                        if entity.need_to_be_tracked and not entity.tracked:
                            print(
                                f"Entity {entity.name} needs to be tracked "
                                "but is not, trying to find it again..."
                            )
                            self.entity_to_find = entity
                            self.counter_try_to_find_entity = 0
                            self.robot_searching_for_entity = True
                            self.node.reset_end_action()
                            self.end_action_received = True
                            self.controller.action.robot_action_class.current_action_step -= 1
                            break
                    if not self.robot_searching_for_entity:
                        self.final_action = self.controller.get_current_action()
                        self.end_action_received = False
                else:
                    self.node.reset_end_action()
                    self.end_action_received = True
                    if self.counter_try_to_find_entity < self.max_try_to_find_entity:
                        print(
                            f"Trying to find entity {self.entity_to_find.name} "
                            f"({self.counter_try_to_find_entity + 1}/"
                            f"{self.max_try_to_find_entity})..."
                        )
                        entity_found = self.controller.perception.segment_one_entity(
                            self.entity_to_find.name
                        )
                        if entity_found:
                            self.objects_found = (
                                self.controller.perception.environment_description_list
                            )
                            print(f"Entity {self.entity_to_find.name} found again.")
                            self.robot_searching_for_entity = False
                            self.entity_to_find = None
                            self.counter_try_to_find_entity = 0
                        else:
                            self.counter_try_to_find_entity += 1

                # --- Send current action ---
                self.action_size, self.action_counter = (
                    self.controller.action_counter()
                )
                self.action_progress = (
                    f"Sending action {self.action_counter + 1}/{self.action_size}:\n"
                    f" {self.controller.get_readable_current_low_level_action()}"
                )
                self.node.send_actions(self.final_action)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Shutting down.")
        finally:
            print("Web controller shut down.")