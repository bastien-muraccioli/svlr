"""
Robot web server — FastAPI + uvicorn
-------------------------------------
Exposes the same HTTP API consumed by WebRobotNode / web_controller.py.

Two modes
---------
  mock   – simulated robot; no hardware required
  real   – drives a real SO-100 follower arm via lerobot

Usage
-----
  # Mock (default)
  python robot_server.py

  # Real robot
  python robot_server.py --mode real \
      --port-id /dev/serial/by-id/usb-... \
      --urdf    /path/to/so101_new_calib.urdf

  # Adjust auto-completion delay (mock only)
  python robot_server.py --action-duration 1.5

Endpoints
---------
  GET  /robot_pose          – current EE / joint pose
  GET  /end_action          – completion flag
  GET  /status              – full debug snapshot
  POST /send_action         – receive an action dict
  POST /reset_end_action    – clear the completion flag
  POST /segment_entity      – trigger perception search (mock: 50 % success)
  POST /stop                – graceful shutdown
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import threading
import time
from typing import Any
import scipy

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

def dicts_close(d1, d2, tol=1e-4):
    if d1.keys() != d2.keys():
        return False

    for k in d1:
        v1, v2 = d1[k], d2[k]

        if isinstance(v1, dict) and isinstance(v2, dict):
            if not dicts_close(v1, v2, tol):
                return False

        elif isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            if not math.isclose(v1, v2, abs_tol=tol):
                return False

        else:
            if v1 != v2:
                return False

    return True

# ===========================================================================
# Pydantic request / response models
# ===========================================================================

class ActionPayload(BaseModel):
    model_config = {"extra": "allow"}   # accept any action fields


class SegmentEntityPayload(BaseModel):
    name: str


class OkResponse(BaseModel):
    ok: bool = True


class EndActionResponse(BaseModel):
    end_action: bool


class PoseResponse(BaseModel):
    pose: list[float] | None


class StatusResponse(BaseModel):
    action_count: int
    current_action: dict[str, Any] | None
    end_action: bool
    pose: dict[str, float] | None
    mode: str


# ===========================================================================
# Abstract robot backend
# ===========================================================================

class RobotBackend:
    """Common interface shared by mock and real backends."""

    def get_pose(self) -> list[float]:
        raise NotImplementedError

    def receive_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError

    def is_end_action(self) -> bool:
        raise NotImplementedError

    def reset_end_action(self) -> None:
        raise NotImplementedError

    def segment_entity(self, name: str) -> bool:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


# ===========================================================================
# Mock backend
# ===========================================================================

class MockBackend(RobotBackend):
    """
    Simulated robot.
    - Pose drifts smoothly in a background thread.
    - Every action auto-completes after `action_duration` seconds.
    - Entity segmentation succeeds on every even-numbered action.
    """

    def __init__(self, action_duration: float = 2.0):
        self._lock = threading.Lock()
        self._pose: list[float] = [0.2270497637597, -0.017332672293606, 0.156185435044701, 0.05362928,  0.98052498, -0.00677439,  0.18880883]  # x, y, z, wq, wx, wy, wz
        self._end_action = False
        self._current_action: dict[str, Any] | None = None
        self._action_count = 0
        self.action_duration = action_duration
        self._timer: threading.Timer | None = None

    def get_pose(self) -> list[float]:
        with self._lock:
            return self._pose

    # -- actions --

    def receive_action(self, action: dict[str, Any]) -> None:
        with self._lock:
            self._current_action = action
            self._action_count += 1
            count = self._action_count
            timer_already_running = self._timer is not None

        print(f"[mock] Action #{count}: {action}")

        if timer_already_running:
            print("[mock] Previous action timer still running, skipping")
            return

        print(f"[mock] Starting completion timer ({self.action_duration}s)")
        t = threading.Timer(self.action_duration, self._complete_action, args=[count])
        t.daemon = True
        with self._lock:
            self._timer = t
        t.start()

    def _complete_action(self, count: int):
        with self._lock:
            self._end_action = True
            self._timer = None
        print(f"[mock] Action #{count} completed → end_action=True")

    # -- flags --

    def is_end_action(self) -> bool:
        with self._lock:
            return self._end_action

    def reset_end_action(self) -> None:
        with self._lock:
            self._end_action = False
        print("[mock] end_action reset")

    # -- perception --

    def segment_entity(self, name: str) -> bool:
        with self._lock:
            found = self._action_count % 2 == 0
        print(f"[mock] segment_entity '{name}' → {found}")
        return found

    # -- debug --

    def status_extra(self) -> dict[str, Any]:
        with self._lock:
            return {
                "action_count": self._action_count,
                "current_action": self._current_action,
                "end_action": self._end_action,
                "pose": dict(self._pose),
            }


# ===========================================================================
# Real lerobot backend
# ===========================================================================
class RealRobotBackend(RobotBackend):
    """
    Drives a real SO-100 follower arm via lerobot.

    Actions are expected to be EE dicts:
      {"ee.x": ..., "ee.y": ..., "ee.z": ...,
       "ee.wx": ..., "ee.wy": ..., "ee.wz": ..., "ee.gripper_pos": ...}

    The server runs a continuous 30 Hz control loop in a background thread.
    When a new action arrives, the loop generates a waypoint trajectory from
    the current position to the target and executes it frame by frame.
    end_action is set once the trajectory is complete and the arm is within
    `completion_threshold` metres of the target for `completion_hold_frames`
    consecutive frames.
    """

    FPS = 30
    LERP_SPEED_M_S = 0.1            # metres per second (5 cm/s default)
    COMPLETION_THRESHOLD_M = 0.020   # 20 mm
    COMPLETION_HOLD_FRAMES = 5       # ~0.17 s at 30 Hz

    FIXED_ORIENTATION = {
        "ee.wx": 0.150820363068056,
        "ee.wy": 2.75750747340798,
        "ee.wz": -0.019051469907274,
    }

    DEFAULT_INITIAL_EE = {
        "ee.x": 0.2270497637597,
        "ee.y": -0.017332672293606,
        "ee.z": 0.146185435044701,
        "ee.gripper_pos": 1.6178736517719567,
    }

    def __init__(
        self,
        port_id: str,
        urdf_path: str,
        robot_id: str = "so100_follower",
        lerp_speed_m_s: float | None = None,
        initial_ee: dict[str, float] | None = None,
        skip_init: bool = False,
    ):
        # Import lerobot lazily so mock mode works without it installed
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.processor import RobotProcessorPipeline
        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            robot_action_to_transition,
            transition_to_robot_action,
        )
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            EEBoundsAndSafety,
            ForwardKinematicsJointsToEE,
            InverseKinematicsEEToJoints,
        )
        from lerobot.utils.robot_utils import precise_sleep

        self._precise_sleep = precise_sleep

        config = SO100FollowerConfig(
            port=port_id, id=robot_id, use_degrees=True
        )
        self._follower = SO100Follower(config)
        motor_names = list(self._follower.bus.motors.keys())

        kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=motor_names,
        )
        self.urdf_path = urdf_path

        self._fk = RobotProcessorPipeline(
            steps=[ForwardKinematicsJointsToEE(kinematics=kin, motor_names=motor_names)],
            to_transition=robot_action_to_transition,
            to_output=transition_to_robot_action,
        )

        self._ik = RobotProcessorPipeline(
            steps=[
                EEBoundsAndSafety(
                    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                    max_ee_step_m=0.10,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=kin,
                    motor_names=motor_names,
                    initial_guess_current_joints=False,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

        self._follower.connect()

        self.lerp_speed_m_s = lerp_speed_m_s if lerp_speed_m_s is not None else self.LERP_SPEED_M_S

        self._lock = threading.Lock()
        self._end_action = False
        self._current_action: dict[str, Any] | None = None
        self._action_count = 0
        self._hold_frames = 0

        # Trajectory state — protected by _lock
        self._trajectory: list[dict[str, float]] = []
        self._trajectory_index: int = 0
        self._target_ee: dict[str, float] | None = None

        # Initialize measured EE from FK
        obs = self._follower.get_observation()
        ee = self._fk(obs)
        self._current_ee: dict[str, float] = {
            "ee.x":          float(ee["ee.x"]),
            "ee.y":          float(ee["ee.y"]),
            "ee.z":          float(ee["ee.z"]),
            "ee.gripper_pos": float(ee.get("ee.gripper_pos", 1.617)),
        }

        # Move to initial position if requested
        if not skip_init:
            target_ee = initial_ee if initial_ee is not None else self.DEFAULT_INITIAL_EE
            self._move_to_initial_position(target_ee)

        # Control loop
        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True
        )
        self._control_thread.start()
        
        print(
            f"[real] Connected to robot on {port_id}  "
            f"(lerp speed={self.lerp_speed_m_s} m/s)"
        )

    # -- initialization helper --

    def _move_to_initial_position(self, target_ee: dict[str, float]) -> None:
        """Move robot smoothly to initial position before starting control loop."""
        print("[INIT] Moving to initial position...")
        
        # Get current position
        current_pos = (
            self._current_ee["ee.x"],
            self._current_ee["ee.y"],
            self._current_ee["ee.z"]
        )
        
        # Target position
        target_pos = (
            target_ee["ee.x"],
            target_ee["ee.y"],
            target_ee["ee.z"]
        )
        
        # Calculate distance and frames
        dist = math.sqrt(
            (target_pos[0] - current_pos[0]) ** 2 +
            (target_pos[1] - current_pos[1]) ** 2 +
            (target_pos[2] - current_pos[2]) ** 2
        )
        duration_s = dist / self.lerp_speed_m_s
        frames = max(1, round(duration_s * self.FPS))
        
        print(f"[INIT] Distance: {dist*100:.1f} cm, Duration: {duration_s:.2f}s, Frames: {frames}")
        
        # Generate trajectory
        waypoints = [current_pos, target_pos]
        trajectory = self._lerp_waypoints(waypoints, frames)
        
        # Execute trajectory
        obs = self._follower.get_observation()
        target_gripper = target_ee.get("ee.gripper_pos", self._current_ee["ee.gripper_pos"])
        
        for i, pos in enumerate(trajectory):
            t0 = time.perf_counter()
            
            # Build EE action
            ee_action = {
                "ee.x": pos[0],
                "ee.y": pos[1],
                "ee.z": pos[2],
                "ee.gripper_pos": target_gripper,
                **self.FIXED_ORIENTATION
            }
            
            # Convert to joint space and send
            joints = self._ik((ee_action, obs))
            print("[INIT] Sending action", joints)
            self._follower.send_action(joints)
            
            # Update observation for next iteration
            obs = self._follower.get_observation()
            
            # Sleep to maintain FPS
            elapsed = time.perf_counter() - t0
            self._precise_sleep(max(1.0 / self.FPS - elapsed, 0.0))
            
            # Progress indicator every 30 frames
            if (i + 1) % 30 == 0 or i == len(trajectory) - 1:
                progress = (i + 1) / len(trajectory) * 100
                print(f"[INIT] Progress: {progress:.1f}%")
        
        # Update current EE to target
        self._current_ee = {
            "ee.x": target_ee["ee.x"],
            "ee.y": target_ee["ee.y"],
            "ee.z": target_ee["ee.z"],
            "ee.gripper_pos": target_gripper,
        }
        
        print("[INIT] Initial position reached", self._current_ee)

    @staticmethod
    def _lerp_waypoints(
        waypoints: list[tuple[float, float, float]],
        total_frames: int
    ) -> list[tuple[float, float, float]]:
        """
        Generate a smooth trajectory through waypoints.
        
        Args:
            waypoints: List of (x, y, z) positions
            total_frames: Total number of frames for the entire trajectory
            
        Returns:
            List of interpolated (x, y, z) positions
        """
        if len(waypoints) < 2:
            return waypoints
        
        trajectory = []
        num_segments = len(waypoints) - 1
        frames_per_segment = total_frames // num_segments
        
        for i in range(num_segments):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Use remaining frames for last segment
            if i == num_segments - 1:
                frames = total_frames - len(trajectory)
            else:
                frames = frames_per_segment
            
            for frame in range(frames):
                t = frame / max(frames, 1)
                pos = (
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                    start[2] + t * (end[2] - start[2]),
                )
                trajectory.append(pos)
        
        # Ensure we end at the final waypoint
        if trajectory[-1] != waypoints[-1]:
            trajectory.append(waypoints[-1])
        
        return trajectory

    # -- control loop --

    def _control_loop(self):
        obs = None  # Will be fetched on first iteration
        obs = self._follower.get_observation()
        
        # Import lerobot lazily so mock mode works without it installed
        from lerobot.model.kinematics import RobotKinematics
        from lerobot.processor import RobotProcessorPipeline
        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            robot_action_to_transition,
            transition_to_robot_action,
        )
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            EEBoundsAndSafety,
            ForwardKinematicsJointsToEE,
            InverseKinematicsEEToJoints,
        )
        motor_names = list(self._follower.bus.motors.keys())

        kin = RobotKinematics(
            urdf_path=self.urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=motor_names,
        )

        local_ik = RobotProcessorPipeline(
            steps=[
                EEBoundsAndSafety(
                    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                    max_ee_step_m=0.10,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=kin,
                    motor_names=motor_names,
                    initial_guess_current_joints=False,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        
        while self._running:
            t0 = time.perf_counter()

            # Get fresh observation at the start of each iteration
            obs = self._follower.get_observation()

            # Update measured EE via FK
            ee_meas = self._fk(obs)
            with self._lock:
                # Check if we have a trajectory to execute
                if self._trajectory_index < len(self._trajectory):
                    # Get current setpoint from trajectory
                    setpoint = self._trajectory[self._trajectory_index]
                    self._trajectory_index += 1
                    
                    target_for_hold = self._target_ee
                elif self._target_ee is not None:
                    # Trajectory finished — keep sending the final target
                    setpoint = dict(self._target_ee)
                    target_for_hold = setpoint
                else:
                    setpoint = None
                    target_for_hold = None

            if setpoint is not None:
                ee_action = {**setpoint, **self.FIXED_ORIENTATION}
                
                # Use the fresh observation from this iteration
                joints = local_ik((ee_action, obs))
                
                self._follower.send_action(joints)
                
                self._current_ee: dict[str, float] = {
                    "ee.x":          float(ee_meas["ee.x"]),
                    "ee.y":          float(ee_meas["ee.y"]),
                    "ee.z":          float(ee_meas["ee.z"]),
                    "ee.gripper_pos": float(ee_meas.get("ee.gripper_pos", 1.617)),
                }

                # Completion check against the final target
                if target_for_hold is not None:
                    dx = target_for_hold["ee.x"] - float(ee_meas["ee.x"])
                    dy = target_for_hold["ee.y"] - float(ee_meas["ee.y"])
                    dz = target_for_hold["ee.z"] - float(ee_meas["ee.z"])
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    print(f"[CONTROL] Distance to target: {dist*1000:.1f} mm")

                    with self._lock:
                        trajectory_done = self._trajectory_index >= len(self._trajectory)
                        
                        if trajectory_done:
                            self._hold_frames += 1
                            if (
                                self._hold_frames >= self.COMPLETION_HOLD_FRAMES
                                and not self._end_action
                            ):
                                self._end_action = True
                                self._hold_frames = 0
                                self._target_ee = None
                                print(
                                    f"[real] Action #{self._action_count} complete "
                                    f"(dist={dist*1000:.1f} mm) → end_action=True"
                                )
                        else:
                            self._hold_frames = 0

            elapsed = time.perf_counter() - t0
            self._precise_sleep(max(1.0 / self.FPS - elapsed, 0.0))

    # -- interface --

    def get_pose(self) -> list[float]:
        with self._lock:
            quat = scipy.spatial.transform.Rotation.from_rotvec([self.FIXED_ORIENTATION["ee.wx"], self.FIXED_ORIENTATION["ee.wy"], self.FIXED_ORIENTATION["ee.wz"]]).as_quat()
            return [
                float(self._current_ee["ee.x"]),
                float(self._current_ee["ee.y"]),
                float(self._current_ee["ee.z"]),
                quat[0],
                quat[1],
                quat[2],
                quat[3]
            ]

    def receive_action(self, action: dict[str, Any]) -> None:
        with self._lock:
            action = {
                "ee.x": action["pos_end_effector"][0],
                "ee.y": action["pos_end_effector"][1],
                "ee.z": action["pos_end_effector"][2],
                "ee.gripper_pos": action["gripper"],
            }
            if self._current_action is None or not dicts_close(action, self._current_action):
                print("NEW ACTION RECEIVED:", action)
                self._current_action = action
                self._action_count += 1
                self._end_action = False
                self._hold_frames = 0

                if "ee.x" in action:
                    # Current position
                    start_pos = (
                        self._current_ee["ee.x"],
                        self._current_ee["ee.y"],
                        self._current_ee["ee.z"]
                    )
                    
                    # Target position
                    end_pos = (
                        float(action["ee.x"]),
                        float(action["ee.y"]),
                        float(action["ee.z"])
                    )
                    
                    target_gripper = float(
                        action.get("ee.gripper_pos", self._current_ee.get("ee.gripper_pos", 1.617))
                    )

                    # Calculate distance and duration
                    dist = math.sqrt(
                        (end_pos[0] - start_pos[0]) ** 2 +
                        (end_pos[1] - start_pos[1]) ** 2 +
                        (end_pos[2] - start_pos[2]) ** 2
                    )
                    duration_s = dist / self.lerp_speed_m_s
                    min_frames = 10  # Ensure at least ~0.33s for smooth motion
                    frames = max(min_frames, round(duration_s * self.FPS))

                    # Generate trajectory through waypoints
                    waypoints = [start_pos, end_pos]
                    position_trajectory = self._lerp_waypoints(waypoints, frames)
                    
                    # Convert to full EE dicts
                    self._trajectory = [
                        {
                            "ee.x": pos[0],
                            "ee.y": pos[1],
                            "ee.z": pos[2],
                            "ee.gripper_pos": target_gripper,
                        }
                        for pos in position_trajectory
                    ]
                    
                    self._trajectory_index = 0
                    self._target_ee = {
                        "ee.x": end_pos[0],
                        "ee.y": end_pos[1],
                        "ee.z": end_pos[2],
                        "ee.gripper_pos": target_gripper,
                    }

                    print(
                        f"[real] Action #{self._action_count}: trajectory "
                        f"({start_pos[0]:+.4f}, {start_pos[1]:+.4f}, {start_pos[2]:+.4f}) → "
                        f"({end_pos[0]:+.4f}, {end_pos[1]:+.4f}, {end_pos[2]:+.4f})  "
                        f"dist={dist*100:.1f} cm  "
                        f"duration={duration_s:.2f}s  frames={frames}"
                    )

    def is_end_action(self) -> bool:
        with self._lock:
            return self._end_action
            # return False

    def reset_end_action(self) -> None:
        with self._lock:
            self._end_action = False
        print("[real] end_action reset")

    def segment_entity(self, name: str) -> bool:
        # Placeholder — wire up your perception pipeline here
        print(f"[real] segment_entity '{name}' (not implemented, returning False)")
        return False

    def shutdown(self) -> None:
        self._running = False
        time.sleep(0.2)
        self._follower.disconnect()
        print("[real] Robot disconnected")

    def status_extra(self) -> dict[str, Any]:
        with self._lock:
            return {
                "action_count": self._action_count,
                "current_action": self._current_action,
                "end_action": self._end_action,
                "pose": dict(self._current_ee),
                "trajectory_progress": f"{self._trajectory_index}/{len(self._trajectory)}"
            }

# ===========================================================================
# FastAPI application factory
# ===========================================================================

def create_app(backend: RobotBackend, mode: str) -> FastAPI:

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        backend.shutdown()

    app = FastAPI(
        title="Robot web server",
        description="HTTP bridge between the web controller and a robot backend.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # GET /robot_pose
    # ------------------------------------------------------------------ #
    @app.get("/robot_pose", response_model=PoseResponse)
    async def robot_pose():
        pose = await asyncio.to_thread(backend.get_pose)
        return {"pose": pose}

    # ------------------------------------------------------------------ #
    # GET /end_action
    # ------------------------------------------------------------------ #
    @app.get("/end_action", response_model=EndActionResponse)
    async def end_action():
        flag = await asyncio.to_thread(backend.is_end_action)
        return {"end_action": flag}

    # ------------------------------------------------------------------ #
    # GET /status
    # ------------------------------------------------------------------ #
    @app.get("/status", response_model=StatusResponse)
    async def status():
        extra = await asyncio.to_thread(backend.status_extra)
        return {**extra, "mode": mode}

    # ------------------------------------------------------------------ #
    # POST /send_action
    # ------------------------------------------------------------------ #
    @app.post("/send_action", response_model=OkResponse)
    async def send_action(payload: ActionPayload):
        await asyncio.to_thread(backend.receive_action, payload.model_dump())
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # POST /reset_end_action
    # ------------------------------------------------------------------ #
    @app.post("/reset_end_action", response_model=OkResponse)
    async def reset_end_action():
        await asyncio.to_thread(backend.reset_end_action)
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # POST /segment_entity
    # ------------------------------------------------------------------ #
    @app.post("/segment_entity")
    async def segment_entity(payload: SegmentEntityPayload):
        found = await asyncio.to_thread(backend.segment_entity, payload.name)
        return {"found": found, "name": payload.name}

    # ------------------------------------------------------------------ #
    # POST /stop
    # ------------------------------------------------------------------ #
    @app.post("/stop", response_model=OkResponse)
    async def stop():
        async def _shutdown():
            await asyncio.sleep(0.1)
            raise SystemExit(0)
        asyncio.create_task(_shutdown())
        return {"ok": True, "message": "shutting down"}

    return app


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Robot web server")
    p.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Backend mode (default: mock)",
    )
    p.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    p.add_argument(
        "--port", type=int, default=65500, help="Bind port (default: 65500)"
    )
    # Mock options
    p.add_argument(
        "--action-duration",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="[mock] seconds before action auto-completes (default: 2.0)",
    )
    # Real robot options
    p.add_argument(
        "--port-id",
        default="",
        metavar="SERIAL_PATH",
        help="[real] serial port, e.g. /dev/serial/by-id/usb-...",
    )
    p.add_argument(
        "--urdf",
        default="",
        metavar="URDF_PATH",
        help="[real] path to the SO-101 URDF file",
    )
    p.add_argument(
        "--robot-id",
        default="so100_follower",
        help="[real] robot ID string (default: so100_follower)",
    )
    p.add_argument(
        "--lerp-speed",
        type=float,
        default=None,
        metavar="M_PER_S",
        help=(
            "[real] EE travel speed for lerp in m/s — frame count is derived "
            f"from distance / speed (default: {RealRobotBackend.LERP_SPEED_M_S} m/s)"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "real":
        if not args.port_id:
            raise SystemExit("--port-id is required in real mode")
        if not args.urdf:
            raise SystemExit("--urdf is required in real mode")
        print(f"[server] Starting in REAL mode (port={args.port_id})")
        backend: RobotBackend = RealRobotBackend(
            port_id=args.port_id,
            urdf_path=args.urdf,
            robot_id=args.robot_id,
            lerp_speed_m_s=args.lerp_speed,
        )
    else:
        print(
            f"[server] Starting in MOCK mode "
            f"(action_duration={args.action_duration}s)"
        )
        backend = MockBackend(action_duration=args.action_duration)

    app = create_app(backend, mode=args.mode)

    print(
        f"[server] Listening on http://{args.host}:{args.port}\n"
        "  GET  /robot_pose\n"
        "  GET  /end_action\n"
        "  GET  /status\n"
        "  POST /send_action\n"
        "  POST /reset_end_action\n"
        "  POST /segment_entity\n"
        "  POST /stop\n"
       f"  Docs → http://localhost:{args.port}/docs\n"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()