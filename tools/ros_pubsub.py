import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from svlr_msgs.msg import Action, Actions  # Assuming these are custom messages
from geometry_msgs.msg import Pose
from rclpy.task import Future
from sensor_msgs.msg import Image

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import struct


class RosPubSub(Node):
    def __init__(
        self,
        pub_topic="",
        sub_topic="",
        camera_topic="/camera/camera/image_raw",
        camera_points_topic="/camera/camera/depth/color/points",
        robot_pose_topic="/current_pose",
    ):
        super().__init__("ros_pubsub_node")

        self.bridge = CvBridge()
        self.message_received = False
        self.camera_topic = camera_topic
        self.camera_points_topic = camera_points_topic
        self.robot_pose_topic = robot_pose_topic

        if pub_topic:
            self.publisher_ = self.create_publisher(Actions, pub_topic, 10)

        if sub_topic:
            self.subscription = self.create_subscription(
                Bool, sub_topic, self.callback, 10
            )

    def get_robot_pose(self, timeout_sec=5.0):
        """
        Blocks until a robot pose is received or timeout.
        """
        future = Future()

        def callback(msg):
            if not future.done():
                future.set_result(msg)

        sub = self.create_subscription(Pose, self.robot_pose_topic, callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        self.destroy_subscription(sub)

        if not future.done():
            self.get_logger().error(f"Timeout while waiting for pose on topic: {self.robot_pose_topic}")
            return None

        pose_msg = future.result()
        position = pose_msg.position
        orientation = pose_msg.orientation
        return [
            position.x,
            position.y,
            position.z,
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z,
        ]

    def callback(self, msg):
        # self.get_logger().info(f"Received on {self.subscription.topic_name}: {msg.data}")
        self.message_received = True

    def get_camera_image_ros(self, timeout_sec=5.0):
        """
        Blocks until a camera image is received or timeout.
        """
        future = Future()

        def callback(msg):
            if not future.done():
                future.set_result(msg)

        sub = self.create_subscription(Image, self.camera_topic, callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        self.destroy_subscription(sub)

        if not future.done():
            self.get_logger().error(f"Timeout while waiting for image on topic: {self.camera_topic}")
            return None

        image_msg = future.result()
        return self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

    def get_pointcloud_message(self, timeout_sec=5.0):
        """
        Waits for a PointCloud2 message from the RealSense camera.
        """
        future = Future()

        def callback(msg):
            if not future.done():
                future.set_result(msg)

        sub = self.create_subscription(PointCloud2, self.camera_points_topic, callback, 10)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        self.destroy_subscription(sub)

        if not future.done():
            self.get_logger().error(f"Timeout waiting for PointCloud2 on topic: {self.camera_points_topic}")
            return None

        return future.result()

    def get_3d_point_from_pixel(self, pointcloud_msg, u, v):
        """
        Given a pixel coordinate (u, v) in the organized point cloud, returns (x, y, z).
        """
        try:
            width = pointcloud_msg.width
            height = pointcloud_msg.height
            point_step = pointcloud_msg.point_step
            row_step = pointcloud_msg.row_step

            # Safety check
            if u < 0 or v < 0 or u >= width or v >= height:
                self.get_logger().warn("Pixel coordinate out of bounds.")
                return None

            # Compute index of the pixel in the data array
            index = v * row_step + u * point_step

            # Extract bytes for x, y, z (float32)
            data = pointcloud_msg.data[index:index + 12]
            x, y, z = struct.unpack('fff', data)
            return (x, y, z)

        except Exception as e:
            self.get_logger().error(f"Error extracting 3D point: {e}")
            return None

    def send_actions(self, action_list):
        action_msg_list = []

        for action_dict in action_list:
            action_msg = Action()
            # self.get_logger().info(f"target pos x,y,z: {action_dict['pos_end_effector'][:3]}")
            action_msg.pose.position.x = float(action_dict["pos_end_effector"][0])
            action_msg.pose.position.y = float(action_dict["pos_end_effector"][1])
            action_msg.pose.position.z = float(action_dict["pos_end_effector"][2])

            action_msg.pose.orientation.w = float(action_dict["pos_end_effector"][3])
            action_msg.pose.orientation.x = float(action_dict["pos_end_effector"][4])
            action_msg.pose.orientation.y = float(action_dict["pos_end_effector"][5])
            action_msg.pose.orientation.z = float(action_dict["pos_end_effector"][6])

            action_msg.gripper_cmd.data = int(action_dict["gripper"])
            action_msg_list.append(action_msg)

        actions_msg = Actions()
        actions_msg.actions = action_msg_list

        self.publisher_.publish(actions_msg)
        # self.get_logger().info("Published Actions")

    def end_action_received(self):
        return self.message_received
    
    def reset_end_action(self):
        self.message_received = False

def main(args=None):
    rclpy.init(args=args)

    pub_topic = "/actions_topic"
    sub_topic = "/end_of_actions"

    node = RosPubSub(pub_topic, sub_topic)

    robot_info = {
        "init_pose": {"pos_end_effector": [0.5, 0.3, 0.2, 1.0, 0.0, 0.0, 0.0]},
        "gripper": {"open": 80, "close": 0},
    }

    action_data = [
        {
            "pos_end_effector": robot_info["init_pose"]["pos_end_effector"],
            "gripper": robot_info["gripper"]["open"],
        },
        {
            "pos_end_effector": robot_info["init_pose"]["pos_end_effector"],
            "gripper": robot_info["gripper"]["close"],
        },
    ]

    rate = node.create_rate(0.1)  # 10s loop

    try:
        while rclpy.ok():
            node.get_logger().info("Sending actions...")
            node.send_actions(action_data)
            rclpy.spin_once(node, timeout_sec=1.0)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
