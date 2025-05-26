import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from svlr_msgs.msg import Action, Actions  # Assuming these are custom messages
from geometry_msgs.msg import Pose
from rclpy.task import Future
from sensor_msgs.msg import Image


class RosPubSub(Node):
    def __init__(self, pub_topic="", sub_topic=""):
        super().__init__("ros_pubsub_node")

        self.bridge = CvBridge()
        self.message_received = False

        if pub_topic:
            self.publisher_ = self.create_publisher(Actions, pub_topic, 10)

        if sub_topic:
            self.subscription = self.create_subscription(
                Bool, sub_topic, self.callback, 10
            )

    def callback(self, msg):
        self.get_logger().info(f"Received on {self.subscription.topic_name}: {msg.data}")
        self.message_received = True

    def get_camera_image_ros(self, topic="/camera/image_raw", timeout_sec=5.0):
        """
        Mimics rospy.wait_for_message in ROS 2 by creating a temporary subscriber
        and blocking until a message is received or timeout is hit.
        """
        future = Future()

        def callback(msg):
            if not future.done():
                future.set_result(msg)

        # Create a one-shot temporary subscription
        sub = self.create_subscription(Image, topic, callback, 10)

        # Spin until the message is received or timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        # Destroy the temporary subscription
        self.destroy_subscription(sub)

        if not future.done():
            self.get_logger().error(f"Timeout while waiting for image on topic: {topic}")
            return None

        image_msg = future.result()
        return self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

    def send_actions(self, action_list):
        action_msg_list = []

        for action_dict in action_list:
            action_msg = Action()
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
        self.get_logger().info("Published Actions")


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
