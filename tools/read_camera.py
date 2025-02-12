import cv2


def get_camera_image(device="/dev/video2", width=1920, height=1080):
    # Open the video device
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"Error: Could not open video device {device}")
        return None

    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Capture a single frame
    ret, frame = cap.read()

    # Release the video device
    cap.release()

    if not ret:
        print("Error: Could not read frame from video device")
        return None

    return frame


def get_camera_image_ros(topic="/camera/image_raw"):
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rospy.init_node("svlr_image_subsriber_node", anonymous=True)

    try:
        image_msg = rospy.wait_for_message(topic, Image)
    except Exception as e:
        rospy.logerr(f"Error getting the image on topic : {topic}, {e}")
        return None

    cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")

    cv2.imshow("Captured Image", cv_image)
    cv2.waitKey(0)

    return cv_image
