import cv2

def get_camera_image(device='/dev/video2', width=1920, height=1080):
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