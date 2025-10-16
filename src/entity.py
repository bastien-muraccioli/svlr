from tools.robot_tool import RobotCamera

class Entity:
    def __init__(self, name: str, robot_camera: RobotCamera = None):
        self.name = name
        self.pixel_pos = None  # (x, y) in image coordinates
        self.camera_frame_pos = None  # (x, y, z) in camera coordinates
        self.robot_frame_pos = None  # (x, y, z) in robot coordinates
        self.found = False  # Whether the entity has been detected by the VLM
        self.tracked = False  # Whether the entity is currently being tracked
        self.need_to_be_tracked = False  # Whether the entity needs to be tracked, working only if the entity has been found at least once
        self.robot_camera = robot_camera
        self.bbox = None  # (x_min, y_min, x_max, y_max)
        self.tracker = None
        self.mask = None  # binary mask of the entity in the image


    def __str__(self):
        return f"Entity(name={self.name}, found={self.found})"
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def objects_str(self):
        return ", ".join(str(e) for e in self)

    def update_position(self, pixel_pos: tuple):
        self.pixel_pos = pixel_pos
        if pixel_pos is not None:
            self.found = True
            if self.robot_camera is not None:
                self.camera_frame_pos = self.robot_camera.pixel_to_camera_coordinates(
                    list(pixel_pos)
                )
                self.robot_frame_pos = self.robot_camera.camera_to_robot(
                    list(self.camera_frame_pos)
                )
            else:
                print(f"No camera initialized for entity {self.name}, only pixel position updated.")
        else:
            print(f"Pixel position for entity {self.name} is None, cannot update positions.")
            self.found = False
            self.reset_tracking()

    def reset_tracking(self):
        self.need_to_be_tracked = False
        self.tracked = False
        self.tracker = None
        self.mask = None
