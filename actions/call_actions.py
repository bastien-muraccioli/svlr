import importlib


def call_robot_class(robot_name: str):
    """
    Dynamically loads the robot action class (e.g., UR10Actions)
    and returns an instance of it.
    """
    module_name = f"actions.{robot_name}_actions"
    class_name = f"{robot_name}Actions"

    try:
        # Import the robot actions class dynamically
        module = importlib.import_module(module_name)
        RobotClass = getattr(module, class_name)
        return RobotClass()

    except ImportError:
        print(f"❌ Failed to import module '{module_name}'.")
    except AttributeError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error while loading '{class_name}': {e}")
