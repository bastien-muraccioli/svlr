from tools.read_json import read_robot_json

import importlib

def call_robot_function(robot_name: str, function_name: str, *params):
    program = ""

    robot = read_robot_json(robot_name=robot_name)
    
    for action in robot["actions"]:
        if action["name"] == function_name:
            program = action["program"]
            break
    
    if not program:
        print(f"{robot_name}'s function: {function_name} wasn't found.")
        return

    module_name = f"actions.{program}"

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        if params != (None,):
            return func(*params)
        else:
            return func()
    except ImportError:
        print(f"Failed to import module '{module_name}'")
    except AttributeError:
        print(f"Function '{function_name}' not found in module '{module_name}'")
    except TypeError as e:
        print(f"Error calling function '{function_name}': {e}")