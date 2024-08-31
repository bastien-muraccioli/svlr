# SVLR: Scalable, Training-Free Visual Language Robotics: a modular multi-model framework for consumer-grade GPUs

[![arXiv](https://img.shields.io/badge/arXiv--df2a2a.svg?style=for-the-badge)](comming_soon)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**How to add a new robot and new actions**](#how-to-add-a-new-robot-and-new-actions) | [**How to add new AI models**](#how-to-add-new-ai-models) | [**Project Website**](https://scalable-visual-language-robotics.github.io/) | [**Citation**](#citation)


<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2024-09-01] Initial release

<hr style="border: 2px solid gray;"></hr>

## **Scalable Visual Language Robotics (SVLR) Framework**

A modular framework for controlling robots using visual and language inputs, based on multi-model approach.

Utilizes a Visual Language Model (VLM), zero-shot image segmentation, a Large Language Model (LLM), and a sentence similarity model to process images and instructions.

## Installation
```bash
# Create and activate conda environment
conda create -n svlr python=3.10 -y
conda activate svlr

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Clone and install the slvr repo
git clone https://github.com/bastien-muraccioli/svlr.git
cd svlr
pip install -r requirements.txt
```

## Getting Started

### Running SVLR in simulation
This mode was initially made for debug purposes, but can be used to test the framework without a robot.

At the end you will see the image with the detected objects and the predicted actions.
``` bash
# It will run the SVLR framework with test.png in the pictures folder
python main.py --show_image --simulation

# You can also specify a custom image, put your image in the pictures folder and run:
python main.py --show_image --simulation --simulation_image_file your_image.png
```

### Running SVLR with the UR10 robot

This mode will allow you to control the UR10 robot with the SVLR framework.

#### Requirements

- [ROS Noetic](http://wiki.ros.org/noetic/Installation) with the [UR10 controller](https://github.com/ThomasDuvinage/ur_robotiq_controller), custom controller can be used, as it can receive data from the SVLR framework.
- UR10 Robot + camera + gripper (Robotiq 2F-140) (However, SVLR is adaptable to any robot with any gripper and camera, but you will need to create a custom controller)
- Have a calibrated camera : If you have a USB camera you can use the ROS package [https://github.com/bastien-muraccioli/logicool) and follow the instructions to calibrate the camera. At the end, you will need to save the calibration matrix usb_cam.yaml into the svlr root folder and rename it to calibration.yaml.
- In slvr/actions/UR10_action.json, you need to specify: 
  - the init_pose in the end effector coordinates 
  - the eye_to_hand: dx and dy that are the offsets between the camera and the end effector.
  - the eye_to_hand: depth that is the distance between the camera and your setup during the init pose, if you are using a table, it's the distance between the camera and the table.
- In slvr/actions/UR10_pick_place.py, you need to specify the zmin where your robot can reach the objects on the table.

#### How to run
``` bash
# --show_image will display the image of the camera but the argument is optional
python main.py --show_image
```

## Arguments
Below is a list of arguments you can use when running `main.py` to control the robot:

- **Robot and Server Information:**
  - `--robot_name` (`str`, default: `"UR10"`): Specifies the name of the robot.
  - `--server` (`str`, default: `'127.0.0.1'`): Sets the robot server's IP address.
  - `--port` (`int`, default: `65500`): Defines the port number for the robot server.
  - `--buffer` (`int`, default: `1024`): Determines the buffer size for the server.

- **Camera Settings:**
  - `--camera_device` (`str`, default: `'/dev/video2'`): Specifies the camera device path.
  - `--camera_width` (`int`, default: `640`): Sets the width of the camera feed.
  - `--camera_height` (`int`, default: `480`): Sets the height of the camera feed.

- **Large Language Model (LLM) Configuration:**
  - `--llm_name` (`str`, default: `'microsoft/Phi-3-mini-4k-instruct'`): Name of the LLM to be used.
  - `--llm_provider` (`str`, default: `'HuggingFace'`): LLM provider (`HuggingFace` or `OpenAI`).
  - `--llm_temperature` (`float`, default: `0.1`): Sets the LLM temperature (value between 0.1 and 1.0).
  - `--llm_is_chat` (flag): Indicates if the LLM is a chat model.

- **Simulation Mode:**
  - `--simulation` (flag): Runs the robot in simulation mode.
  - `--simulation_image_file` (`str`, default: `'test.png'`): Specifies the image file to use in simulation mode.

- **Image Handling:**
  - `--show_image` (flag): Displays the captured image.
  - `--save_image` (flag): Saves the captured image.

## Repository Structure

High-level overview of repository/project file-tree:

+ `actions/` - JSON files describing the robots and their associated actions programs files.
+ `pictures/` - camera images, generated plot and segmentation predictions.
+ `similarity_model/` - all-MiniLM-L6-v2 files for sentence similarity.
+ `src/` - main source code for the SVLR framework.
+ `tools/` - tools for the SVLR framework.
+ `calibration.yaml` - camera calibration matrix.
+ `control_loop.py` - main control loop for the SVLR framework.
+ `llm_prompt.json` - JSON file with the LLM prompt sytem templates.
+ `main.py` - main file to run the SVLR framework.
+ `requirements.txt` - Python dependencies.
+ `LICENSE` - All code is made available under the MIT License. 
+ `README.md` - You are here!

---

## How to add a new robot and new actions
In this section, we will explain how to add a new robot and new actions to the SVLR framework.

### Add a new robot
1. Create a new JSON file in the `actions/` folder, named {robot_name}_action.json.
2. Add the robot description in the JSON file, following the existing format.
``` json
{
    "robot_name": "robot's name",
    "description": "robot's description",
    "init_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "eye_to_hand": {
        "dx": 0.0,
        "dy": 0.0,
        "depth": 0.0
    },
}
```
3. You can also add more components that can be used in the actions. For example, we had the open and close gripper values in the UR10_action.json file like this:
``` json
{
  "gripper": {
     "open": 30,
     "close": 220
   },
}
```
4. Your robotic controller will need to receive the actions generated by the SVLR framework. The SVLR framework sends a list of what your actions return. Whatever, we recommend you to return a list of dict with the robot control information. It the case of the UR10, we return the end effector position and the gripper value as the following:
``` python
[
  {
    "end_effector": [x, y, z, rx, ry, rz],
    "gripper": gripper_value
  },
]
```
These information are sent to the robot controller by socket, you will need to specify the address and the port of your controller with the --server and --port arguments when running the main.py file.

### Add new actions
To add a new action to the SVLR framework, you need to modify the robot_action.json file and create a new Python file in the `actions/` folder.

It's important to notice that the action described in the json file will be used by the LLM to fulfill the user's request, more the description is clear, more the LLM will be able to understand what your action is doing.

Also, in the current implementation of SVLR, the parameters can only be the objects detected in the image. The LLM will generate the action with the detected objects names in the image, but your action program will receive the position of these objects, in the end effector coordinates, to execute the action.
We recommend you to explore the UR10 files to understand this process.

1. In your robot_action.json file, add the new actions with the following format:
``` json
{
    "actions": 
  [
    {
    "name": "action_name",
    "program": "{robot_name}_{program_name}",
    "description": "action description",
    "parameters": [{
      "type": "type",
      "description": "[parameter description]",
      "required": true
      }]
    },
  ]
}
```
2. Create a new Python file in the `actions/` folder, named {robot_name}_{program_name}.py. This file will contain the program for the action. You only need to return a list with the content that required your controller (e.g. With the UR10, we need to return a list of dict with the positions of the end effector and gripper values). The list is needed to let the framework execute multiple actions in a row.

## How to add new AI models
By default, the SVLR framework uses the following models from HuggingFace:
- VLM: OpenGVLab/Mini-InternVL-Chat-2B-V1-5
- LLM: microsoft/Phi-3-mini-4k-instruct
- Sentence Similarity: all-MiniLM-L6-v2
- Zero-Shot Image Segmentation: CIDAS/clipseg-rd64-refined

### Add a new VLM
As the lightweights open-source VLM are recent, it can be a bit tricky to add a new one. However, as it concerns the SVLR framework, you will only need to update the src/vlm.py file to use the new model.

### Add a new LLM
To add a new LLM, you need to specify its system prompt in the llm_prompt.json file, otherwise it will use the default prompt, that is not recommended. 

Then you will need to specify its name and its provider (HuggingFace or OpenAI) with the --llm_name and --llm_provider arguments when running the main.py file. If you want to use a chat model, you will need to specify the --llm_is_chat argument.

As it concerns the OpenAI models, you can create a .env file in the root folder with the following content:
``` bash
OPENAI_API_KEY=your_openai_api_key
```

If you want to add other LLM providers such as Ollama, you will need to modify the src/llm.py file and install the necessary dependencies.

By default, the LLM from HuggingFace are run with a 4bit quantization, if you want to use the full precision, you will need to modify the src/llm.py file.

### Add a new Sentence Similarity model
In src/action.py, you will need to replace the model_path variable by the name of the new model.

### Add a new Zero-Shot Image Segmentation model
In src/perception.py, you will need to replace the seg_model_name variable by the name of the new model.


## Citation

**comming soon**

[//]: # (If you find our code useful in your work, please cite [our paper]&#40;comming_soon&#41;:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{svlr,)

[//]: # (    title={Scalable, Training-Free Visual Language Robotics: a modular multi-model framework for consumer-grade GPUs},)

[//]: # (    author={Marie Samson and Bastien Muraccioli and Fumio Kanehiro},)

[//]: # (    journal = {},)

[//]: # (    year={2024})

[//]: # (} )

[//]: # (```)