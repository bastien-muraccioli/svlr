# SVLR: Scalable Visual Language Robotics

**A modular multi-model framework for consumer-grade GPUs**

[![arXiv](https://img.shields.io/badge/arXiv--df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2502.01071)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Adding Robots & Actions**](#how-to-add-a-new-robot-and-new-actions) | [**Adding AI Models**](#how-to-add-new-ai-models) | [**Project Website**](https://scalable-visual-language-robotics.github.io/) | [**Citation**](#citation)

---

## Latest Updates
- **2024-09-01** - Initial release
- **2026-04-07** - Support for SO-ARM100

---

## Overview

SVLR is a scalable, training-free framework for controlling robots using visual and language inputs. It leverages a modular multi-model approach combining:

- **Visual Language Model (VLM)** - For scene understanding
- **Zero-shot Image Segmentation** - For object detection
- **Large Language Model (LLM)** - For instruction interpretation
- **Sentence Similarity Model** - For semantic matching

This architecture enables intuitive robot control through natural language commands and visual perception without requiring custom training.

---

## Installation

### Prerequisites
- Python 3.10 or 3.12.3
- [Ollama](https://ollama.com/) installed on your system
- CUDA-compatible GPU (recommended)

### Setup

1. **Install PyTorch**
   ```bash
   # Check https://pytorch.org/get-started/locally/ for platform-specific instructions
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Clone and install SVLR**
   ```bash
   git clone https://github.com/bastien-muraccioli/svlr.git
   cd svlr
   pip install -r requirements.txt
   ```

3. **Download the VLM model**
   ```bash
   ollama run llava-phi3
   ```

**Note:** SVLR is compatible with both `venv` and `conda` virtual environments.

---

## Getting Started

### Simulation Mode

Test the framework without a physical robot:

```bash
# Run with default test image
python main.py --simulation

# Run with custom image (place in pictures/ folder)
python main.py --simulation --simulation_image_file your_image.png
```

The output displays detected objects and predicted actions overlaid on the image.

---

### Running with UR10 Robot

Control a UR10 robot arm with SVLR.

#### Prerequisites

**Hardware:**
- UR10 robot arm
- USB camera
- Robotiq 2F-140 gripper (or compatible)

**Software:**
- [ROS Noetic](http://wiki.ros.org/noetic/Installation)
- [UR10 controller](https://github.com/ThomasDuvinage/ur_robotiq_controller) (or custom controller)

#### Setup

1. **Camera Calibration**
   - Install the ROS [logicool](https://github.com/bastien-muraccioli/logicool) package
   - Follow calibration instructions
   - Save calibration matrix as `usb_cam.yaml`
   - Copy to SVLR root and rename to `calibration.yaml`

2. **Configure Robot Parameters** (`svlr/actions/UR10_action.json`)
   - **`init_pose`**: Initial end-effector coordinates
   - **`eye_to_hand.dx/dy`**: Camera-to-end-effector offsets (meters)
   - **`eye_to_hand.depth`**: Camera-to-workspace distance at init pose

3. **Set Workspace Limits** (`svlr/actions/UR10_pick_place.py`)
   - Define **`zmin`**: Minimum reachable z-coordinate on table

#### Running

```bash
python main.py
```

**Note:** SVLR is adaptable to other robot arms, grippers, and cameras with custom controllers.

---

### Running with SO-ARM100 Robot

Control the [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) robot using a client-server architecture.

#### Prerequisites

**Hardware:**
- SO-ARM100 robot arm
- USB camera
- Raspberry Pi (optional, for remote control)

**Software:**
- [LeRobot](https://github.com/huggingface/lerobot) v0.5.1+
- Python 3.x

#### Setup

##### 1. Camera Calibration

1. Install the ROS [logicool](https://github.com/bastien-muraccioli/logicool) package
2. Follow calibration instructions
3. Save as `usb_cam.yaml` → copy to SVLR root → rename to `calibration.yaml`

##### 2. Configuration Files

**Action Configuration** (`svlr/actions/SO100_action.json`):
- **`init_pose`**: Initial end-effector coordinates
- **`eye_to_hand.dx/dy`**: Camera-to-end-effector offsets (meters)
- **`eye_to_hand.depth`**: Camera-to-workspace distance at init pose
- **`rot_mat`**: Rotation matrix (camera frame → robot frame)

**Pick-and-Place Configuration** (`svlr/actions/SO100_pick_place.py`):
- **`zmin`**: Minimum reachable z-coordinate on table

#### Running the System

##### Server Setup (Raspberry Pi or control machine)

1. **Download SO-ARM100 URDF files**
   ```bash
   git clone https://github.com/TheRobotStudio/SO-ARM100.git
   ```

2. **Transfer server script**
   ```bash
   scp lerobot_webserver.py pi@<raspberry-pi-ip>:~/
   ```

3. **Start server**
   ```bash
   python lerobot_webserver.py \
     --mode real \
     --urdf /path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
   ```

**Server Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--mode {mock,real}` | Backend mode | `mock` |
| `--host HOST` | Server bind address | `0.0.0.0` |
| `--port PORT` | Server port | `8000` |
| `--action-duration SECONDS` | [mock] Auto-completion delay | `2.0` |
| `--port-id SERIAL_PATH` | [real] Serial port path | - |
| `--urdf URDF_PATH` | [real] SO-101 URDF file path | - |
| `--robot-id ROBOT_ID` | [real] Robot identifier | `so100_follower` |
| `--lerp-speed M_PER_S` | [real] End-effector travel speed | `0.1` |

**Example with custom settings:**
```bash
python lerobot_webserver.py \
  --mode real \
  --host 0.0.0.0 \
  --port 8000 \
  --urdf ~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --lerp-speed 0.15
```

##### Client Setup (SVLR)

```bash
python main.py --robot_name SO100 --http_server <raspberry-pi-ip>
```

**Example:**
```bash
python main.py --robot_name SO100 --http_server 192.168.1.100
```

---

## Command-Line Arguments

Configure SVLR behavior using the following arguments:

### Robot & Server
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot_name` | `str` | `"UR10"` | Robot name |
| `--server` | `str` | `'127.0.0.1'` | Robot server IP address |
| `--port` | `int` | `65500` | Robot server port |
| `--buffer` | `int` | `1024` | Server buffer size |
| `--http_server` | `str` | - | HTTP server IP (for SO-ARM100) |

### Camera
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--camera_topic` | `str` | `""` | ROS camera image topic |
| `--camera_device` | `str` | `'/dev/video2'` | Camera device path |
| `--camera_width` | `int` | `640` | Camera feed width |
| `--camera_height` | `int` | `480` | Camera feed height |

### Large Language Model (LLM)
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--llm_name` | `str` | `'microsoft/Phi-3-mini-4k-instruct'` | LLM model name |
| `--llm_provider` | `str` | `'HuggingFace'` | Provider (`HuggingFace` or `OpenAI`) |
| `--llm_temperature` | `float` | `0.1` | Sampling temperature (0.1-1.0) |
| `--llm_is_chat` | flag | - | Use chat model variant |

### Simulation & Debug
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--simulation` | flag | - | Run in simulation mode |
| `--simulation_image_file` | `str` | `'test.png'` | Image file for simulation |
| `--show_image` | flag | - | Display captured image |
| `--save_image` | flag | - | Save captured image |

---

## Repository Structure

```
svlr/
├── actions/              # Robot definitions and action programs
├── pictures/             # Camera images and visualizations
├── similarity_model/     # all-MiniLM-L6-v2 sentence similarity
├── src/                  # Core SVLR framework code
├── tools/                # Utility scripts
├── calibration.yaml      # Camera calibration matrix
├── control_loop.py       # Main control loop
├── llm_prompt.json       # LLM system prompt templates
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md             # Documentation
```

---

## How to Add a New Robot and New Actions

### Adding a New Robot

1. **Create robot definition** - `actions/{robot_name}_action.json`

   ```json
   {
     "robot_name": "MyRobot",
     "description": "Description of the robot",
     "init_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     "eye_to_hand": {
       "dx": 0.0,
       "dy": 0.0,
       "depth": 0.0
     }
   }
   ```

2. **Add robot-specific components** (optional)

   Example - gripper configuration:
   ```json
   {
     "gripper": {
       "open": 30,
       "close": 220
     }
   }
   ```

3. **Implement robot controller**

   Your controller should receive actions from SVLR via socket. We recommend returning a list of dictionaries with control information:

   ```python
   [
     {
       "end_effector": [x, y, z, rx, ry, rz],
       "gripper": gripper_value
     }
   ]
   ```

   Configure the controller address using `--server` and `--port` arguments.

### Adding New Actions

Actions enable the LLM to fulfill user requests. Clear descriptions improve LLM understanding.

**Current limitation:** Action parameters must be objects detected in the image. The LLM generates actions using object names, but your program receives object positions in end-effector coordinates.

**Recommended:** Study the UR10 implementation files to understand this workflow.

#### Steps

1. **Define action** in `actions/{robot_name}_action.json`

   ```json
   {
     "actions": [
       {
         "name": "pick_and_place",
         "program": "{robot_name}_pick_place",
         "description": "Pick up an object and place it at a target location",
         "parameters": [
           {
             "type": "object",
             "description": "Object to pick up",
             "required": true
           },
           {
             "type": "object",
             "description": "Target placement location",
             "required": true
           }
         ]
       }
     ]
   }
   ```

2. **Implement action program** - `actions/{robot_name}_{program_name}.py`

   Return a list of control dictionaries compatible with your robot controller. The list structure allows executing multiple sequential actions.

---

## How to Add New AI Models

### Default Models

SVLR uses the following models from HuggingFace by default:

- **VLM:** `llava-phi3`
- **LLM:** `microsoft/Phi-3-mini-4k-instruct`
- **Sentence Similarity:** `all-MiniLM-L6-v2`
- **Zero-Shot Segmentation:** `CIDAS/clipseg-rd64-refined`

### Adding a New LLM

1. **Define system prompt** in `llm_prompt.json`
   
   ```json
   {
     "model_name": {
       "system": "Your custom system prompt here"
     }
   }
   ```

2. **Run with custom LLM**

   ```bash
   python main.py --llm_name "your-model-name" --llm_provider "HuggingFace"
   ```

   For chat models, add `--llm_is_chat` flag.

#### OpenAI Models

Create `.env` in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
```

Run with:
```bash
python main.py --llm_name "gpt-4" --llm_provider "OpenAI"
```

#### Other Providers (e.g., Ollama)

Modify `src/llm.py` and install necessary dependencies.

#### Quantization

By default, HuggingFace LLMs use 4-bit quantization. For full precision, modify `src/llm.py`.

### Adding a New Sentence Similarity Model

In `src/action.py`, update the `model_path` variable:

```python
model_path = "sentence-transformers/your-model-name"
```

### Adding a New Zero-Shot Segmentation Model

In `src/perception.py`, update the `seg_model_name` variable:

```python
seg_model_name = "your-segmentation-model"
```

---

## License

All code is made available under the [MIT License](LICENSE).

---

## Troubleshooting

### Common Issues

**Camera calibration errors:**
- Verify `calibration.yaml` is in the SVLR root directory
- Re-run calibration if matrix appears incorrect

**Connection issues (SO-ARM100):**
- Ensure client and server are on the same network
- Verify Raspberry Pi IP address is correct
- Check server is running with `--mode real`

**Robot unreachable positions:**
- Adjust `zmin` in `{robot_name}_pick_place.py`
- Verify `init_pose` is within robot workspace

**Model loading failures:**
- Ensure Ollama is running: `ollama serve`
- Verify model is downloaded: `ollama list`
- Check GPU memory availability

**LLM errors:**
- Verify model name matches exactly
- Check `.env` file for OpenAI API key
- Ensure system prompt exists in `llm_prompt.json`

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## Support

- **Issues:** [GitHub Issues](https://github.com/bastien-muraccioli/svlr/issues)
- **Website:** [scalable-visual-language-robotics.github.io](https://scalable-visual-language-robotics.github.io/)
- **Paper:** [arXiv:2502.01071](https://arxiv.org/abs/2502.01071)

## Citation
If you find our work useful, please consider citing us!

```bibtex
@misc{samson2025scalabletrainingfreevisuallanguage,
      title={Scalable, Training-Free Visual Language Robotics: A Modular Multi-Model Framework for Consumer-Grade GPUs},
      author={Marie Samson and Bastien Muraccioli and Fumio Kanehiro},
      year={2025},
      eprint={2502.01071},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2502.01071},
}
```
