#!/usr/bin/env python

import argparse
import logging
import time
from pathlib import Path
import json
import re
import os
import yaml
import cv2

import numpy as np
import tensorflow_datasets as tfds

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene

CONFIG_PATH = "/home/joonseo/rh20t_api/configs/configs.json"
robot_configs = load_conf(CONFIG_PATH)
TASK_DESCRIPTION_FILE = "/home/joonseo/rh20t_api/task_description.json"
with open(TASK_DESCRIPTION_FILE, "r", encoding="utf-8") as f:
    task_dict = json.load(f)

SETTING_FILE = "/home/joonseo/rh20t_api/configs/default.yaml"
with open(SETTING_FILE, 'r') as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

SERIALS = [
    '036422060215',
    '037522062165',
    '104122061850',
    '104122063678',
    '104422070044',
    '104422071090',
    '105422061350',
    'f0461559',
]
main_camera_serial_num = SERIALS[settings['chosen_cam_idx']]
features = {
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": [
                "gripper_pos_x",
                "gripper_pos_y",
                "gripper_pos_z",
                "gripper_quat_x",
                "gripper_quat_y",
                "gripper_quat_z",
                "gripper_quat_w",
            ]
        },
        "fps": 100.0,
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "axes": [
                "joint_1_angle",
                "joint_2_angle",
                "joint_3_angle",
                "joint_4_angle",
                "joint_5_angle",
                "joint_6_angle",
                "joint_7_angle",
                "gripper_width",
                "gripper_force_x",
                "gripper_force_y",
                "gripper_force_z",
                "gripper_torque_x",
                "gripper_torque_y",
                "gripper_torque_z",
            ]
        },
        "fps": 10.0,
    },
    "observation.cam_036422060215": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_037522062165": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_104122061850": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_104122063678": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_104422070044": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_104422071090": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_105422061350": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.cam_f0461559": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 360,
            "video.width": 640,
            "video.codec": "H.264",
            "video.pix_fmt": "gbrp",
            "video.is_depth_map": False,
            "video.fps": 10,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None, "fps": 10.0},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": 10.0},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": 10.0},
    "index": {"dtype": "int64", "shape": (1,), "names": None, "fps": 10.0},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": 10.0},
}

def generate_lerobot_frames(episode_path: str):
    # Because of RH20TScene implementation, we use both high freq and low freq scenes.
    high_freq_scene = RH20TScene(episode_path, robot_configs)
    low_freq_scene = RH20TScene(episode_path, robot_configs)

    # Task info
    filename = os.path.basename(episode_path)
    task_name = re.search(r"(task_\d{4})", filename).group(1)
    task_index = int(task_name.split("_")[1]) - 1
    task = task_dict[task_name]["task_description_english"]

    # Make Frames for each time stamp
    timestamps = low_freq_scene.low_freq_timestamps[main_camera_serial_num]
    for _t in timestamps:
        frame = {
            "timestamp": _t,
            "task": task,
            "task_index": task_index,
        }

        # Action
        tcp_aligned = high_freq_scene.get_tcp_aligned(_t)
        frame["action"] = tcp_aligned
        
        # Observation.cam_{serial_num}
        image_path_pairs = low_freq_scene.get_image_path_pairs(_t, image_types=["color"]) #Dict{Serial num: image path}
        for serial_num in image_path_pairs:
            color_video_file = os.path.join(episode_path, f"cam_{serial_num}", "color.mp4")
            cap = cv2.VideoCapture(color_video_file)
            ret, img = cap.read()
            if not ret:
                continue
            frame[f"observation.cam_{serial_num}"] = img

        # Observation.state
        joint_angles = low_freq_scene.get_joint_angles_aligned(_t)
        gripper_width = low_freq_scene.get_gripper_command(_t)
        force_torque = high_freq_scene.get_ft_aligned(_t)
        frame["observation.state"] = np.concatenate([joint_angles, [gripper_width], force_torque])
    
        # Cast fp64 to fp32
        for key in frame:
            if isinstance(frame[key], np.ndarray) and frame[key].dtype == np.float64:
                frame[key] = frame[key].astype(np.float32)

        yield frame


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--repo-id",
    #     type=str,
    #     help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    # )
    # args = parser.parse_args()
    # repo_id = args.repo_id
    repo_id = "joonseo-jang/rh20t_to_lerobot"

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="Franka",
        fps=10.0,
        features=features,
    )

    for frame in generate_lerobot_frames("/home/joonseo/rh20t_api/RH20T_cfg5/task_0001_user_0007_scene_0001_cfg_0005"):
        if frame is None:
            continue
        lerobot_dataset.add_frame(frame)
    
    lerobot_dataset.save_episode()
    lerobot_dataset.finalize()
    lerobot_dataset.push_to_hub(
        # Add openx tag, since it belongs to the openx collection of datasets
        tags=["openx"],
        private=False,
    )
    # generate_lerobot_frames("/home/joonseo/rh20t_api/RH20T_cfg5/task_0001_user_0007_scene_0001_cfg_0005")
