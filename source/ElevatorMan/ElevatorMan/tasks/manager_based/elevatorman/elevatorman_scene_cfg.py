# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

import os

# Get the absolute path to the elevator asset
# This file is at: source/ElevatorMan/ElevatorMan/tasks/manager_based/elevatorman/elevatorman_scene_cfg.py
# Project root is 6 levels up: elevatorman -> manager_based -> tasks -> ElevatorMan -> ElevatorMan -> source -> root
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_CUR_DIR))))))
ELEVATOR_ASSET_PATH = os.path.join(_PROJECT_ROOT, "assets", "Collected_elevator_urdf", "elevator.usd")


##
# Scene definition
##
@configclass
class ElevatormanSceneCfg(InteractiveSceneCfg):
    """Configuration for the elevatorman scene with a robot.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Elevator
    elevator = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Elevator",
        spawn=UsdFileCfg(usd_path=ELEVATOR_ASSET_PATH),
    )

