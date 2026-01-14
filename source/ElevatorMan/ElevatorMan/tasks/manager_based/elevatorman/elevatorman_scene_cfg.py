# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from cfg.elevator import ELEVATOR_CFG

# Get door USD asset paths
# Use absolute paths calculated from config location (like cfg/elevator.py)
# File is at: source/ElevatorMan/ElevatorMan/tasks/manager_based/elevatorman/elevatorman_scene_cfg.py
# Walk up directory tree to find project root (where assets/ directory exists)
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = _CUR_DIR
# Walk up to find project root (max 10 levels to avoid infinite loop)
for _ in range(10):
    assets_dir = os.path.join(_PROJECT_ROOT, "assets")
    if os.path.exists(assets_dir):
        break
    parent = os.path.dirname(_PROJECT_ROOT)
    if parent == _PROJECT_ROOT:  # Reached filesystem root
        break
    _PROJECT_ROOT = parent
DOOR1_USD_PATH = os.path.join(_PROJECT_ROOT, "assets", "door1.usdc")
DOOR2_USD_PATH = os.path.join(_PROJECT_ROOT, "assets", "door2.usdc")


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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Elevator
    elevator: ArticulationCfg = ELEVATOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Elevator",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=ELEVATOR_CFG.init_state.joint_pos,  # preserve original joint positions
            pos=(0.0, 0.0, 0.0),  # Match ground plane and robot z-position
        ),
    )

    # Door1 - animated door (moving, controlled via USD mesh translation)
    door1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Door1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOOR1_USD_PATH
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Door2 - static door (not moving, for reference/display)
    door2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Door2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOOR2_USD_PATH
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

