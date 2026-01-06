# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Elevator.

The following configurations are available:

* :obj:`ELEVATOR`: Elevator


"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get the absolute path to the elevator asset
# This assumes the cfg directory is at the project root level
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_CUR_DIR)
ELEVATOR_ASSET_PATH = os.path.join(_PROJECT_ROOT, "assets", "Collected_elevator_urdf", "elevator.usd")

##
# Configuration
##

ELEVATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ELEVATOR_ASSET_PATH}",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "button_0_0_joint": 0.0,
            "button_0_1_joint": 0.0,
            "button_1_0_joint": 0.0,
            "button_1_1_joint": 0.0,
            "button_2_0_joint": 0.0,
            "button_2_1_joint": 0.0,
            "button_3_0_joint": 0.0,
            "button_3_1_joint": 0.0,
            "door2_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),  # init pos of the articulation for teleop
    ),
    actuators={
        # Elevator buttons
        "elevator_buttons": ImplicitActuatorCfg(
            joint_names_expr=["button_[0-3]_[0-1]_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=50.0,
            damping=10.0,
        ),
        # Elevator doors
        "elevator_doors": ImplicitActuatorCfg(
            joint_names_expr=["door2_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100,
            stiffness=100.0,
            damping=20.0,
        ),
    },
)