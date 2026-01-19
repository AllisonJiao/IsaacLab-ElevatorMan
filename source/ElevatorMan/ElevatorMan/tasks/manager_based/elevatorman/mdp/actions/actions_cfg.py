# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for action terms."""

from dataclasses import MISSING
from typing import Union

from isaaclab.envs.mdp.actions.actions_cfg import ActionTermCfg
from isaaclab.utils import configclass

from . import door_actions


@configclass
class DoorCommandActionCfg(ActionTermCfg):
    """Configuration for door command action term."""

    class_type: type = door_actions.DoorCommandAction
    """The class type for the action term."""

    asset_name: Union[str, type(MISSING)] = MISSING  # type: ignore
    """Name of the elevator articulation in the scene (required for door joint control)."""

    door_joint_name: str = "door1_joint"
    """Name of the door joint to control. Defaults to 'door1_joint'."""

    command_name: str = "elevator_door"
    """Name of the command term to read from. Defaults to 'elevator_door'."""

