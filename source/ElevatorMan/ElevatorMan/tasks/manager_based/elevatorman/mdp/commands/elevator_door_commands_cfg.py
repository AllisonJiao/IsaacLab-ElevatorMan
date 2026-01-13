# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for elevator door commands."""

from dataclasses import MISSING
from typing import Union

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from . import elevator_door_commands as door_cmd


@configclass
class ElevatorDoorCommandCfg(CommandTermCfg):
    """Configuration for elevator door command generator.
    
    This command term generates door open/close commands for the elevator.
    The command can be used to control when the elevator door should be opened or closed.
    """

    class_type: type = door_cmd.ElevatorDoorCommand
    """The class type for the command term."""

    elevator_name: Union[str, type(MISSING)] = MISSING  # type: ignore
    """Name of the elevator articulation in the scene."""

    door_joint_name: str = "door2_joint"
    """Name of the door joint to control. Defaults to 'door2_joint'."""

    door_open_position: float = -0.5
    """Target position for door open state. Defaults to -0.5 (50 cm along chosen axis)."""

    door_close_position: float = 0.0
    """Target position for door close state. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for door commands."""

        open_probability: tuple[float, float] = (0.0, 1.0)
        """Probability range for door to be open when resampled. 
        
        A value of 1.0 means door will always be open, 0.0 means always closed.
        Defaults to (0.0, 1.0) for random open/close.
        """

    ranges: Ranges = Ranges()
    """Ranges for the door commands."""

    position_only: bool = True
    """Whether to only control door position (True) or also include velocity commands (False).
    
    Defaults to True for position-only control.
    """

