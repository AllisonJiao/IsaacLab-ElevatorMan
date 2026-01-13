# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for elevator door control."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import elevator_door_commands_cfg as door_cmd_cfgs


class ElevatorDoorCommand(CommandTerm):
    """Elevator door command generator.
    
    This command term generates door open/close commands for the elevator.
    The command can be resampled at specified intervals to change the door state.
    
    Outputs:
        The command buffer has shape (num_envs, 1): `(door_target_position)`.
        - 0.0 = closed
        - -0.5 = open (50 cm along chosen axis)
        - Values in between represent partial opening
    
    Metrics:
        `door_position_error` is computed between the commanded door position
        and the current door joint position.
    """

    cfg: door_cmd_cfgs.ElevatorDoorCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: door_cmd_cfgs.ElevatorDoorCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the elevator articulation
        self.elevator: Articulation = env.scene[cfg.elevator_name]
        
        # Find door joint index
        self._door_joint_idx = None
        try:
            self._door_joint_idx = self.elevator.joint_names.index(cfg.door_joint_name)
        except ValueError:
            raise ValueError(
                f"Door joint '{cfg.door_joint_name}' not found in elevator articulation '{cfg.elevator_name}'. "
                f"Available joints: {self.elevator.joint_names}"
            )

        # create buffers
        # -- commands: door target position (0.0 = closed, -0.5 = open)
        self.door_command = torch.zeros(self.num_envs, 1, device=self.device)
        
        # -- metrics
        self.metrics["door_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["door_is_open"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def __str__(self) -> str:
        msg = "ElevatorDoorCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tDoor joint: {self.cfg.door_joint_name} (index: {self._door_joint_idx})\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired door position command. Shape is (num_envs, 1).
        
        Values range from 0.0 (closed) to -0.5 (open, 50 cm along chosen axis).
        """
        return self.door_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics based on current door state."""
        # Get current door joint positions
        current_door_pos = self.elevator.data.joint_pos[:, self._door_joint_idx]
        
        # Compute position error
        target_pos = self.door_command.squeeze(-1)
        self.metrics["door_position_error"] = torch.abs(current_door_pos - target_pos)
        
        # Check if door is considered open (within threshold of open position)
        open_threshold = 0.1  # Consider door open if within 0.1 of target
        self.metrics["door_is_open"] = (
            torch.abs(current_door_pos - self.cfg.door_open_position) < open_threshold
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample door commands for specified environments.
        
        Args:
            env_ids: Environment IDs to resample commands for.
        """
        # Sample door state based on open probability range
        r = torch.empty(len(env_ids), device=self.device)
        open_prob = r.uniform_(*self.cfg.ranges.open_probability)
        
        # Determine door state: -0.5 = open, 0.0 = closed
        # Use probability to decide: if random value < open_prob, door should be open
        door_state = (r.uniform_(0.0, 1.0) < open_prob).float()
        
        # Map to door positions: 0.0 -> close, -0.5 -> open
        self.door_command[env_ids, 0] = (
            door_state * self.cfg.door_open_position + 
            (1.0 - door_state) * self.cfg.door_close_position
        )

    def _update_command(self):
        """Update the door command.
        
        This is called each step. For door commands, we typically don't need
        to update the command continuously (it's set via resampling).
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization for the door command.
        
        Args:
            debug_vis: Whether to enable debug visualization.
        """
        # TODO: Implement debug visualization if needed
        # This could visualize the target door position vs current position
        if debug_vis:
            # Create markers to visualize door state
            pass
        else:
            # Hide markers
            pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        
        Args:
            event: The timeline event.
        """
        # TODO: Implement debug visualization callback
        # This could update markers showing door target vs current position
        if not self.elevator.is_initialized:
            return
        # Update visualization markers here if implemented
        pass

