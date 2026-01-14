# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for elevator door control."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

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

        # Note: Door is now a standalone USD mesh, not a joint in the elevator articulation
        # We still need elevator_name for base class, but don't use it for door control
        
        # create buffers
        # -- commands: door target position (0.0 = closed, -0.5 = open)
        self.door_command = torch.zeros(self.num_envs, 1, device=self.device)
        
        # -- metrics
        # Note: Metrics are simplified since we don't track door joint position anymore
        # The door position is controlled via USD mesh translation in the door actions
        self.metrics["door_position_error"] = torch.zeros(self.num_envs, device=self.device)
        # Store as float (0.0 = closed, 1.0 = open) to allow torch.mean() in reset()
        self.metrics["door_is_open"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Debug print: show initialization
        print(f"\033[94m[DOOR_CMD] Initialized: resampling_time_range={cfg.resampling_time_range}, "
              f"open_position={cfg.door_open_position}, close_position={cfg.door_close_position}\033[0m")

    def __str__(self) -> str:
        msg = "ElevatorDoorCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tDoor control: USD mesh translation (position delta: 0.0 to -0.5)\n"
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
        # Note: Door is now controlled via USD mesh translation, not joint positions
        # Metrics are simplified - we just track the command state, not actual door position
        # For accurate metrics, we would need to read from USD mesh, which is complex
        
        target_pos = self.door_command.squeeze(-1)
        # Position error is always 0 since we can't measure actual door position here
        # The door actions handle the actual mesh translation
        self.metrics["door_position_error"] = torch.zeros_like(target_pos)
        
        # Check if door command is open (based on command, not actual position)
        # Door is "open" if command is close to open position
        open_threshold = 0.1
        self.metrics["door_is_open"] = (
            torch.abs(target_pos - self.cfg.door_open_position) < open_threshold
        ).float()

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
        
        # Map to door positions: 0.0 -> close, -0.5 = open
        old_commands = self.door_command[env_ids, 0].clone()
        self.door_command[env_ids, 0] = (
            door_state * self.cfg.door_open_position + 
            (1.0 - door_state) * self.cfg.door_close_position
        )
        
        # Debug print: show when commands are resampled
        for i, env_id in enumerate(env_ids):
            old_cmd = old_commands[i].item()
            new_cmd = self.door_command[env_id, 0].item()
            state_str = "OPEN" if abs(new_cmd - self.cfg.door_open_position) < 0.01 else "CLOSED"
            print(f"\033[92m[DOOR_CMD] Env {env_id}: Resampled command: {old_cmd:.3f} -> {new_cmd:.3f} ({state_str})\033[0m")

    def _update_command(self):
        """Update the door command.
        
        This is called each step. For door commands, we typically don't need
        to update the command continuously (it's set via resampling).
        """
        # Debug: Print current command value periodically (every 60 steps ~ 1 second at 60Hz)
        if not hasattr(self, '_update_counter'):
            self._update_counter = 0
        self._update_counter += 1
        
        if self._update_counter % 60 == 0:
            for env_id in range(self.num_envs):
                cmd_val = self.door_command[env_id, 0].item()
                state_str = "OPEN" if abs(cmd_val - self.cfg.door_open_position) < 0.01 else "CLOSED"
                print(f"\033[96m[DOOR_CMD] Env {env_id}: Current command={cmd_val:.3f} ({state_str})\033[0m")

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
        # Note: Door is now a USD mesh, not part of elevator articulation
        # Visualization would need to be updated to work with USD meshes if needed
        pass

