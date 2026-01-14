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

        # Get elevator articulation to monitor button joints
        from isaaclab.assets import Articulation
        self._elevator: Articulation = env.scene[cfg.elevator_name]
        
        # Find button joint indices (all buttons match pattern: button_[0-3]_[0-1]_joint)
        button_joint_names = [
            "button_0_0_joint", "button_0_1_joint",
            "button_1_0_joint", "button_1_1_joint",
            "button_2_0_joint", "button_2_1_joint",
            "button_3_0_joint", "button_3_1_joint",
        ]
        self._button_joint_ids, _ = self._elevator.find_joints(button_joint_names)
        self._button_joint_ids = torch.as_tensor(self._button_joint_ids, device=self.device, dtype=torch.long)
        
        # Button press detection threshold (button is "pressed" when position < threshold)
        self._button_press_threshold = -0.01  # Negative y position indicates button press
        
        # Debouncing: track previous button states to detect button press events (not just held state)
        self._prev_button_pressed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # create buffers
        # -- commands: door target position (0.0 = closed, -0.5 = open)
        # Initialize to closed position by default
        self.door_command = torch.full(
            (self.num_envs, 1), 
            self.cfg.door_close_position, 
            device=self.device
        )
        
        # -- metrics
        # Note: Metrics are simplified since we don't track door joint position anymore
        # The door position is controlled via USD mesh translation in the door actions
        self.metrics["door_position_error"] = torch.zeros(self.num_envs, device=self.device)
        # Store as float (0.0 = closed, 1.0 = open) to allow torch.mean() in reset()
        self.metrics["door_is_open"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Debug print: show initialization
        print(f"\033[94m[DOOR_CMD] Initialized: button-triggered mode, "
              f"open_position={cfg.door_open_position}, close_position={cfg.door_close_position}, "
              f"button_joints={len(self._button_joint_ids)}\033[0m")

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
        
        This is called during reset. We don't resample based on time anymore,
        but we reset the button state tracking.
        
        Args:
            env_ids: Environment IDs to resample commands for.
        """
        # Reset button state tracking for resampled environments
        self._prev_button_pressed[env_ids] = False
        
        # Optionally reset door to closed state on reset
        # (or keep current state - uncomment below to reset to closed)
        # self.door_command[env_ids, 0] = self.cfg.door_close_position
        
        # Debug print: show reset
        for env_id in env_ids:
            cmd_val = self.door_command[env_id, 0].item()
            state_str = "OPEN" if abs(cmd_val - self.cfg.door_open_position) < 0.01 else "CLOSED"
            print(f"\033[94m[DOOR_CMD] Env {env_id}: Reset (button state tracking cleared), "
                  f"current command={cmd_val:.3f} ({state_str})\033[0m")

    def _update_command(self):
        """Update the door command based on button presses.
        
        This is called each step. Checks if any button is pressed and toggles door state.
        """
        # Get current button joint positions
        button_positions = self._elevator.data.joint_pos[:, self._button_joint_ids]  # Shape: (num_envs, num_buttons)
        
        # Check if any button is pressed (position < threshold) for each environment
        # A button is pressed if its position is below the threshold (negative y)
        buttons_pressed = (button_positions < self._button_press_threshold).any(dim=1)  # Shape: (num_envs,)
        
        # Detect button press events (transition from not pressed to pressed)
        button_press_events = buttons_pressed & ~self._prev_button_pressed
        
        # Toggle door state when button is pressed
        for env_id in range(self.num_envs):
            if button_press_events[env_id]:
                # Toggle door: if currently open, close it; if currently closed, open it
                current_cmd = self.door_command[env_id, 0].item()
                is_currently_open = abs(current_cmd - self.cfg.door_open_position) < 0.01
                
                if is_currently_open:
                    # Close the door
                    new_cmd = self.cfg.door_close_position
                    state_str = "CLOSED"
                else:
                    # Open the door
                    new_cmd = self.cfg.door_open_position
                    state_str = "OPEN"
                
                self.door_command[env_id, 0] = new_cmd
                
                # Debug print: show button press event
                button_pos = button_positions[env_id].min().item()  # Get most pressed button position
                print(f"\033[92m[DOOR_CMD] Env {env_id}: Button pressed! (pos={button_pos:.4f}) "
                      f"Toggling door: {current_cmd:.3f} -> {new_cmd:.3f} ({state_str})\033[0m")
        
        # Update previous button state for next step
        self._prev_button_pressed = buttons_pressed.clone()

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

