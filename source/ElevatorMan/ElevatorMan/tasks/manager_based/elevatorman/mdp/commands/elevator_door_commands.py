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
        
        # Find button joint indices for door control
        # Only include button_G (ground) and buttons 2-6, exclude button_open and button_close
        # Mapping based on URDF mesh files:
        #   button_0_0_joint → button_5.dae (button 5)
        #   button_0_1_joint → button_6.dae (button 6)
        #   button_1_0_joint → button_3.dae (button 3)
        #   button_1_1_joint → button_4.dae (button 4)
        #   button_2_0_joint → button_ground.dae (button_G)
        #   button_2_1_joint → button_2.dae (button 2)
        #   button_3_0_joint → button_open.dae (EXCLUDE - open button)
        #   button_3_1_joint → button_close.dae (EXCLUDE - close button)
        button_joint_names = [
            "button_0_0_joint",  # button 5
            "button_0_1_joint",  # button 6
            "button_1_0_joint",  # button 3
            "button_1_1_joint",  # button 4
            "button_2_0_joint",  # button_G (ground)
            "button_2_1_joint",  # button 2
            # Excluded: button_3_0_joint (open), button_3_1_joint (close)
        ]
        self._button_joint_ids, _ = self._elevator.find_joints(button_joint_names)
        self._button_joint_ids = torch.as_tensor(self._button_joint_ids, device=self.device, dtype=torch.long)
        
        # Store default/initial button positions for relative movement detection
        # We'll initialize this after the first scene update when positions are available
        self._button_default_positions = None
        
        # Button press detection: button is "pressed" when moved at least 0.001 (0.1 cm) from default
        self._button_press_threshold = 0.001  # 0.1 cm movement threshold
        
        # Debouncing: track previous button states to detect button press events (not just held state)
        self._prev_button_pressed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Automatic door closing: track when door was opened and close it after delay
        self._door_open_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)  # Step when door was opened (-1 = not open)
        self._door_auto_close_delay_steps = int(3.0 / env.step_dt)  # Close door after 3 seconds (convert to steps)
        
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
              f"button_joints={len(self._button_joint_ids)} (button_G, button_2-6 only, excluding open/close), "
              f"auto_close_delay={3.0}s ({self._door_auto_close_delay_steps} steps)\033[0m")

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
        but we reset the button state tracking and re-initialize default positions.
        
        Args:
            env_ids: Environment IDs to resample commands for.
        """
        # Reset button state tracking for resampled environments
        self._prev_button_pressed[env_ids] = False
        
        # Reset door open step tracking
        self._door_open_step[env_ids] = -1
        
        # Re-initialize default button positions after reset (will be set on next _update_command call)
        # This ensures we track movement relative to the reset state
        if self._button_default_positions is not None:
            # Get current positions as new defaults after reset
            button_positions = self._elevator.data.joint_pos[:, self._button_joint_ids]
            self._button_default_positions[env_ids] = button_positions[env_ids].clone()
        
        # Reset door to closed state on reset
        self.door_command[env_ids, 0] = self.cfg.door_close_position
        
        # Debug print: show reset
        for env_id in env_ids:
            cmd_val = self.door_command[env_id, 0].item()
            state_str = "OPEN" if abs(cmd_val - self.cfg.door_open_position) < 0.01 else "CLOSED"
            print(f"\033[94m[DOOR_CMD] Env {env_id}: Reset (button state tracking cleared, defaults updated, door closed), "
                  f"current command={cmd_val:.3f} ({state_str})\033[0m")

    def _update_command(self):
        """Update the door command based on button presses and automatic closing.
        
        This is called each step. Checks if any button is pressed (moved at least 0.001 from default)
        and opens the door. Also automatically closes the door after a delay if it's open.
        """
        # Get current step counter from environment
        current_step = self._env.common_step_counter
        
        # Get current button joint positions
        button_positions = self._elevator.data.joint_pos[:, self._button_joint_ids]  # Shape: (num_envs, num_buttons)
        
        # Initialize default positions on first call (after scene is updated)
        if self._button_default_positions is None:
            self._button_default_positions = button_positions.clone()
            print(f"\033[94m[DOOR_CMD] Initialized button default positions: {self._button_default_positions[0]}\033[0m")
            return  # Skip detection on first call
        
        # Calculate movement from default position (absolute value of displacement)
        button_movements = torch.abs(button_positions - self._button_default_positions)  # Shape: (num_envs, num_buttons)
        
        # Check if any button is pressed (moved at least threshold from default) for each environment
        # A button is pressed if its movement exceeds the threshold (0.001 = 0.1 cm)
        buttons_pressed = (button_movements > self._button_press_threshold).any(dim=1)  # Shape: (num_envs,)
        
        # Detect button press events (transition from not pressed to pressed)
        button_press_events = buttons_pressed & ~self._prev_button_pressed
        
        # Handle button press events: open the door (don't toggle - always open on button press)
        for env_id in range(self.num_envs):
            if button_press_events[env_id]:
                # Open the door when button is pressed (regardless of current state)
                current_cmd = self.door_command[env_id, 0].item()
                is_currently_open = abs(current_cmd - self.cfg.door_open_position) < 0.01
                
                if not is_currently_open:
                    # Open the door
                    self.door_command[env_id, 0] = self.cfg.door_open_position
                    self._door_open_step[env_id] = current_step  # Record when door was opened
                    
                    # Debug print: show button press event
                    max_movement = button_movements[env_id].max().item()  # Get maximum button movement
                    print(f"\033[92m[DOOR_CMD] Env {env_id}: Button pressed! (movement={max_movement:.4f}m) "
                          f"Opening door: {current_cmd:.3f} -> {self.cfg.door_open_position:.3f} (OPEN)\033[0m")
                else:
                    # Door is already open, just update the open step counter
                    self._door_open_step[env_id] = current_step
                    max_movement = button_movements[env_id].max().item()
                    print(f"\033[93m[DOOR_CMD] Env {env_id}: Button pressed while door already open (movement={max_movement:.4f}m), "
                          f"resetting auto-close timer\033[0m")
        
        # Automatic door closing: close door if it's been open for too long
        for env_id in range(self.num_envs):
            current_cmd = self.door_command[env_id, 0].item()
            is_currently_open = abs(current_cmd - self.cfg.door_open_position) < 0.01
            
            if is_currently_open and self._door_open_step[env_id] >= 0:
                # Check if enough time has passed since door was opened
                steps_since_open = current_step - self._door_open_step[env_id]
                if steps_since_open >= self._door_auto_close_delay_steps:
                    # Automatically close the door
                    self.door_command[env_id, 0] = self.cfg.door_close_position
                    self._door_open_step[env_id] = -1  # Reset open step counter
                    print(f"\033[93m[DOOR_CMD] Env {env_id}: Auto-closing door after {steps_since_open} steps "
                          f"({steps_since_open * self._env.step_dt:.2f}s)\033[0m")
            elif not is_currently_open:
                # Door is closed, reset open step counter
                self._door_open_step[env_id] = -1
        
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

