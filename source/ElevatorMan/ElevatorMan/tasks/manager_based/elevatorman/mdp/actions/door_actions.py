# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for applying elevator door commands."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DoorCommandAction(ActionTerm):
    """Action term that applies door commands from the command manager to the elevator door joint.
    
    This action term reads the door command from the command manager and applies it directly
    to the door joint using joint position control. It doesn't require any input actions (action_dim = 0).
    """

    cfg: actions_cfg.DoorCommandActionCfg
    """The configuration of the action term."""
    
    _door_joint_id: torch.Tensor
    """Joint ID of the door joint."""
    
    _door_current_positions: torch.Tensor
    """Current door joint positions for each environment."""
    
    _door_target_positions: torch.Tensor
    """Target door joint positions from commands for each environment."""
    
    _door_animation_speed: float
    """Speed of door animation in m/s (how fast the door moves per second)."""

    def __init__(self, cfg: actions_cfg.DoorCommandActionCfg, env: ManagerBasedEnv):
        """Initialize the door command action term.

        Args:
            cfg: The configuration parameters.
            env: The environment object.
        """
        # Initialize the action term (base class will set self._asset from cfg.asset_name)
        super().__init__(cfg, env)

        # Store command name for lookup
        self._command_name = cfg.command_name

        # Get elevator articulation
        self._elevator: Articulation = env.scene[cfg.asset_name]
        
        # Find door joint
        door_joint_ids, _ = self._elevator.find_joints([cfg.door_joint_name])
        if len(door_joint_ids) == 0:
            raise RuntimeError(
                f"Door joint '{cfg.door_joint_name}' not found in elevator articulation '{cfg.asset_name}'."
            )
        self._door_joint_id = torch.as_tensor(door_joint_ids[0], device=self.device, dtype=torch.long)
        
        # Initialize door position tracking
        self._door_current_positions = torch.zeros(self.num_envs, device=self.device)
        self._door_target_positions = torch.zeros(self.num_envs, device=self.device)
        self._door_animation_speed = 1.6  # m/s - door moves 1.6 m/s (faster for reliable closing, especially with recording)
        
        # Debug print: show initialization
        print(f"\033[94m[DOOR_ACTION] Initialized door joint '{cfg.door_joint_name}' (ID: {self._door_joint_id.item()}) "
              f"for {self.num_envs} environments with animation speed={self._door_animation_speed} m/s\033[0m")

    @property
    def action_dim(self) -> int:
        """Action dimension (zero since we don't need input actions)."""
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions (empty since we don't use input actions)."""
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (empty since we use commands instead)."""
        return torch.zeros(self.num_envs, 0, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        """Process actions (no-op since we use commands instead)."""
        # No processing needed - we use commands, not actions
        pass

    def apply_actions(self):
        """Apply the door command to the elevator door joint with smooth animation."""
        # Get the door command from the command manager
        door_command = self._env.command_manager.get_command(self._command_name)
        
        # door_command shape is (num_envs, 1)
        # Each value represents the door joint target position (e.g., 0.8 for open, 0.0 for closed)
        
        # Get simulation timestep for animation
        dt = self._env.step_dt
        
        # Squeeze to (num_envs,) shape
        door_targets = door_command.squeeze(-1)  # Shape: (num_envs,)
        
        # Check if target changed significantly (for debugging)
        old_targets = self._door_target_positions.clone()
        for env_id in range(self.num_envs):
            old_target = old_targets[env_id].item()
            new_target = door_targets[env_id].item()
            if abs(old_target - new_target) > 0.01:
                # Target changed significantly
                if new_target < 0.1 and old_target > 0.7:  # Closing from open
                    print(f"\033[95m[DOOR_ACTION] Env {env_id}: Command changed to CLOSE ({old_target:.3f} -> {new_target:.3f})\033[0m")
                elif new_target > 0.7 and old_target < 0.1:  # Opening from closed
                    print(f"\033[95m[DOOR_ACTION] Env {env_id}: Command changed to OPEN ({old_target:.3f} -> {new_target:.3f})\033[0m")
        
        # Update target positions
        self._door_target_positions = door_targets.clone()
        
        # Get current door joint positions
        # Convert joint_id to int for indexing
        door_joint_id_int = int(self._door_joint_id.item())
        joint_pos_data = self._elevator.data.joint_pos  # Shape: (num_envs, num_joints)
        self._door_current_positions = joint_pos_data[:, door_joint_id_int]
        
        # Calculate movement step size based on animation speed
        # Joint limits are 0.0 to 0.8 (80 cm), so 0.8 m max travel
        max_step_size = self._door_animation_speed * dt / 0.8  # Normalize by joint range (0.8 m)
        
        # Interpolate towards target position for smooth animation
        delta_to_target = self._door_target_positions - self._door_current_positions
        
        # Apply smooth animation
        abs_delta = torch.abs(delta_to_target)
        step_size = torch.minimum(abs_delta, torch.full_like(abs_delta, max_step_size))
        step = torch.sign(delta_to_target) * step_size
        
        # Only move if delta is significant (> 0.001)
        move_mask = abs_delta > 0.001
        new_positions = torch.where(
            move_mask,
            self._door_current_positions + step,
            self._door_current_positions
        )
        
        # Clamp to joint limits
        new_positions = torch.clamp(new_positions, 0.0, 0.8)
        
        # Set door joint position target directly (set_joint_position_target can target specific joints)
        door_joint_id_int = int(self._door_joint_id.item() if isinstance(self._door_joint_id, torch.Tensor) else self._door_joint_id)
        self._elevator.set_joint_position_target(
            new_positions.unsqueeze(-1),  # Shape: (num_envs, 1)
            joint_ids=door_joint_id_int  # Single joint ID
        )
        self._elevator.write_data_to_sim()

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the door to initial position for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Ensure env_ids is a 1D tensor (handle scalar case)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.dim() == 0:  # 0-dimensional tensor (scalar)
                env_ids = env_ids.unsqueeze(0)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device)
            if env_ids.dim() == 0:
                env_ids = env_ids.unsqueeze(0)
        
        # Reset door positions to closed (0.0)
        self._door_current_positions[env_ids] = 0.0
        self._door_target_positions[env_ids] = 0.0
        
        # Set door joint to closed position (0.0) for specified environments
        closed_positions = torch.zeros(len(env_ids), 1, device=self.device)
        door_joint_id_int = int(self._door_joint_id.item())
        self._elevator.set_joint_position_target(
            closed_positions,  # Shape: (len(env_ids), 1)
            joint_ids=door_joint_id_int,  # Single joint ID
            env_ids=env_ids
        )
        self._elevator.write_data_to_sim()
