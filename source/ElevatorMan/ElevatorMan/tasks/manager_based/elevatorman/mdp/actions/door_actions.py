# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for applying elevator door commands."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DoorCommandAction(ActionTerm):
    """Action term that applies door commands from the command manager to the elevator door joint.
    
    This action term reads the door command from the command manager and applies it directly
    to the door joint. It doesn't require any input actions (action_dim = 0).
    """

    cfg: actions_cfg.DoorCommandActionCfg
    """The configuration of the action term."""

    _door_joint_idx: int
    """Index of the door joint."""

    def __init__(self, cfg: actions_cfg.DoorCommandActionCfg, env: ManagerBasedEnv):
        """Initialize the door command action term.

        Args:
            cfg: The configuration parameters.
            env: The environment object.
        """
        # Initialize the action term (base class will set self._asset from cfg.asset_name)
        super().__init__(cfg, env)

        # Find door joint index
        try:
            self._door_joint_idx = self._asset.joint_names.index(cfg.door_joint_name)
        except ValueError:
            raise ValueError(
                f"Door joint '{cfg.door_joint_name}' not found in elevator articulation '{cfg.asset_name}'. "
                f"Available joints: {self._asset.joint_names}"
            )

        # Store command name for lookup
        self._command_name = cfg.command_name

        # This action term doesn't require any input actions
        # We'll just apply the command directly

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
        """Apply the door command to the elevator door joint."""
        # Get the door command from the command manager
        door_command = self._env.command_manager.get_command(self._command_name)
        
        # door_command shape is (num_envs, 1)
        # Extract the door position value
        door_position = door_command.squeeze(-1)  # Shape: (num_envs,)
        
        # Get current joint positions for all joints
        current_positions = self._asset.data.joint_pos.clone()
        
        # Update only the door joint position
        current_positions[:, self._door_joint_idx] = door_position
        
        # Apply the updated position to the door joint
        self._asset.set_joint_position_target(
            current_positions,
            joint_ids=[self._door_joint_idx]
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the action term (no-op)."""
        pass

