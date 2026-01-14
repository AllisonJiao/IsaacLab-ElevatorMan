# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for applying elevator door commands."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.usd
from pxr import UsdGeom, Gf

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sim.utils.queries import find_matching_prim_paths

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DoorCommandAction(ActionTerm):
    """Action term that applies door commands from the command manager to the elevator door mesh.
    
    This action term reads the door command from the command manager and applies it directly
    to the door mesh using USD xformOp:translate. It doesn't require any input actions (action_dim = 0).
    """

    cfg: actions_cfg.DoorCommandActionCfg
    """The configuration of the action term."""

    _door_prim_paths: list[str]
    """List of door prim paths for each environment."""

    _door_translate_attrs: list
    """List of door translate attributes for each environment."""

    _door_init_translates: list[tuple[float, float, float]]
    """List of initial door translate positions for each environment."""

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

        # Resolve door prim paths for all environments
        door_prim_path_expr = cfg.door_prim_path
        if "{ENV_REGEX_NS}" in door_prim_path_expr:
            # Replace with actual environment namespace pattern
            env_ns = env.scene.env_regex_ns
            door_prim_path_expr = door_prim_path_expr.replace("{ENV_REGEX_NS}", env_ns)
        
        # Find all matching prim paths for all environments
        self._door_prim_paths = find_matching_prim_paths(door_prim_path_expr)
        if len(self._door_prim_paths) != self.num_envs:
            raise RuntimeError(
                f"Found {len(self._door_prim_paths)} door prim paths for {self.num_envs} environments. "
                f"Expected {self.num_envs}. Prim path expression: {door_prim_path_expr}"
            )
        
        # Get USD stage
        stage = omni.usd.get_context().get_stage()
        
        # Initialize door translate attributes and cache initial positions
        self._door_translate_attrs = []
        self._door_init_translates = []
        
        for env_id, prim_path in enumerate(self._door_prim_paths):
            door_prim = stage.GetPrimAtPath(prim_path)
            
            if not door_prim.IsValid():
                raise RuntimeError(f"Door prim not found at: {prim_path} for environment {env_id}")
            
            # Directly access translate attribute (avoid XformCommonAPI incompatibility)
            translate_attr = door_prim.GetAttribute("xformOp:translate")
            
            # If translate attribute doesn't exist, try to get/create it via Xformable
            if not translate_attr or not translate_attr.IsValid():
                # Check for different possible translate op names
                for op_name in ["xformOp:translate", "xformOp:translation", "xformOp:translateX"]:
                    attr = door_prim.GetAttribute(op_name)
                    if attr and attr.IsValid():
                        translate_attr = attr
                        break
                
                # If still not found, create one using Xformable
                if not translate_attr or not translate_attr.IsValid():
                    door_xformable = UsdGeom.Xformable(door_prim)  # type: ignore
                    translate_op = door_xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)  # type: ignore
                    translate_attr = translate_op.GetAttr()
            
            self._door_translate_attrs.append(translate_attr)
            
            # Cache initial transform
            init_t = translate_attr.Get()
            if init_t is None:
                init_t = (0.0, 0.0, 0.0)
            else:
                # Convert to tuple if it's a Vec3d
                if hasattr(init_t, '__iter__') and len(init_t) == 3:
                    init_t = tuple(init_t)
                else:
                    init_t = (0.0, 0.0, 0.0)
            self._door_init_translates.append(init_t)

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
        """Apply the door command to the elevator door mesh using USD translate."""
        # Get the door command from the command manager
        door_command = self._env.command_manager.get_command(self._command_name)
        
        # door_command shape is (num_envs, 1)
        # Each value represents the door delta position (e.g., -0.5 for open, 0.0 for closed)
        
        # Convert to CPU numpy/tuple for USD attribute setting
        door_deltas = door_command.cpu().squeeze(-1).numpy()
        
        # Apply door translation for each environment
        for env_id in range(self.num_envs):
            door_delta = float(door_deltas[env_id])
            init_t = self._door_init_translates[env_id]
            
            # Update door position: initial position + delta along X axis
            new_t = (init_t[0] + door_delta, init_t[1], init_t[2])
            self._door_translate_attrs[env_id].Set(new_t)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the door to initial position for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset door positions to initial state
        env_ids_list = env_ids.cpu().tolist()
        for env_id in env_ids_list:
            init_t = self._door_init_translates[env_id]
            self._door_translate_attrs[env_id].Set(init_t)

