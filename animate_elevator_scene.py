# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom elevator to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--robot_animation_range", type=float, default=0.1, help="Range of robot arm animation (0.0-1.0, where 1.0 = full 2π rotation)")
parser.add_argument("--simultaneous_arms", action="store_true", help="Move both arms simultaneously. If not set, arms move sequentially (left first, then right)")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import os

import isaaclab.sim as sim_utils
# import prims as prim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG  # isort: skip
from cfg.elevator import ELEVATOR_CFG

# USD access for direct mesh control
import omni.usd
from pxr import UsdGeom, Gf, Usd

# Get door USD asset paths
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_CUR_DIR)
DOOR1_USD_PATH = os.path.join(_PROJECT_ROOT, "assets", "door1.usdc")
DOOR2_USD_PATH = os.path.join(_PROJECT_ROOT, "assets", "door2.usdc")

@configclass
class ElevatorSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # elevator (for buttons)
    elevator: ArticulationCfg = ELEVATOR_CFG.replace(prim_path="/World/elevator")

    # Door1 - animated door (moving)
    door1 = AssetBaseCfg(
        prim_path="/World/Door1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOOR1_USD_PATH
        ),
    )

    # Door2 - static door (not moving)
    door2 = AssetBaseCfg(
        prim_path="/World/Door2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOOR2_USD_PATH
        ),
    )

    # robot
    agibot: ArticulationCfg = AGIBOT_A2D_CFG.replace(
        prim_path="/World/Agibot",
        spawn=AGIBOT_A2D_CFG.spawn.replace(
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,  # preserve original joint positions
            pos=(-2.0, 2.0, 0.0),
            rot=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)), # (w,x,y,z)
        ),
    )

def set_robot_pose_demo(
    agibot: Articulation, 
    phase: float, 
    left_joint_groups: dict[str, torch.Tensor],
    right_joint_groups: dict[str, torch.Tensor],
    robot_animation_range: float = 1.0,
    symmetric_base: bool = True,
    sequential: bool = True,
    sequential_linkages: bool = True,
    cached_symmetric_refs: dict[str, torch.Tensor] = None,
    cached_symmetric_ref_all: torch.Tensor = None,
    lift_body_joint_ids: torch.Tensor = None,
    lift_body_target: torch.Tensor = None
):
    """Set robot joints based on phase for smooth animation.
    
    Args:
        agibot: The robot articulation
        phase: Normalized phase value [0, 1] for animation cycle
        left_joint_groups: Dict mapping group names to tensors of left arm joint indices (e.g., {"shoulder": [...], "forearm": [...], "gripper": [...]})
        right_joint_groups: Dict mapping group names to tensors of right arm joint indices
        robot_animation_range: Multiplier for animation range (default 1.0 = full 2π rotation)
        symmetric_base: If True, ensures symmetric starting positions for left and right arms
        sequential: If True, moves arms one at a time (left first, then right). If False, moves both simultaneously.
        sequential_linkages: If True, animates joint groups (shoulder, forearm, gripper) sequentially within each arm
    """
    # Get all joint IDs from groups for checking
    all_left_ids = torch.cat([ids for ids in left_joint_groups.values()]) if left_joint_groups else torch.tensor([], dtype=torch.long, device=agibot.device)
    all_right_ids = torch.cat([ids for ids in right_joint_groups.values()]) if right_joint_groups else torch.tensor([], dtype=torch.long, device=agibot.device)
    
    if len(all_left_ids) == 0 and len(all_right_ids) == 0:
        return
    
    # Calculate joint positions based on phase (smooth rotation)
    joint_pos_target = agibot.data.default_joint_pos.clone()
    
    # Get group names in order (shoulder, forearm, gripper)
    group_names = list(left_joint_groups.keys()) if left_joint_groups else list(right_joint_groups.keys())
    num_groups = len(group_names)
    
    def animate_joint_group(group_ids, group_phase, symmetric_ref_group=None, is_left=True):
        """Helper to animate a joint group based on phase"""
        if len(group_ids) == 0:
            return

        animation_offset = group_phase * (2 * torch.pi * robot_animation_range)
        if symmetric_ref_group is not None:
            if is_left:
                joint_pos_target[:, group_ids] = symmetric_ref_group + animation_offset
            else:
                joint_pos_target[:, group_ids] = symmetric_ref_group - animation_offset
        else:
            if is_left:
                joint_pos_target[:, group_ids] += animation_offset
            else:
                joint_pos_target[:, group_ids] -= animation_offset
    
    if symmetric_base and len(all_left_ids) > 0 and len(all_right_ids) > 0:
        # For symmetric motion: use average of left/right defaults as symmetric reference
        # Use cached references if provided, otherwise compute them
        if cached_symmetric_refs is not None and cached_symmetric_ref_all is not None:
            symmetric_refs = cached_symmetric_refs
            symmetric_ref_all = cached_symmetric_ref_all
        else:
            # Compute symmetric ref for each group separately
            symmetric_refs = {}
            for group_name in group_names:
                left_group_ids = left_joint_groups[group_name]
                right_group_ids = right_joint_groups[group_name]
                if len(left_group_ids) > 0 and len(right_group_ids) > 0:
                    left_default_group = agibot.data.default_joint_pos[:, left_group_ids]
                    right_default_group = agibot.data.default_joint_pos[:, right_group_ids]
                    symmetric_refs[group_name] = (left_default_group + right_default_group) / 2.0
            
            # Also compute for all joints together (for non-sequential-linkages case)
            left_default_all = agibot.data.default_joint_pos[:, all_left_ids]
            right_default_all = agibot.data.default_joint_pos[:, all_right_ids]
            symmetric_ref_all = (left_default_all + right_default_all) / 2.0
        
        if sequential:
            # Sequential movement: left arm moves first (phase 0-0.5), then right arm (phase 0.5-1.0)
            if phase < 0.5:
                left_phase = phase * 2.0  # Map [0, 0.5] to [0, 1]
                
                if sequential_linkages and num_groups > 0:
                    # Animate left arm groups sequentially
                    group_phase_range = 1.0 / num_groups
                    for i, group_name in enumerate(group_names):
                        group_start = i * group_phase_range
                        group_end = (i + 1) * group_phase_range
                        if left_phase >= group_end:
                            # This group is complete, set to final position
                            animate_joint_group(left_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=True)
                        elif left_phase >= group_start:
                            # This group is currently animating
                            group_phase = (left_phase - group_start) / group_phase_range
                            animate_joint_group(left_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=True)
                        else:
                            # This group hasn't started yet, keep at reference position
                            joint_pos_target[:, left_joint_groups[group_name]] = symmetric_refs[group_name]
                    
                    # Right arm stays at reference
                    for group_name in group_names:
                        joint_pos_target[:, right_joint_groups[group_name]] = symmetric_refs[group_name]
                else:
                    # Animate all left joints together
                    animation_offset = left_phase * (2 * torch.pi * robot_animation_range)
                    joint_pos_target[:, all_left_ids] = symmetric_ref_all + animation_offset
                    joint_pos_target[:, all_right_ids] = symmetric_ref_all
            else:
                right_phase = (phase - 0.5) * 2.0  # Map [0.5, 1.0] to [0, 1]
                
                # Left arm at final position
                left_final_offset = (2 * torch.pi * robot_animation_range)
                joint_pos_target[:, all_left_ids] = symmetric_ref_all + left_final_offset
                
                if sequential_linkages and num_groups > 0:
                    # Animate right arm groups sequentially
                    group_phase_range = 1.0 / num_groups
                    for i, group_name in enumerate(group_names):
                        group_start = i * group_phase_range
                        group_end = (i + 1) * group_phase_range
                        if right_phase >= group_end:
                            # This group is complete, set to final position
                            animate_joint_group(right_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=False)
                        elif right_phase >= group_start:
                            # This group is currently animating
                            group_phase = (right_phase - group_start) / group_phase_range
                            animate_joint_group(right_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=False)
                        else:
                            # This group hasn't started yet, keep at reference position
                            joint_pos_target[:, right_joint_groups[group_name]] = symmetric_refs[group_name]
                else:
                    # Animate all right joints together
                    animation_offset = right_phase * (2 * torch.pi * robot_animation_range)
                    joint_pos_target[:, all_right_ids] = symmetric_ref_all - animation_offset
        else:
            # Simultaneous movement: both arms move together
            if sequential_linkages and num_groups > 0:
                group_phase_range = 1.0 / num_groups
                for i, group_name in enumerate(group_names):
                    group_start = i * group_phase_range
                    group_end = (i + 1) * group_phase_range
                    if phase >= group_end:
                        # This group is complete, set to final position
                        animate_joint_group(left_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=True)
                        animate_joint_group(right_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=False)
                    elif phase >= group_start:
                        # This group is currently animating
                        group_phase = (phase - group_start) / group_phase_range
                        animate_joint_group(left_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=True)
                        animate_joint_group(right_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=False)
                    else:
                        # This group hasn't started yet, keep at reference position
                        joint_pos_target[:, left_joint_groups[group_name]] = symmetric_refs[group_name]
                        joint_pos_target[:, right_joint_groups[group_name]] = symmetric_refs[group_name]
            else:
                animation_offset = phase * (2 * torch.pi * robot_animation_range)
                joint_pos_target[:, all_left_ids] = symmetric_ref_all + animation_offset
                joint_pos_target[:, all_right_ids] = symmetric_ref_all - animation_offset
    else:
        # Non-symmetric base - simplified version (can be expanded if needed)
        animation_offset = phase * (2 * torch.pi * robot_animation_range) if not sequential else (phase * 2.0 if phase < 0.5 else (phase - 0.5) * 2.0) * (2 * torch.pi * robot_animation_range)
        if len(all_left_ids) > 0:
            joint_pos_target[:, all_left_ids] += animation_offset if phase < 0.5 or not sequential else (2 * torch.pi * robot_animation_range)
        if len(all_right_ids) > 0:
            joint_pos_target[:, all_right_ids] -= animation_offset if phase >= 0.5 or not sequential else 0
    
    # Apply lift_body animation if provided (this preserves arm joint targets)
    if lift_body_joint_ids is not None and lift_body_target is not None and len(lift_body_joint_ids) > 0:
        joint_pos_target[:, lift_body_joint_ids] = lift_body_target
    
    # Clamp to joint limits
    joint_pos_target = joint_pos_target.clamp_(
        agibot.data.soft_joint_pos_limits[..., 0], 
        agibot.data.soft_joint_pos_limits[..., 1]
    )
    agibot.set_joint_position_target(joint_pos_target)
    agibot.write_data_to_sim()


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    agibot: Articulation,
    elevator: Articulation,
    left_joint_groups: dict[str, torch.Tensor],
    right_joint_groups: dict[str, torch.Tensor],
    cached_symmetric_refs: dict[str, torch.Tensor],
    cached_symmetric_ref_all: torch.Tensor,
    elevator_button_ids: torch.Tensor,
    lift_body_joint_ids: torch.Tensor,
    door1_xform: UsdGeom.XformCommonAPI,
    door1_init_t: Gf.Vec3d,
    robot_animation_range: float,
    lift_body_range: float,
):
    """Run the simulation loop with robot and elevator animations."""
    # Animation parameters
    count = 0
    period = 500
    open_position = -0.5  # Maximum opening position (negative X)
    close_position = 0.0  # Closed position

    print("[INFO] Done. Close the window to exit.")

    while simulation_app.is_running():
        # Reset robot and elevator to default positions at the start of each period
        if count % period == 0:
            # Reset robot joint positions to default
            agibot.write_joint_state_to_sim(
                agibot.data.default_joint_pos.clone(),
                agibot.data.default_joint_vel.clone()
            )
            agibot.reset()
            
            # Reset elevator button positions to default
            elevator.write_joint_state_to_sim(
                elevator.data.default_joint_pos.clone(),
                elevator.data.default_joint_vel.clone()
            )
            elevator.reset()
            
            # Reset door1 position to closed (initial position)
            door1_xform.SetTranslate(door1_init_t, Usd.TimeCode.Default())
            
            count = 0
        
        # Calculate phase for animations
        phase = count % period
        alpha = phase / max(1, period - 1)  # Normalized phase [0, 1] for robot animation

        # Calculate door1 animation: open (0→-0.5) then close (-0.5→0)
        # First half: opening (0→-0.5), second half: closing (-0.5→0)
        if phase < period / 2:
            # Opening phase: 0 to -0.5
            t = phase / (period / 2.0)  # Normalized phase [0, 1] for opening
            door_delta = close_position + t * (open_position - close_position)  # Goes from 0.0 to -0.5
        else:
            # Closing phase: -0.5 to 0
            t = (phase - period / 2.0) / (period / 2.0)  # Normalized phase [0, 1] for closing
            door_delta = open_position + t * (close_position - open_position)  # Goes from -0.5 to 0.0

        # Update door1 position using USD Xform (along negative X axis)
        new_t = Gf.Vec3d(door1_init_t[0] + door_delta, door1_init_t[1], door1_init_t[2])
        door1_xform.SetTranslate(new_t, Usd.TimeCode.Default())

        # Update elevator button positions - press down gradually over the period
        # Button press animation: starts at 0, reaches max press at phase 0.5, stays pressed
        joint_pos_target = elevator.data.default_joint_pos.clone()
        button_press_delta = min(phase / (period / 2.0), 1.0) * 0.05  # Max press distance of 0.05
        joint_pos_target[:, elevator_button_ids] += button_press_delta
        
        # Clamp button joints to their limits
        joint_pos_target = joint_pos_target.clamp_(
            elevator.data.soft_joint_pos_limits[..., 0], elevator.data.soft_joint_pos_limits[..., 1]
        )
        elevator.set_joint_position_target(joint_pos_target)
        elevator.write_data_to_sim()

        # Calculate lift_body animation target (using door delta for consistency)
        lift_body_target = None
        if len(lift_body_joint_ids) > 0:
            # Apply lift_body animation: default position + scaled door_delta
            lift_body_target = agibot.data.default_joint_pos[:, lift_body_joint_ids] + (abs(door_delta) * lift_body_range)

        # Update robot pose using phase-based animation (with sequential linkage movement)
        # This includes both arm animation and lift_body animation, set together to avoid overwriting
        set_robot_pose_demo(
            agibot, alpha, left_joint_groups, right_joint_groups, robot_animation_range,
            sequential=not args_cli.simultaneous_arms, sequential_linkages=True,
            cached_symmetric_refs=cached_symmetric_refs,
            cached_symmetric_ref_all=cached_symmetric_ref_all,
            lift_body_joint_ids=lift_body_joint_ids if len(lift_body_joint_ids) > 0 else None,
            lift_body_target=lift_body_target
        )

        if count % 20 == 0:
            print(f"[door1-mesh] delta={door_delta:+.4f} translate={new_t}")

        sim.step()
        scene.update(sim.get_physics_dt())

        count += 1


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = ElevatorSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Access the robot articulation (because we used ArticulationCfg)
    agibot: Articulation = scene["agibot"]
    elevator: Articulation = scene["elevator"]

    # Setup door1 mesh transform animation (door2 stays static)
    DOOR1_PRIM_PATH = "/World/Door1"
    stage = omni.usd.get_context().get_stage()
    door1_prim = stage.GetPrimAtPath(DOOR1_PRIM_PATH)
    
    if not door1_prim.IsValid():
        raise RuntimeError(f"Door1 prim not found at: {DOOR1_PRIM_PATH}")

    door1_xform = UsdGeom.XformCommonAPI(door1_prim)

    # Cache initial transform (we'll only offset translation)
    tc = Usd.TimeCode.Default()
    init_t, init_r, init_s, init_pivot, _ = door1_xform.GetXformVectors(tc)
    init_t = Gf.Vec3d(init_t)  # make sure it's a Vec3d
    print("[INFO] Door1 initial translate:", init_t)

    # Setup robot joint animation - organize into groups: shoulder (1-3), forearm (4-5), gripper (6-7)
    left_shoulder_names = ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3"]
    left_forearm_names = ["left_arm_joint4", "left_arm_joint5"]
    left_gripper_names = ["left_arm_joint6", "left_arm_joint7"]

    right_shoulder_names = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3"]
    right_forearm_names = ["right_arm_joint4", "right_arm_joint5"]
    right_gripper_names = ["right_arm_joint6", "right_arm_joint7"]

    # Find joint indices for each group
    left_shoulder_ids, _ = agibot.find_joints(left_shoulder_names)
    left_forearm_ids, _ = agibot.find_joints(left_forearm_names)
    left_gripper_ids, _ = agibot.find_joints(left_gripper_names)

    right_shoulder_ids, _ = agibot.find_joints(right_shoulder_names)
    right_forearm_ids, _ = agibot.find_joints(right_forearm_names)
    right_gripper_ids, _ = agibot.find_joints(right_gripper_names)

    # Organize into groups
    left_joint_groups = {
        "shoulder": torch.as_tensor(left_shoulder_ids, device=agibot.device, dtype=torch.long),
        "forearm": torch.as_tensor(left_forearm_ids, device=agibot.device, dtype=torch.long),
        "gripper": torch.as_tensor(left_gripper_ids, device=agibot.device, dtype=torch.long),
    }
    right_joint_groups = {
        "shoulder": torch.as_tensor(right_shoulder_ids, device=agibot.device, dtype=torch.long),
        "forearm": torch.as_tensor(right_forearm_ids, device=agibot.device, dtype=torch.long),
        "gripper": torch.as_tensor(right_gripper_ids, device=agibot.device, dtype=torch.long),
    }

    total_left = sum(len(ids) for ids in left_joint_groups.values())
    total_right = sum(len(ids) for ids in right_joint_groups.values())

    if total_left > 0 or total_right > 0:
        print(f"[INFO] Organized arm joints into groups:")
        print(f"  Left arm - Shoulder: {len(left_joint_groups['shoulder'])}, Forearm: {len(left_joint_groups['forearm'])}, Gripper: {len(left_joint_groups['gripper'])}")
        print(f"  Right arm - Shoulder: {len(right_joint_groups['shoulder'])}, Forearm: {len(right_joint_groups['forearm'])}, Gripper: {len(right_joint_groups['gripper'])}")

        # Ensure data is updated
        scene.update(sim.get_physics_dt())

        # Cache symmetric reference positions once (computed from default positions)
        # This ensures consistent symmetry across all animation cycles
        group_names = list(left_joint_groups.keys())
        cached_symmetric_refs = {}
        all_left_ids = torch.cat([ids for ids in left_joint_groups.values()])
        all_right_ids = torch.cat([ids for ids in right_joint_groups.values()])
        for group_name in group_names:
            left_group_ids = left_joint_groups[group_name]
            right_group_ids = right_joint_groups[group_name]
            if len(left_group_ids) > 0 and len(right_group_ids) > 0:
                left_default_group = agibot.data.default_joint_pos[:, left_group_ids]
                right_default_group = agibot.data.default_joint_pos[:, right_group_ids]
                cached_symmetric_refs[group_name] = (left_default_group + right_default_group) / 2.0
        cached_symmetric_ref_all = (agibot.data.default_joint_pos[:, all_left_ids] + agibot.data.default_joint_pos[:, all_right_ids]) / 2.0
    else:
        left_joint_groups = {}
        right_joint_groups = {}
        cached_symmetric_refs = {}
        cached_symmetric_ref_all = None
        print("[WARN] No arm joints found for animation. Robot will use default pose.")

    # Setup elevator button joints (doors are now standalone USD meshes)
    elevator_button_joint_names = ["button_0_0_joint", "button_0_1_joint", "button_1_0_joint", "button_1_1_joint", "button_2_0_joint", "button_2_1_joint", "button_3_0_joint", "button_3_1_joint"]
    elevator_button_ids, _ = elevator.find_joints(elevator_button_joint_names)
    elevator_button_ids = torch.as_tensor(elevator_button_ids, device=elevator.device, dtype=torch.long)
    
    print(f"[INFO] Button joint IDs: {elevator_button_ids}, Shape: {elevator_button_ids.shape}")

    # Setup robot's joint_lift_body prismatic joint animation (for testing/reference)
    lift_body_joint_names = ["joint_lift_body"]
    lift_body_joint_ids, _ = agibot.find_joints(lift_body_joint_names)
    lift_body_joint_ids = torch.as_tensor(lift_body_joint_ids, device=agibot.device, dtype=torch.long)

    robot_animation_range = args_cli.robot_animation_range
    lift_body_range = 0.2  # Range for joint_lift_body animation (meters)

    # Run the simulator
    run_simulator(
        sim, scene, agibot, elevator,
        left_joint_groups, right_joint_groups,
        cached_symmetric_refs, cached_symmetric_ref_all,
        elevator_button_ids, lift_body_joint_ids,
        door1_xform, init_t,
        robot_animation_range, lift_body_range
    )

    simulation_app.close()


if __name__ == "__main__":
    main()