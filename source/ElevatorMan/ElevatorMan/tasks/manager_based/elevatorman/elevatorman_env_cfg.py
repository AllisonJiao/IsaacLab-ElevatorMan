# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from .elevatorman_scene_cfg import ElevatormanSceneCfg
from . import mdp as elevatorman_mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG  # isort: skip
from isaaclab.controllers.config.rmp_flow import AGIBOT_RIGHT_ARM_RMPFLOW_CFG  # isort: skip

##
# Command settings
##

# Create Ranges config with 50/50 open/close probability (outside class to avoid being treated as command term)
# Following the Isaac Lab tutorial pattern: ranges are configured separately
_door_ranges = elevatorman_mdp.ElevatorDoorCommandCfg.Ranges()
_door_ranges.open_probability = (0.5, 0.5)  # 50% probability - door randomly opens or closes on resample

@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    Following the Isaac Lab tutorial pattern for command configuration.
    The door command is triggered by button presses - when the robot pushes any elevator
    button (joint position goes negative), the door toggles between open and closed.
    """

    elevator_door = elevatorman_mdp.ElevatorDoorCommandCfg(
        elevator_name="elevator",  # Required: used to access button joints for press detection
        resampling_time_range=(999.0, 999.0),  # Disable time-based resampling (use button presses instead)
        debug_vis=False,
        ranges=_door_ranges,  # Not used in button-triggered mode, but kept for compatibility
        door_open_position=0.8,  # Joint position 0.8 = open (door opens 80 cm along -X axis)
        door_close_position=0.0,  # Joint position 0.0 = closed
    )


##
# Event settings
##


@configclass
class EventCfgElevatorman:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


#
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=mdp.ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    
    # Door command action - applies door commands from command manager to door joint
    door_action: elevatorman_mdp.DoorCommandActionCfg = elevatorman_mdp.DoorCommandActionCfg(
        asset_name="elevator",  # Required: elevator articulation containing door1_joint
        door_joint_name="door1_joint",  # Joint name for door control
        command_name="elevator_door",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ElevatormanEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the elevatorman environment."""

    # Scene settings
    scene: ElevatormanSceneCfg = ElevatormanSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # MDP managers
    commands: CommandsCfg = CommandsCfg()
    # Unused managers
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # set viewer to see the whole scene
        self.viewer.eye = [1.5, 1.8, 2.5]
        self.viewer.lookat = [-2.0, 1.0, 0]


"""
Env to Replay Sim2Lab Demonstrations with JointSpaceAction
"""


class RmpFlowAgibotElevatormanEnvCfg(ElevatormanEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events = EventCfgElevatorman()

        # Set Agibot as robot
        # Use position and rotation from animate_elevator_scene.py
        # Note: With 90° rotation, robot parts extend differently, so z-height may need adjustment
        self.scene.robot = AGIBOT_A2D_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,  # preserve original joint positions
                pos=(-2.0, 1.2, 0.0),  # z=0.0: ground is at z=0.0
                rot=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)),  # (w,x,y,z) - 90° rotation around x-axis
            ),
        )

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Agibot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_gripper_center",
            controller=AGIBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
            open_command_expr={"right_hand_joint1": 0.994, "right_.*_Support_Joint": 0.994},
            close_command_expr={"right_hand_joint1": 0.20, "right_.*_Support_Joint": 0.20},
        )

        # find joint ids for grippers
        self.gripper_joint_names = ["right_hand_joint1", "right_Right_1_Joint"]
        self.gripper_open_val = 0.994
        self.gripper_threshold = 0.2


        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # add contact force sensor for grasped checking (if needed)
        # self.scene.contact_grasp = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/right_.*_Pad_Link",
        #     update_period=0.05,
        #     history_length=6,
        #     debug_vis=True,
        #     filter_prim_paths_expr=[],
        # )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0
