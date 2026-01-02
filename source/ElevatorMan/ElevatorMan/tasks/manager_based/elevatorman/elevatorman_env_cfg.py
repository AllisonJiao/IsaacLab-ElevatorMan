# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

# from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG
# from cfg.elevator import ELEVATOR_CFG  # Commented out to debug arm wiggling
from isaaclab.controllers.config.rmp_flow import AGIBOT_RIGHT_ARM_RMPFLOW_CFG  # isort: skip

##
# Scene definition
##


@configclass
class EventCfgPlaceToy2Box:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    # init_elevator_position = EventTerm(
    #     func=franka_stack_events.randomize_object_pose,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "asset_cfgs": [SceneEntityCfg("elevator")],
    #     },
    # )


@configclass
class ElevatormanSceneCfg(InteractiveSceneCfg):
    """Configuration for an elevator man scene."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # articulations
    # elevator: ArticulationCfg = ELEVATOR_CFG.replace(prim_path="/World/elevator")  # Commented out to debug arm wiggling


##
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

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # grasp = ObsTerm(
        #     func=place_mdp.object_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("elevator"),  # Commented out to debug arm wiggling
        #         "diff_threshold": 0.05,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # subtask_terms: SubtaskCfg = SubtaskCfg()  # Commented out since elevator is disabled


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class ElevatormanEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ElevatormanSceneCfg = ElevatormanSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # Set simulation parameters

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # set viewer to see the whole scene
        self.viewer.eye = [3.5, 0.0, 3.2]
        self.viewer.lookat = [0.0, 0.0, 0.5]


class RmpFlowAgibotElevatormanEnvCfg(ElevatormanEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Events are not needed for now - events is set to None in base class

        # Set Agibot as robot - matching reference config
        # OLD CODE (commented out):
        # self.scene.robot = AGIBOT_A2D_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot",
        #     init_state=ArticulationCfg.InitialStateCfg(
        #         joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,  # preserve original joint positions
        #         pos=(-2.0, -0.2, 0.0),  # Custom position for elevator setup
        #         rot=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)),  # (w,x,y,z) - 90° rotation around x-axis
        #     ),
        # )
        # NEW CODE (matching reference):
        self.scene.robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Agibot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_gripper_center",
            controller=AGIBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,  # Keep scale at 1.0 for teleop - sensitivity is controlled by device config
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
        self.scene.contact_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_.*_Pad_Link",
            update_period=0.05,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=[],  # Add filter patterns here if needed
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,  # Same as reference - adjust if needed
                    rot_sensitivity=0.05,  # Same as reference - adjust if needed
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,  # Same as reference
                    rot_sensitivity=0.05,  # Same as reference
                    sim_device=self.sim.device,
                ),
            }
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0

