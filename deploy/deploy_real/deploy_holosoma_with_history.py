import sys
import os.path as osp
root_dir = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config_holosoma import Config

import phas_gait
import copy
import pickle

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.gait_state = phas_gait.LocomotionGait(self.config.control_dt)
        self.gait_state.setup()

        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.history_obs = {
            "last_action": [],
            "last_action": [],
            "base_ang_vel": [],
            "cmd_ang": [],
            "cmd_lin": [],
            "cos_phase": [],
            "joint_pos": [],
            "joint_vel": [],
            "gravity_orientation": [],
            "sin_phase": []
        }
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.dump = {
            "obs": [],
            "actions": [],
        }

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        self.gait_state.reset(None)
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _process_obs(self, name: str):
        obs = self.history_obs[name]

        count = len(obs)
        dim = obs[0].shape[0]

        if count < self.config.obs_history_length:
            out = np.zeros((self.config.obs_history_length * dim), dtype = np.float32)

            sid = - count * dim
            tmp = np.concatenate(obs, axis = 0)
            out[sid: ] = tmp
        else:
            out = np.concatenate(obs, axis = 0)

        if count >= self.config.obs_history_length:
            self.history_obs[name] = obs[1: ]

        return out

    def _pre_process_history_obs(self):
        obs = []
        for name in ["last_action", "base_ang_vel", "cmd_ang",
                        "cmd_lin", "cos_phase", "joint_pos", "joint_vel",
                        "gravity_orientation", "sin_phase"]:

            subobs = self._process_obs(name)
            obs.append(subobs)

        return np.concatenate(obs, axis = 0)

    def run(self):
        self.counter += 1


        ##
        episode_length_buf = torch.tensor([self.counter], dtype=torch.float32)
        self.gait_state.step(self.cmd, episode_length_buf)

        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles)# * self.config.dof_pos_scale
        dqj_obs = dqj_obs # * self.config.dof_vel_scale
        if len(ang_vel.shape) == 2:
            ang_vel = ang_vel[0]
        else:
            ang_vel = ang_vel # * self.config.ang_vel_scale
        '''
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        '''

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        self.cmd[0] = min(self.cmd[0], self.config.lin_vel_x_ranges[1])
        self.cmd[0] = max(self.cmd[0], self.config.lin_vel_x_ranges[0])

        self.cmd[1] = min(self.cmd[1], self.config.lin_vel_y_ranges[1])
        self.cmd[1] = max(self.cmd[1], self.config.lin_vel_y_ranges[0])

        self.cmd[2] = min(self.cmd[2], self.config.ang_vel_yaw_ranges[1])
        self.cmd[2] = max(self.cmd[2], self.config.ang_vel_yaw_ranges[0])

        num_actions = self.config.num_actions

        sin_phase = torch.sin(self.gait_state.phase).numpy()[0]
        cos_phase = torch.cos(self.gait_state.phase).numpy()[0]
        # last_action
        self.history_obs["last_action"].append(self.action * self.config.obs_last_action_scale)
        # base_ang_vel
        self.history_obs["base_ang_vel"].append(ang_vel * self.config.obs_base_ang_vel_scale)
        # cmd_ang
        self.history_obs["cmd_ang"].append(self.cmd[2:] * self.config.obs_cmd_ang_scale)
        # cmd_lin
        self.history_obs["cmd_lin"].append(self.cmd[:2] * self.config.obs_cmd_lin_scale)
        # cos_phase
        self.history_obs["cos_phase"].append(cos_phase * self.config.obs_cos_phase_scale)
        # joint_pos
        self.history_obs["joint_pos"].append(qj_obs * self.config.obs_joint_pos_scale)
        # joint_vel
        self.history_obs["joint_vel"].append(dqj_obs * self.config.obs_joint_vel_scale)
        # gravity_orientation
        self.history_obs["gravity_orientation"].append(gravity_orientation * self.config.obs_gravity_orientation_scale)
        # sin_phase
        self.history_obs["sin_phase"].append(sin_phase * self.config.obs_sin_phase_scale)

        # Get the action from the policy network
        obs = self._pre_process_history_obs()
        obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        self.dump["obs"].append(copy.deepcopy(obs))
        self.dump["actions"].append(copy.deepcopy(self.action))

        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)



if __name__ == "__main__":
    #import argparse

    #parser = argparse.ArgumentParser()
    #parser.add_argument("net", type=str, help="network interface")
    #parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    #args = parser.parse_args()
    config_net = "enp3s0"
    config_file = "holosoma_g123dof_loc.yaml"
    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{config_file}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, config_net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

    with open("dum_data.pkl", 'wb') as fd:
        pickle.dump(controller.dump, fd)
