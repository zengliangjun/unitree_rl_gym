import time
import sys
import os.path as osp
root_dir = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import copy

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config_file", type=str, help="config file name in the config folder")
    #args = parser.parse_args()
    config_file = "rl_lab_g1_23.yaml" #args.config_file

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        action_scale = config["action_scale"]

        obs_base_ang_vel_scale = config["obs_base_ang_vel_scale"]
        obs_gravity_orientation_scale = config["obs_gravity_orientation_scale"]
        obs_cmd_scale = config["obs_cmd_scale"]
        obs_joint_pos_scale = config["obs_joint_pos_scale"]
        obs_joint_vel_scale = config["obs_joint_vel_scale"]
        obs_last_action_scale = config["obs_last_action_scale"]

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        history_length = config["history_length"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    history_obs = np.zeros(num_obs * history_length, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    lab_qpos = np.zeros(num_actions, dtype=np.float32)
    lab_qvel = np.zeros(num_actions, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    mujoco2labids = []
    mujoco_joint_names = []
    for joint_id in range(m.njnt):
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name not in config["lab_joint_names"]:
            continue

        print(f"mujoco  {joint_id}: {joint_name}")
        mujoco_joint_names.append(joint_name)

        labid = config["lab_joint_names"].index(joint_name)
        mujoco2labids.append(labid)

    print("mujoco2labids:", mujoco2labids)

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            qpos = copy.deepcopy(d.qpos[7:])
            qvel = copy.deepcopy(d.qvel[6:])

            lab_qpos[mujoco2labids] = qpos
            lab_qvel[mujoco2labids] = qvel

            lab_tau = pd_control(target_dof_pos, lab_qpos, kps, np.zeros_like(kds), lab_qvel, kds)

            d.ctrl[:] = lab_tau[mujoco2labids]
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                lab_qpos[mujoco2labids] = qj
                lab_qvel[mujoco2labids] = dqj
                gravity_orientation = get_gravity_orientation(quat)

                ## base_ang_vel
                obs[:3] = omega * obs_base_ang_vel_scale
                ## gravity_orientation_scale
                obs[3:6] = gravity_orientation * obs_gravity_orientation_scale
                ## cmd
                obs[6:9] = cmd * obs_cmd_scale
                ## joint_pos
                obs[9:9 + num_actions] = lab_qpos * obs_joint_pos_scale
                ## joint_vel
                obs[9 + num_actions:9 + num_actions * 2] = lab_qvel * obs_joint_vel_scale
                ## joint_vel
                obs[9 + num_actions * 2:9 + num_actions * 3] = action * obs_last_action_scale

                history_obs[: num_obs * (history_length -1)] = copy.deepcopy(history_obs[num_obs: ])
                history_obs[num_obs * (history_length -1): ] = obs

                obs_tensor = torch.from_numpy(history_obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
