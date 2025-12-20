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

import deploy_mujoco
import phas_gait

if __name__ == "__main__":
    #config_file = "holosoma_g1_23.yaml"
    config_file = "holosoma_g1_23_unitree_init.yaml"
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

        obs_last_action_scale = config["obs_last_action_scale"]
        obs_base_ang_vel_scale = config["obs_base_ang_vel_scale"]
        obs_cmd_ang_scale = config["obs_cmd_ang_scale"]
        obs_cmd_lin_scale = config["obs_cmd_lin_scale"]
        obs_cos_phase_scale = config["obs_cos_phase_scale"]
        obs_joint_pos_scale = config["obs_joint_pos_scale"]
        obs_joint_vel_scale = config["obs_joint_vel_scale"]
        obs_gravity_orientation_scale = config["obs_gravity_orientation_scale"]
        obs_sin_phase_scale = config["obs_sin_phase_scale"]

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    step = 0

    dt = simulation_dt * control_decimation
    gait_state = phas_gait.LocomotionGait(dt)
    gait_state.setup()
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    for joint_id in range(m.njnt):
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(f"{joint_id}: {joint_name}")


    # load policy
    policy = torch.jit.load(policy_path)
    gait_state.reset(None)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = deploy_mujoco.pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                step += 1
                #
                episode_length_buf = torch.tensor([step], dtype=torch.float32)
                gait_state.step(cmd, episode_length_buf)

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles)
                dqj = dqj
                gravity_orientation = deploy_mujoco.get_gravity_orientation(quat)
                omega = omega

                #
                sin_phase = torch.sin(gait_state.phase).numpy()[0]
                cos_phase = torch.cos(gait_state.phase).numpy()[0]

                # last_action
                obs[:num_actions] = action * obs_last_action_scale
                # base_ang_vel
                obs[num_actions: num_actions + 3] = omega * obs_base_ang_vel_scale
                # cmd_ang
                obs[num_actions + 3: num_actions + 4] = cmd[2:] * obs_cmd_ang_scale
                # cmd_lin
                obs[num_actions + 4: num_actions + 6] = cmd[:2] * obs_cmd_lin_scale
                # cos_phase
                obs[num_actions + 6: num_actions + 8] = cos_phase * obs_cos_phase_scale
                # joint_pos
                obs[num_actions + 8: num_actions * 2 + 8] = qj * obs_joint_pos_scale
                # joint_vel
                obs[num_actions * 2 + 8: num_actions * 3 + 8] = dqj * obs_joint_vel_scale
                # gravity_orientation
                obs[num_actions * 3 + 8: num_actions * 3 + 11] = gravity_orientation * obs_gravity_orientation_scale
                # sin_phase
                obs[num_actions * 3 + 11: num_actions * 3 + 13] = sin_phase * obs_sin_phase_scale

                ####
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
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
