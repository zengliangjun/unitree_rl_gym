from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.action_scale = config["action_scale"]

            if "obs_history_length" in config or \
                hasattr(config, "obs_history_length"):
                self.obs_history_length = config["obs_history_length"]
            else:
                self.obs_history_length = 1

            ##
            self.obs_last_action_scale = config["obs_last_action_scale"]
            self.obs_base_ang_vel_scale = config["obs_base_ang_vel_scale"]
            self.obs_cmd_ang_scale = config["obs_cmd_ang_scale"]
            self.obs_cmd_lin_scale = config["obs_cmd_lin_scale"]
            self.obs_cos_phase_scale = config["obs_cos_phase_scale"]
            self.obs_joint_pos_scale = config["obs_joint_pos_scale"]
            self.obs_joint_vel_scale = config["obs_joint_vel_scale"]
            self.obs_gravity_orientation_scale = config["obs_gravity_orientation_scale"]
            self.obs_sin_phase_scale = config["obs_sin_phase_scale"]

            #
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]


            if "ang_vel_yaw_ranges" in config or \
                hasattr(config, "ang_vel_yaw_ranges"):
                self.ang_vel_yaw_ranges = config["ang_vel_yaw_ranges"]
            else:
                self.ang_vel_yaw_ranges = [-0.5, 0.5]

            if "lin_vel_x_ranges" in config or \
                hasattr(config, "lin_vel_x_ranges"):
                self.lin_vel_x_ranges = config["lin_vel_x_ranges"]
            else:
                self.lin_vel_x_ranges = [-0.5, 1.0]

            if "lin_vel_y_ranges" in config or \
                hasattr(config, "lin_vel_y_ranges"):
                self.lin_vel_y_ranges = config["lin_vel_y_ranges"]
            else:
                self.lin_vel_y_ranges = [-0.5, 0.5]

