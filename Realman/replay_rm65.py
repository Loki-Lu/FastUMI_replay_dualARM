import time
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from realman.python3.robotic_arm_package.robotic_arm import *
from realman.python3.ik_qp import *

def calculate_new_pose(x, y, z, quaternion, distance):
    """
    基于给定的6D位姿 (x, y, z, 四元数), 计算沿着 z 轴“负方向”平移 distance 后的新位姿。
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]        # 取出姿态矩阵的 z 轴 (第三列)
    new_position = np.array([x, y, z]) - distance * z_axis
    return new_position[0], new_position[1], new_position[2], quaternion

if __name__ == "__main__":

    dT = 0.01 # 单位:sec

    # 实例化逆解库             
    qp = QPIK("RM65B", dT)    
    qp.set_install_angle([90, 180, 0], 'deg')

    # 限制肘部朝外
    # qp.set_elbow_min_angle(3, 'deg')

    # 设置运行过程中的关节速度约束
    qp.set_dq_max_weight([1.0, 1.0, 1.0, 0.1, 1.0, 1.0])

    # 连接机器人, 此案例为real_robot遥操作sim_robot
    # right_robot = Arm(RM75, "192.168.1.19")
    # sim_robot.Movej_Cmd([0, 0, 90, 0, 90, 0,0], 20, 0, 0, True)

    left_robot = Arm(RM65, "192.168.1.18")
    # real_robot.Movej_Cmd([0, 0, 90, 0, 90, 0,0], 20, 0, 0, True)

    # 读取当前机械臂位姿数据
    left_pose = left_robot.Get_Current_Arm_State()
    # print("Current pose", left_pose)

    # 准备数据
    input_file = '/home/onestar/FastUMI_replay_duelARM/test_tcp_1/episode_18.hdf5'
    with h5py.File(input_file, 'r') as f_in:
        # 读取位置与姿态
        xyz_data = f_in['robot_0/action'][:, :3]
        q_data = f_in['robot_0/action'][:, 3:7]
        gripper_data = f_in['robot_0/action'][:, 7]

    # 将四元数转换为欧拉角（XYZ顺序，单位：度）
    rotation = R.from_quat(q_data)
    euler_angles_data = rotation.as_euler('xyz', degrees=False)

    # 主循环，依次执行轨迹中的动作
    for i in range(xyz_data.shape[0]):
        xyz_action = xyz_data[i]           # [x, y, z]
        euler_action = euler_angles_data[i]  # [roll, pitch, yaw] in degrees
        gripper_raw = gripper_data[i]      # 原始 gripper 值 (0~1)

        # 计算 gripper 的整数值 (0~255)，并保持与原逻辑一致
        gripper_cmd = int((1 - gripper_raw) * 255)

        # 计算与 gripper 线性关联的位移距离
        # 当 gripper_cmd=0 -> distance=0.09; 当 gripper_cmd=255 -> distance=0.105
        current_distance = -0.82 + 0.015 * (gripper_cmd / 255.0)

        # 计算新的平移后坐标
        quat_action = q_data[i]
        x_new, y_new, z_new, _ = calculate_new_pose(
            xyz_action[0], xyz_action[1], xyz_action[2],
            quat_action, current_distance
        )
        
        # 设置机器人状态 (注意：此方法根据你的机器人接口可自行调整)
        # goal_pose = [x_new, y_new, z_new, euler_action[0], euler_action[1], euler_action[2]]
        goal_pose = [xyz_action[0], xyz_action[1], xyz_action[2], euler_action[0], euler_action[1], euler_action[2]]
        # print("Final pose", goal_pose)
        left_robot.Movej_P_Cmd(goal_pose, 100, 0, trajectory_connect=0, block=True)

        # 控制抓手
        # bestman.gripper_goto(gripper_cmd)
        # bestman.gripper_goto_robotiq(gripper_cmd)

        # 可视需要在此插入合适的延时
        # time.sleep(0.01)



