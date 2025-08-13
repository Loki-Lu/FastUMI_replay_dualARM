import time
import sys
import os
import argparse
import csv
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(1, "..")
sys.path.insert(2, "../Robotics_API")
from robotic_arm_package.robotic_arm import *

# from Robotics_API import Bestman_Real_Realman
# from Bestman_Real_Realman_duel import Bestman_Real_Realman_duel

# 如果 Pose.py 和本脚本不在同级，需要根据实际路径进行修改
# from Pose import Pose 
from Robotics_API.Pose import Pose  

def read_hdf5_file(file_path):
    """
    读取 HDF5 文件，返回动作数据和图像数据。
    
    Args:
        file_path (str): HDF5 文件路径。
        
    Returns:
        tuple: 动作数据和图像数据。
    """
    with h5py.File(file_path, 'r') as f:
        left_trajectory = f['/robot_0/observations/qpos'][:]
        right_trajectory = f['/robot_1/observations/qpos'][:]
    
    
    # init_left = left_trajectory[0]
    # init_right = right_trajectory[0]

    for i in range(len(left_trajectory)):
        lx,ly,lz,lqx,lqy,lqz,lqw = left_trajectory[i]
        left_mat = R.from_quat([lqx, lqy, lqz, lqw]).as_matrix()
        pos = np.array([lx-0.14565, ly, lz+0.1586])
        pos += 0.14565 * left_mat[:, 0]
        pos -= 0.1586 * left_mat[:, 2]
        left_trajectory[i][:3] = pos

        rx,ry,rz,rqx,rqy,rqz,rqw = right_trajectory[i]
        right_mat = R.from_quat([rqx, rqy, rqz, rqw]).as_matrix()
        pos = np.array([rx-0.14565, ry, rz+0.1586])
        pos += 0.14565 * right_mat[:, 0]
        pos -= 0.1586 * right_mat[:, 2]
        right_trajectory[i][:3] = pos

    init_left = left_trajectory[0]
    init_right = right_trajectory[0]

    return init_left, init_right, left_trajectory, right_trajectory


def calculate_relative_trajectory_pose(init_left, init_right, left_trajectory, right_trajectory):
    """
    计算相对位姿。
    
    Args:
        init_left (np.ndarray): 初始左臂位姿。
        init_right (np.ndarray): 初始右臂位姿。
        left_trajectory (np.ndarray): 左臂轨迹数据。
        right_trajectory (np.ndarray): 右臂轨迹数据。
        
    Returns:
        tuple: 相对位姿数据。
    """
    init_left_mat = np.eye(4)
    init_left_mat[:3, :3] = R.from_quat(init_left[3:]).as_matrix()
    init_left_mat[:3, 3] = init_left[:3]

    init_right_mat = np.eye(4)
    init_right_mat[:3, :3] = R.from_quat(init_right[3:]).as_matrix()
    init_right_mat[:3, 3] = init_right[:3]
    
    inv_left_mat = np.linalg.inv(init_left_mat)
    inv_right_mat = np.linalg.inv(init_right_mat)
    
    relative_left_trajectory = []
    relative_right_trajectory = []
    
    for i in range(len(left_trajectory)):    
        left_mat = np.eye(4)
        left_mat[:3, :3] = R.from_quat(left_trajectory[i][3:]).as_matrix()
        left_mat[:3, 3] = left_trajectory[i][:3]
        
        right_mat = np.eye(4)
        right_mat[:3, :3] = R.from_quat(right_trajectory[i][3:]).as_matrix()
        right_mat[:3, 3] = right_trajectory[i][:3]

        relative_left_mat = inv_left_mat @ left_mat
        relative_right_mat = inv_right_mat @ right_mat
        relative_left_quat = R.from_matrix(relative_left_mat[:3, :3]).as_quat()
        relative_right_quat = R.from_matrix(relative_right_mat[:3, :3]).as_quat()   
        relative_left_trajectory.append(np.concatenate([relative_left_mat[:3, 3], relative_left_quat]))
        relative_right_trajectory.append(np.concatenate([relative_right_mat[:3, 3], relative_right_quat]))

    return relative_left_trajectory, relative_right_trajectory


def calculate_relative_pose_alt(init_pose1, init_pose2, pose1, pose2):
    rot_matrix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    # init_rot1 = R.from_quat(init_pose1[3:]).as_matrix()
    init_rot1 = R.from_euler('xyz', init_pose1[3:]).as_matrix() 
    init_rot2 = R.from_euler('xyz', init_pose2[3:]).as_matrix() 
    # init_rot2 = R.from_quat(init_pose2[3:]).as_matrix()
    rot1 = R.from_quat(pose1[3:]).as_matrix()
    rot2 = R.from_quat(pose2[3:]).as_matrix()
    
    relative_quat1 = R.from_matrix(init_rot1 @ rot_matrix @ rot1 @ np.linalg.inv(rot_matrix)).as_quat()
    relative_quat2 = R.from_matrix(init_rot2 @ rot_matrix @ rot2 @ np.linalg.inv(rot_matrix)).as_quat()
    
    relative_trans1 = np.zeros(3)
    relative_trans2 = np.zeros(3)
    relative_trans1[0] = init_pose1[0] - pose1[1] 
    relative_trans1[1] = init_pose1[1] + pose1[0] 
    relative_trans1[2] = init_pose1[2] + pose1[2] 
    
    relative_trans2[0] = init_pose2[0] - pose2[1]
    relative_trans2[1] = init_pose2[1] + pose2[0] 
    relative_trans2[2] = init_pose2[2] + pose2[2]
    
    relative_pose1 = np.concatenate([relative_trans1, relative_quat1])
    relative_pose2 = np.concatenate([relative_trans2, relative_quat2])
    return relative_pose1, relative_pose2

def main(csv_filename):
    """
    读取 hdf5 轨迹数据，并逐步发送给机械臂。
    
    Args:
        csv_filename (str): 轨迹数据文件名。
    """
    # 读取 CSV 轨迹数据
    init_left, init_right, left_trajectory, right_trajectory = read_hdf5_file(csv_filename)
    trajectory = []
    relative_left_trajectory, relative_right_trajectory = calculate_relative_trajectory_pose(init_left, init_right, left_trajectory, right_trajectory)
    for i in range(len(relative_left_trajectory)):
        trajectory.append((i, relative_left_trajectory[i], relative_right_trajectory[i]))

    # 初始化双臂机器人
    right_robot = Arm(RM75, "192.168.1.19")
    left_robot = Arm(RM75, "192.168.1.18")

    # duel_robot = Bestman_Real_Realman_duel()
    time.sleep(0.3)  # 等待回调函数接收数据
    init_pose1 = left_robot.Get_Current_Arm_State()
    init_pose2 = right_robot.Get_Current_Arm_State()
    # init_pose1, init_pose2 = duel_robot.get_current_eef_pose_dpu()
    init_pose1 = np.array(init_pose1[2])
    init_pose2 = np.array(init_pose2[2])
    time.sleep(0.3)  # 等待回调函数接收数据
    
    for i, (timestamp, left_pose, right_pose) in enumerate(trajectory):

        # 发送轨迹点到机械臂
        # time.sleep(1/10)
        relative_pose1, relative_pose2 = calculate_relative_pose_alt(init_pose1, init_pose2, left_pose, right_pose)
        # duel_robot.move_arm_to_joint_values_high_freq(joint1_left, joint2_right, follow=False)
        print(f"Sending step {i + 1}/{len(trajectory)}: Left = {relative_pose1}, Right = {relative_pose2}")

        output_relative_pose1 = relative_pose1[:6]
        output_relative_pose2 = relative_pose2[:6]

        qx, qy, qz, qw = relative_pose1[3:]
        # 四元数转欧拉角（roll, pitch, yaw）
        rotation = R.from_quat([qx, qy, qz, qw])
        output_relative_pose1[3: ]= rotation.as_euler('xyz', degrees=False)  # 弧度制

        qx, qy, qz, qw = relative_pose2[3:]
        # 四元数转欧拉角（roll, pitch, yaw）
        rotation = R.from_quat([qx, qy, qz, qw])
        output_relative_pose2[3: ]= rotation.as_euler('xyz', degrees=False)  # 弧度制

        # relative_pose1 = Pose(relative_pose1)
        # relative_pose2 = Pose(relative_pose2)
        
        left_robot.Movej_P_Cmd(output_relative_pose1, 20, 0, trajectory_connect=0, block=False)
        right_robot.Movej_P_Cmd(output_relative_pose2, 20, 0, trajectory_connect=0, block=False)

        # duel_robot.move_eef_to_goal_pose_high_freq(relative_pose1, relative_pose2, follow=False)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Send trajectory from CSV to dual-arm robot.")
    # parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file containing trajectory data.")
    # args = parser.parse_args()
    hdf5_file = "/home/pjlab/1-9-7/h5py/episode_10.hdf5"
    # hdf5_file = "/home/pjlab/11-10-5-hdf5/episode_50.hdf5"
    main(hdf5_file)

