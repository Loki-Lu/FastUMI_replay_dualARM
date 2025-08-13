import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import ikpy.chain
import cv2
import ikpy.utils.plot as plot_utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json

import sys
import os
parent_dir = "/home/jzj/XARM/BestMan_Xarm"
sys.path.append(os.path.join(parent_dir, 'RoboticsToolBox'))
# import pyRobotiqGripper
from Bestman_real_xarm6 import *
import  time

my_chain = ikpy.chain.Chain.from_urdf_file("/home/jzj/fastumi_data_10w_internal/assets/xarm6_robot.urdf", base_elements=['world']) # Path to the URDF file of the robot model (replace with your robot's URDF file in config.json)

def cartesian_to_joints(position, quaternion, initial_joint_angles=None, **kwargs):
    """
    Convert Cartesian coordinates to robot joint angles using inverse kinematics.
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()

    if initial_joint_angles is None:
        initial_joint_angles = [0] * len(my_chain)

    joint_angles = my_chain.inverse_kinematics(
        position,
        rotation_matrix,
        orientation_mode='all',
        initial_position=initial_joint_angles
    )

    return joint_angles


def judge_ik_result(trajectory, init):
    """
    Judge if the IK result is valid.
    """
    base_x, base_y, base_z = 0.510, -0.00588, 0.54984
    base_roll, base_pitch, base_yaw = np.deg2rad([179.94725,
        -89.999981,
        0.0])
    
    # 创建基座坐标系到局部坐标系的旋转矩阵
    # rotation_base_to_local = R.from_euler('zyx', [base_yaw, base_pitch, base_roll]).as_matrix()
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    
    # 构建齐次变换矩阵 T_base_to_local
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    
    qpos_data = trajectory.copy()
    
    normalized_qpos = np.copy(qpos_data)
    for i in range(normalized_qpos.shape[0]):
        x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]

        x -= 0.14565
        z += 0.1586

        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base = transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local)
        ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()

        pos = np.array([x_base, y_base, z_base])
        pos += 0.14565 * ori[:, 2] 
        pos -= 0.1586 * ori[:, 0]
        x_base, y_base, z_base = pos
        # print(x_base, y_base, z_base, roll_base * 180 / np.pi, pitch_base* 180 / np.pi, yaw_base* 180 / np.pi)
        normalized_qpos[i, 0:7] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, ]
        # normalized_qpos[i, 7:] = normalized_qpos[i, 7:] / np.pi * 180  # 转换为角度
    current_angle = init
    for i in range(len(normalized_qpos)):
        # print()
        target = normalized_qpos[i]
        direction, quaternion = calculate_new_pose(
            target[0], target[1], target[2], target[3:7], 0.25)
        # for m in range(3):
        ret = cartesian_to_joints(direction, quaternion, current_angle)
        current_angle = ret.copy()
        # Check if the IK result is valid
        calculated_target_matrix = my_chain.forward_kinematics(ret)
        
        calculated_pos = calculated_target_matrix[:3, 3]
        
        distance = np.linalg.norm(calculated_pos - np.array(direction))
        # print(distance, i)
        if distance > 1e-1:
            print(distance, i)
            return False
    return True

def calculate_new_pose(x, y, z, quaternion, distance):
    """
    Calculate a new pose by translating along the negative Z-axis of the given pose.
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]
    new_position = np.array([x, y, z]) - distance * z_axis
    return [new_position[0], new_position[1], new_position[2]], quaternion

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
       
    # 创建局部坐标系下的旋转矩阵
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # 创建局部坐标系下的齐次变换矩阵 T_local
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    # print(T_local)
    # print(T_base_to_local)
    # 计算基座坐标系下的齐次变换矩阵 T_base
    T_base_r = np.dot(T_local[:3, :3] , T_base_to_local[:3, :3] )
    
    # 提取基座坐标系下的位置
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    
    # 提取基座坐标系下的旋转矩阵并转换为欧拉角
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
    # print(roll_base * 180 / np.pi, pitch_base * 180 / np.pi, yaw_base * 180 / np.pi)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base

def relative_to_init_pos(trajectory):
    """
    Convert trajectory to relative coordinates.
    """
    # Get the initial position and orientation
    init_pos = trajectory[0][0:3]
    init_quat = trajectory[0][3:7]
    
    # Create a transformation matrix for the initial pose
    T_init = np.eye(4)
    T_init[:3, 3] = init_pos
    T_init[:3, :3] = R.from_quat(init_quat).as_matrix()
    T_init_inv = np.linalg.inv(T_init)
    # Transform each pose in the trajectory to relative coordinates
    relative_trajectory = []
    for i in range(len(trajectory)):
        current_mat = np.eye(4)
        current_mat[0:3, 3] = trajectory[i][0:3]
        current_mat[0:3, 0:3] = R.from_quat(trajectory[i][3:7]).as_matrix()
        relative_mat = T_init_inv @ current_mat
        relative_pos = relative_mat[0:3, 3]
        relative_rot = R.from_matrix(relative_mat[0:3, 0:3]).as_quat()
        pose = np.concatenate((relative_pos, relative_rot))
        relative_trajectory.append(np.append(pose, trajectory[i][7]))
    return np.array(relative_trajectory)

def start_replay(trajectory, bestman):
    base_x, base_y, base_z = 0.560, -0.00588, 0.36984
    base_roll, base_pitch, base_yaw = np.deg2rad([179.94725,
        -89.999981,
        0.0])
    
    # 创建基座坐标系到局部坐标系的旋转矩阵
    # rotation_base_to_local = R.from_euler('zyx', [base_yaw, base_pitch, base_roll]).as_matrix()
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    
    # 构建齐次变换矩阵 T_base_to_local
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    
    qpos_data = trajectory.copy()
    
    normalized_qpos = np.copy(qpos_data)
    for i in range(normalized_qpos.shape[0]):
        x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]

        x -= 0.14565
        z += 0.1586

        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base = transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local)
        ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()

        pos = np.array([x_base, y_base, z_base])
        pos += 0.14565 * ori[:, 2] 
        pos -= 0.1586 * ori[:, 0]
        x_base, y_base, z_base = pos
        # print(x_base, y_base, z_base, roll_base * 180 / np.pi, pitch_base* 180 / np.pi, yaw_base* 180 / np.pi)
        normalized_qpos[i, 0:7] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]
        # normalized_qpos[i, 7:] = normalized_qpos[i, 7:] / np.pi * 180  # 转换为角度
    for i in range(len(normalized_qpos)):
        # print()
        target = normalized_qpos[i]
        direction, quaternion = calculate_new_pose(
            target[0], target[1], target[2], target[3:7], 0.08956576500000002)
        
        rpy = R.from_quat(quaternion).as_euler("xyz", degrees=False)

        goal = [direction[0], direction[1], direction[2], rpy[0], rpy[1], rpy[2]]
        print(goal)
        bestman.move_end_effector_to_goal_pose(goal)
        time.sleep(0.07)
        
        

if __name__=="__main__":
    # Load the HDF5 file
    for n in range(200):
        file_path = f"/home/jzj/fastumi_data_10w_internal/single_arm_dataset/task13/2025_4_8/13-2_1_0408/hdf5/episode_{n}.hdf5"
        with h5py.File(file_path, 'r') as f:
            left_trajectory = f['action'][:]
        
        trajectory = relative_to_init_pos(left_trajectory)
        
        # bestman = Bestman_Real_Xarm6("192.168.1.208", None, None)
        # print(bestman.get_current_end_effector_pose())
        # start_replay(trajectory, bestman)
        # exit()

        init = np.array([0, 0, -0.02792803756892681, -0.47473636269569397, 
            -0.0384012907743454, -0.022689493373036385,
            -1.057666301727295, 0.0034878887236118317
        ])

        

        
        
        # print(my_chain.forward_kinematics(init))
        # Check if the IK result is valid
        if judge_ik_result(trajectory, init):
            print(f"{n} IK result is valid.")
        else:
            print(f"{n} IK result is invalid.")