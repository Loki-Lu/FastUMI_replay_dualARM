import argparse
import os
import sys
import time
import numpy as np
from datetime import date, datetime
from scipy.spatial.transform import Rotation as R
# import torch
import cv2
import numpy as np
import tf.transformations as tft
import pickle
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import json
from collections import deque
import threading
from pynput import keyboard  # Import pynput for keyboard listening
import copy
# 将 robotics tool 相关包添加到路径
# sys.path.append('/home/lkh/Project/fastumi_data_10w_internal/BestMan_Xarm/RoboticsToolBox/')
import time
import sys, os
import argparse
import h5py
sys.path.insert(1, "..")
sys.path.insert(2, "/home/jzj/data_quality/BestMan_Chemistry_Test")
sys.path.insert(3, "/home/jzj/data_quality/BestMan_Chemistry_Test/Install/realman_rdk")
from Robotics_API import Bestman_Real_Realman
from src.Robotic_Arm.rm_robot_interface import *


def judge_ik_result( left_trajectory, right_trajectory, init_left, init_right):
    left_arm = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)
    right_arm = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)
    current_left_angle = init_left
    current_right_angle = init_right
    for i in range(len(left_trajectory)):
        left_target = rm_inverse_kinematics_params_t(q_in = current_left_angle, q_pose = left_trajectory[i], flag = 0)
        right_target = rm_inverse_kinematics_params_t(q_in = current_right_angle, q_pose = right_trajectory[i], flag = 0)
        left_ret, left_angle = left_arm.rm_algo_inverse_kinematics(params=left_target)
        right_ret, right_angle = right_arm.rm_algo_inverse_kinematics(params=right_target)
        left_pose = left_arm.rm_algo_forward_kinematics(left_angle, flag=0)
        right_pose = right_arm.rm_algo_forward_kinematics(right_angle, flag=0)
        left_distance = np.linalg.norm(np.array(left_pose)[0:3] - np.array(left_trajectory[i])[0:3])
        right_distance = np.linalg.norm(np.array(right_pose)[0:3] - np.array(right_trajectory[i])[0:3])
        if left_ret != 0 or right_ret != 0:
            return False
        if left_distance > 1e-4 or right_distance > 1e-4:
            return False
    return True

        
# def world_to_base(left_trajectory, right_trajectory):
#     base_left_trajectory = []
#     base_right_trajectory = []
#     ####
#     # TODO: 确定matrix
#     matrix_left = np.eye(4)
#     matrix_right = np.eye(4)
#     ####
#     for i in range(len(left_trajectory)):
#         left_mat = np.eye(4)
#         left_mat[0:3, 3] = left_trajectory[i][0:3]
#         left_mat[0:3, 0:3] = R.from_quat(left_trajectory[i][3:]).as_matrix()
#         base_left_mat = matrix_left @ left_mat
#         base_left_trajectory.append(np.concatenate((base_left_mat[0:3, 3], R.from_matrix(base_left_mat[0:3, 0:3]).as_quat())))
        
#         right_mat = np.eye(4)   
#         right_mat[0:3, 3] = right_trajectory[i][0:3]
#         right_mat[0:3, 0:3] = R.from_quat(right_trajectory[i][3:]).as_matrix()
#         base_right_mat = matrix_right @ right_mat
#         base_right_trajectory.append(np.concatenate((base_right_mat[0:3, 3], R.from_matrix(base_right_mat[0:3, 0:3]).as_quat())))
#     return base_left_trajectory, base_right_trajectory


def load_hdf5_file(file_path):
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
        left_rot = R.from_quat(left_trajectory[i][3:]).as_matrix()
        left_pos = left_trajectory[i][0:3]
        left_pos += 0.14565 * left_rot[:, 2] 
        left_pos -= 0.1586 * left_rot[:, 0]
        left_trajectory[i][0:3] = left_pos
        
        right_rot = R.from_quat(right_trajectory[i][3:]).as_matrix()
        right_pos = right_trajectory[i][0:3]
        right_pos += 0.14565 * right_rot[:, 2]
        right_pos -= 0.1586 * right_rot[:, 0]
        right_trajectory[i][0:3] = right_pos
        
    init_left = left_trajectory[0]
    init_right = right_trajectory[0]
    
    init_left_mat = np.eye(4)   
    init_left_mat[0:3, 3] = init_left[0:3]
    init_left_mat[0:3, 0:3] = R.from_quat(init_left[3:]).as_matrix()
    init_right_mat = np.eye(4)
    init_right_mat[0:3, 3] = init_right[0:3]
    init_right_mat[0:3, 0:3] = R.from_quat(init_right[3:]).as_matrix()
    init_left_mat_inv = np.linalg.inv(init_left_mat)
    init_right_mat_inv = np.linalg.inv(init_right_mat)
    
    for i in range(len(left_trajectory)):
        left_mat = np.eye(4)
        left_mat[0:3, 3] = left_trajectory[i][0:3]
        left_mat[0:3, 0:3] = R.from_quat(left_trajectory[i][3:]).as_matrix()
        left_mat = init_left_mat_inv @ left_mat
        left_trajectory[i][0:3] = left_mat[0:3, 3]
        left_trajectory[i][3:] = R.from_matrix(left_mat[0:3, 0:3]).as_quat()
        
        right_mat = np.eye(4)
        right_mat[0:3, 3] = right_trajectory[i][0:3]
        right_mat[0:3, 0:3] = R.from_quat(right_trajectory[i][3:]).as_matrix()
        right_mat = init_right_mat_inv @ right_mat
        right_trajectory[i][0:3] = right_mat[0:3, 3]
        right_trajectory[i][3:] = R.from_matrix(right_mat[0:3, 0:3]).as_quat()
    
    return init_left, init_right, left_trajectory, right_trajectory






def main():

    left_arm = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)
    right_arm = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E, rm_force_type_e.RM_MODEL_RM_B_E)
    target = rm_inverse_kinematics_params_t(q_in=[90]*7, q_pose=[0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0], flag=0)
    ret, left_angle = left_arm.rm_algo_inverse_kinematics(params=target)
    # print(left_angle)
    # left_pose = left_arm.rm_algo_forward_kinematics(left_angle, flag=0)
    # print(left_pose)
    # print(np.linalg.norm(np.array(left_pose)[0:3] - np.array([0.3, 0.2, 0.4])))
    
    
    left_angle = [13.4399995803833, -77.4990005493164, -30.591999053955078, -119.73300170898438, 85.49500274658203, 59.194000244140625, 62.051998138427734]
    right_angle = [-0.1340000033378601, 90.42500305175781, 7.315000057220459, 112.66300201416016, -90.35800170898438, -83.15499877929688, -65.28299713134766]
    
    
    left_pose = left_arm.rm_algo_forward_kinematics(left_angle, flag=0)
    right_pose = right_arm.rm_algo_forward_kinematics(right_angle, flag=0)
    print(left_pose)
    print(right_pose)
    
    
if __name__ == "__main__":
    main()