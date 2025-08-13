# -*- coding: utf-8 -*-
"""
step1
将相机系下面的数据转换为基座系下的数据，并且将夹爪的宽度归一化
—— 在同一个输出 hdf5 中写出 robot_0 与 robot_1 两个分组
结构：
/robot_0/action
/robot_0/observations/qpos
/robot_0/observations/images/front
/robot_1/action
/robot_1/observations/qpos
/robot_1/observations/images/front
"""

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool

# 加载预定义的字典（沿用你的写法/参数）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

def get_gripper_width(img_list, marker_ids=(0, 1)):
    """
    根据指定的 marker_ids 计算夹爪宽度（与原逻辑一致，仅修正变量名、支持多ID）
    marker_ids: tuple/list，如 (0,1) 或 (6,7)
    """
    distances = []
    distances_index = []
    current_frame = 0
    frame_count = len(img_list)

    for i in range(img_list.shape[0]):
        gray = cv2.cvtColor(img_list[i, :, :, :], cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            current_frame += 1

            # 存储标记的中心点
            marker_centers = []
            for j, marker_id in enumerate(ids.flatten()):  
                if marker_id in marker_ids:
                    marker_corners = corners[j][0]
                    center = np.mean(marker_corners, axis=0).astype(int)
                    marker_centers.append(center)

            # 如果检测到至少两个标记，计算它们之间的距离
            if len(marker_centers) >= 2:
                # 只取前两个标记计算距离
                distance = np.linalg.norm(marker_centers[0] - marker_centers[1])
                distances.append(distance)
                distances_index.append(current_frame)

            elif len(marker_centers) == 1:
                distance = abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2
                distances.append(distance)
                distances_index.append(current_frame)

    distances = np.array(distances)
    distances_index = np.array(distances_index)

    if len(distances) == 0:
        return np.zeros(frame_count, dtype=np.int16)

    distances = ((distances - 140.0) / (566.0 - 140.0) * 850).astype(np.int16).clip(0, 850)

    new_distances = []
    for i in range(len(distances) - 1):
        # 处理第一帧
        if i == 0:
            if distances_index[i] == 1:
                new_distances.append(distances[0])
            else:
                for _ in range(distances_index[0]):
                    new_distances.append(distances[0])
        else:
            if distances_index[i+1] - distances_index[i] == 1:
                new_distances.append(distances[i])
            else:
                for k in range(distances_index[i+1] - distances_index[i]):
                    new_distances.append(int(k * (distances[i+1] - distances[i]) /
                                              (distances_index[i+1] - distances_index[i]) + distances[i]))
    new_distances.append(distances[-1])

    # 若不足 frame_count，尾部补齐；若超过则截断
    if len(new_distances) < frame_count:
        new_distances.extend([distances[-1]] * (frame_count - len(new_distances)))

    return np.array(new_distances[:frame_count], dtype=np.int16)

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    # 创建局部坐标系下的旋转矩阵
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # 创建局部坐标系下的齐次变换矩阵 T_local
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]

    # 与你原代码一致的相乘顺序
    T_base_r = np.dot(T_local[:3, :3], T_base_to_local[:3, :3])

    # 提取基座坐标系下的位置
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]

    # 提取基座坐标系下的旋转矩阵并转换为四元数
    rotation_base = R.from_matrix(T_base_r)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()

    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base

def build_T_base_to_local(base_xyz, base_rpy_deg):
    base_roll, base_pitch, base_yaw = np.deg2rad(base_rpy_deg)
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_base_to_local
    T[:3, 3] = base_xyz
    return T

def process_one_robot(f_in, robot_key, T_base_to_local, marker_ids):
    """
    基于你的原始处理流程，处理单个 robot_*：
    - 读 robot_*/action, robot_*/observations/qpos, robot_*/observations/images/front
    - 做位姿变换与 TCP 偏移
    - 计算夹爪宽度并归一化
    - 返回 (normalized_action_with_gripper, normalized_qpos_with_gripper, images_front 或 None)
    若该 robot_* 不存在则返回 None
    """
    if robot_key not in f_in:
        return None

    # 读取数据（与原路径一致，只是把 robot_0 改成通用 robot_key）
    grp = f_in[robot_key]
    action_data = grp['action'][:] if 'action' in grp else grp['observations/qpos'][:]
    qpos_data = grp['observations/qpos'][:]

    # 图像（有就读，没有就置 None）
    image_data = None
    if 'observations' in grp and 'images' in grp['observations'] and 'front' in grp['observations/images']:
        image_data = grp['observations/images/front'][:]

    # ===== 与原脚本一致的位姿处理 =====
    normalized_qpos = np.copy(qpos_data)
    init_pos = qpos_data[0, 0:3]

    for i in range(normalized_qpos.shape[0]):
        normalized_qpos[i, 0:3] -= init_pos

        x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]

        x -= 0.14565
        z += 0.1586

        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base = transform_to_base_quat(
            x, y, z, qx, qy, qz, qw, T_base_to_local
        )
        ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()

        pos = np.array([x_base, y_base, z_base])
        pos += 0.14565 * ori[:, 2]
        pos -= 0.1586 * ori[:, 0]
        x_base, y_base, z_base = pos

        normalized_qpos[i, :7] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]
        # normalized_qpos[i, 7:] = normalized_qpos[i, 7:] / np.pi * 180  # 如果你需要角度再放开

    # ===== 夹爪宽度（支持不同 ID）=====
    if image_data is not None and len(image_data) > 0:
        gripper_open_width = get_gripper_width(np.array(image_data), marker_ids=marker_ids)
    else:
        gripper_open_width = np.zeros(normalized_qpos.shape[0], dtype=np.int16)

    # 对齐长度（以 qpos 帧数为准）
    T_total = normalized_qpos.shape[0]
    if len(gripper_open_width) < T_total:
        gripper_open_width = np.pad(gripper_open_width, (0, T_total - len(gripper_open_width)), mode='edge')
    elif len(gripper_open_width) > T_total:
        gripper_open_width = gripper_open_width[:T_total]

    gripper_open_width = gripper_open_width / 850.0
    gripper_width = gripper_open_width.reshape(-1, 1)

    normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
    normalized_action_with_gripper = np.copy(normalized_qpos_with_gripper)

    return normalized_action_with_gripper, normalized_qpos_with_gripper, image_data

def normalize_and_save_hdf5(args):
    input_file, output_file = args

    # ====== 两个机械臂各自的外参（按你的原值给 robot_0；robot_1 如有不同请修改）======
    base0_xyz = [0.4, 0.0, 0.13]
    base0_rpy_deg = [179.94725, -89.999981, 0.0]

    base1_xyz = [0.4, 0.0, 0.13]      # TODO: 如果 robot_1 不同，请改这里
    base1_rpy_deg = [179.94725, -89.999981, 0.0]

    T0 = build_T_base_to_local(base0_xyz, base0_rpy_deg)
    T1 = build_T_base_to_local(base1_xyz, base1_rpy_deg)

    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        # ===== 处理 robot_0 =====
        out0 = process_one_robot(f_in, 'robot_0', T0, marker_ids=(0, 1))
        if out0 is not None:
            normalized_action_with_gripper, normalized_qpos_with_gripper, image_data = out0

            g0 = f_out.create_group('robot_0')
            g0.create_dataset('action', data=normalized_action_with_gripper)

            observations_group = g0.create_group('observations')
            images_group = observations_group.create_group('images')

            if image_data is not None:
                max_timesteps = image_data.shape[0]
                cam_hight, cam_width = image_data.shape[1], image_data.shape[2]
                images_group.create_dataset(
                    'front',
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )
                images_group['front'][:] = image_data

            observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)

        # ===== 处理 robot_1 =====
        out1 = process_one_robot(f_in, 'robot_1', T1, marker_ids=(6, 7))
        if out1 is not None:
            normalized_action_with_gripper, normalized_qpos_with_gripper, image_data = out1

            g1 = f_out.create_group('robot_1')
            g1.create_dataset('action', data=normalized_action_with_gripper)

            observations_group = g1.create_group('observations')
            images_group = observations_group.create_group('images')

            if image_data is not None:
                max_timesteps = image_data.shape[0]
                cam_hight, cam_width = image_data.shape[1], image_data.shape[2]
                images_group.create_dataset(
                    'front',
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )
                images_group['front'][:] = image_data

            observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)

        print(f"Normalized data saved to: {output_file}")

if __name__ == "__main__":
    input_dir = '/home/onestar/FastUMI_replay_duelARM/test'
    output_dir = '/home/onestar/FastUMI_replay_duelARM/test_tcp_1'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    file_list = [
        filename for filename in os.listdir(input_dir)
        if filename.endswith('.hdf5')
    ]

    args_list = []
    for filename in file_list:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        args_list.append((input_file, output_file))

    print("开始并行处理...")

    # 使用所有可用的CPU核心数（你原来固定 4，这里保持一致也行）
    num_processes = min(4, os.cpu_count() or 1)

    with Pool(num_processes) as pool:
        list(
            tqdm(pool.imap_unordered(normalize_and_save_hdf5, args_list),
                 total=len(args_list),
                 desc="Processing files"))

    print("所有文件处理完成。")
