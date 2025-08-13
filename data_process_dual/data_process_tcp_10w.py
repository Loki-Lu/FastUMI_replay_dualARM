'''
step1
将相机系下面的数据转换为基座系下的数据，并且将夹爪的宽度归一化
'''


import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# 加载预定义的字典
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

def get_gripper_width(img_list):

    distances = []
    distances_index = []
    current_frame = 0
    frame_count = len(img_list)

    for i in range(img_list.shape[0]):
        gray = cv2.cvtColor(img_list[i, :, :, :], cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # print(f"Detected markers in frame {current_frame}: {ids.flatten()}")
            current_frame += 1

            # 存储标记的中心点
            marker_centers = []
            for i, marker_id in enumerate(ids.flatten()):
                # 获取标记的四个角点
                if marker_id == 0 or marker_id == 1:
                    marker_corners = corners[i][0]
                    # 计算标记的中心点
                    center = np.mean(marker_corners, axis=0).astype(int)
                    marker_centers.append(center)
                # 在图像上绘制标记的ID

            # 如果检测到至少两个标记，计算它们之间的距离
            if len(marker_centers) >= 2:
                # 只取前两个标记计算距离130...
                distance = np.linalg.norm(marker_centers[0] - marker_centers[1])

                distances.append(distance)
                distances_index.append(current_frame)

            elif len(marker_centers) == 1:
                distance = abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2

                distances.append(distance)
                distances_index.append(current_frame)


    distances = np.array(distances)
    distances_index = np.array(distances_index)
    distances = ((distances - 140.0) / (566.0 - 140.0) * 850).astype(np.int16).clip(0, 850)
    new_distances = []
    for i in range(len(distances) - 1):
        #处理第一帧
        if i == 0:
            if distances_index[i] == 1:
                new_distances.append(distances[0])
                continue
            else:
                for _ in range(distances_index[0]):
                    new_distances.append(distances[0])
        else:
            if distances_index[i+1] - distances_index[i]==1:
                new_distances.append(distances[i])
            else:
                for k in range(distances_index[i+1] - distances_index[i]):
                    new_distances.append(int( k * (distances[i+1] - distances[i]) / (distances_index[i+1] - distances_index[i]) + distances[i] ))
    new_distances.append(distances[-1])
    if len(new_distances) < frame_count:
        for _ in range(frame_count - len(new_distances)):
            new_distances.append(distances[-1])
    
    return np.array(new_distances)

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


def normalize_and_save_hdf5(args):
    input_file, output_file = args
    # 基座坐标系下局部坐标系的原点位姿
    base_x, base_y, base_z = 0.3212697143554687, -0.141122863769, 0.650
    base_roll, base_pitch, base_yaw = np.deg2rad([-90,
        0,
        -90])
    # 创建基座坐标系到局部坐标系的旋转矩阵
    # rotation_base_to_local = R.from_euler('zyx', [base_yaw, base_pitch, base_roll]).as_matrix()
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    
    # 构建齐次变换矩阵 T_base_to_local
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    
    # 打开输入HDF5文件
    with h5py.File(input_file, 'r') as f_in:
        # 读取需要处理的数据
        # 读取需要处理的数据
        action_data = f_in['robot_1/action'][:]
        qpos_data = f_in['robot_1/observations/qpos'][:]
        image_data = f_in['robot_1/observations/images/front'][:]   

        # image = f_in['observations/images/front'][i, :]
        # print(image.shape)
        # exit()

        # 获取初始点
        initial_action = action_data[0, :]
        initial_qpos = qpos_data[0, :]
        normalized_qpos = np.copy(qpos_data)
        init_pos = initial_qpos[0:3]

        for i in range(normalized_qpos.shape[0]):
            normalized_qpos[i, 0:3] -= init_pos
        
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
            normalized_qpos[i, :] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]
            # normalized_qpos[i, 7:] = normalized_qpos[i, 7:] / np.pi * 180  # 转换为角度


        image_data = np.array(image_data)
        gripper_open_width = get_gripper_width(image_data)
        gripper_open_width = gripper_open_width / 850

        gripper_width = gripper_open_width.reshape(-1, 1)
        normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
        # print(new_joint_angles.shape)
        # exit()

        normalized_action_with_gripper = np.copy(normalized_qpos_with_gripper)
        # 创建输出HDF5文件
        # exit()
        with h5py.File(output_file, 'w') as f_out:
            # 复制原文件的结构并写入归一化数据
            f_out.create_dataset('action', data=normalized_action_with_gripper)
            observations_group = f_out.create_group('observations')
            images_group = observations_group.create_group('images')
            

            max_timesteps = f_in['robot_1/observations/images/front'].shape[0]
            cam_hight, cam_width = f_in['robot_1/observations/images/front'].shape[1], f_in['robot_1/observations/images/front'].shape[2]
            # 复制 images/front 数据集
            images_group.create_dataset(
                'front',
                (max_timesteps, cam_hight, cam_width, 3),
                dtype='uint8',
                chunks=(1, cam_hight, cam_width, 3),
                compression='gzip',
                compression_opts=4
            )
            images_group['front'][:] = f_in['robot_1/observations/images/front'][:]




            # # 复制 images/front 数据集
            # images_group.create_dataset('front', data=f_in['observations/images/front'][:])
            
            # 写入归一化后的 qpos 数据
            observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)
            
            # 复制原始 qvel 数据（假设不需要处理）
            # observations_group.create_dataset('qvel', data=f_in['observations/qvel'][:])
            
            print(f"Normalized data saved to: {output_file}")

if __name__ == "__main__":
    input_dir = '/home/zak/CRA/BestMan_CRA/data/tcp_undefined/'
    output_dir = '/home/zak/CRA/BestMan_CRA/data/tcp_undefined1/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # 使用所有可用的CPU核心数
    num_processes = 15

    with Pool(num_processes) as pool:
        list(
            tqdm(pool.imap_unordered(normalize_and_save_hdf5, args_list),
                    total=len(args_list),
                    desc="Processing files"))

    print("所有文件处理完成。")


