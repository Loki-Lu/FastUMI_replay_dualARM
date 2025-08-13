import h5py
import pandas as pd
import os
import json
import re
import cv2
from tqdm import tqdm
import numpy as np
import subprocess
import concurrent.futures  # 导入线程池模块
import multiprocessing
from scipy.spatial.transform import Rotation as R


def extract_sequence_numbers(filenames, template):
  """
  从文件名列表中提取序号，根据给定的模版。

  Args:
    filenames: 要提取序号的文件名列表 (list of strings).
    template: 文件名模版，其中序号位置用 {} 表示 (string).

  Returns:
    提取到的序号的集合 (set of ints)。如果匹配失败，则不包含在集合中.
  """

  sequence_numbers = set()
  regex_pattern = template.replace("{}", r"(\d+)")

  for filename in filenames:
    match = re.search(regex_pattern, filename)

    if match:
      try:
        sequence_number = int(match.group(1))
        sequence_numbers.add(sequence_number)
      except ValueError:
        # 如果提取的不是数字，跳过
        pass
  return sequence_numbers


def get_video_state(video_path):
    """使用ffprobe获取视频时长（秒）"""
    if not os.path.isfile(video_path):
        print(f'[ERROR] Video file not found: {video_path}')
        
        return False
    try:
        # 构建ffprobe命令
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        # 执行命令并获取输出
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f'[ERROR] ffprobe failed: {result.stderr.strip()}')
            return False
            
        return float(result.stdout.strip())>0
    except Exception as e:
        print(f'[ERROR] Failed to get video duration: {str(e), video_path}')
        return False


with open('config/config.json', 'r') as f:
    config = json.load(f)

# Set environment variables
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Set device
ROBOT_TYPE = config['device_settings']["robot_type"]
TASK_CONFIG = config['task_config']

cfg = TASK_CONFIG
robot = ROBOT_TYPE

task = []

LEFT_VIDEO_PATH_TEMP = "left_temp_video_{}.mp4"
RIGHT_VIDEO_PATH_TEMP = "right_temp_video_{}.mp4"
LEFT_TRAJECTORY_PATH_TEMP = "left_temp_trajectory_{}.csv"
RIGHT_TRAJECTORY_PATH_TEMP = "right_temp_trajectory_{}.csv"
LEFT_VIDEO_TIMESTAMP_PATH_TEMP = "left_temp_video_timestamps_{}.csv"
RIGHT_VIDEO_TIMESTAMP_PATH_TEMP = "right_temp_video_timestamps_{}.csv"




def get_box(trajectory):
    x = trajectory['Pos X']
    y = trajectory['Pos Y']
    z = trajectory['Pos Z']
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    box = np.zeros(3)
    box[0] = np.max(x) - np.min(x)
    box[1] = np.max(y) - np.min(y)
    box[2] = np.max(z) - np.min(z)
    # print(f"{num}Box: {box}")
    return box

def judge_box(left_box, right_box, task_id):
    left_task_expected_box = {
        "1": [0.5, 0.5, 0.5],
        "2": [0.5, 0.5, 0.5],
        # "4": []
    }
    right_task_expected_box = {
        "1": [0.5, 0.5, 0.5],
        "2": [0.5, 0.5, 0.5]
    }
    flag = True
    left_target = left_task_expected_box[task_id]
    right_target = right_task_expected_box[task_id] 
    for i in range(len(left_target)):
        if left_box[i] > left_target[i] or right_box[i] > right_target[i]:
            flag = False
            break
    
    return flag
    

def process_sequence(num, CAMERA, CSV, HDF5):  # 将每个序列号的处理过程封装成函数
    data_dict = {}
    for i in range(2):
        data_dict[f'/robot_{i}/observations/qpos'] = []
        data_dict[f'/robot_{i}/action'] = []
        for cam_name in cfg['camera_names']:
            data_dict[f'/robot_{i}/observations/images/{cam_name}'] = []
    left_video_path = os.path.join(CAMERA, LEFT_VIDEO_PATH_TEMP.format(num))
    right_video_path = os.path.join(CAMERA, RIGHT_VIDEO_PATH_TEMP.format(num))
    left_trajectory_path = os.path.join(CSV, LEFT_TRAJECTORY_PATH_TEMP.format(num))
    right_trajectory_path = os.path.join(CSV, RIGHT_TRAJECTORY_PATH_TEMP.format(num))
    left_video_timestamp_path = os.path.join(CSV, LEFT_VIDEO_TIMESTAMP_PATH_TEMP.format(num))
    right_video_timestamp_path = os.path.join(CSV, RIGHT_VIDEO_TIMESTAMP_PATH_TEMP.format(num))

    # Load data
    left_video_cap = cv2.VideoCapture(left_video_path)
    left_trajectory = pd.read_csv(left_trajectory_path)
    left_video_timestamps = pd.read_csv(left_video_timestamp_path)
    left_downsampled_timestamps = left_video_timestamps.iloc[::3].reset_index(drop=True)
    right_video_cap = cv2.VideoCapture(right_video_path)
    right_trajectory = pd.read_csv(right_trajectory_path)
    right_video_timestamps = pd.read_csv(right_video_timestamp_path)
    right_downsampled_timestamps = right_video_timestamps.iloc[::3].reset_index(drop=True)
    
    if left_trajectory.empty or right_trajectory.empty or left_video_timestamps.empty or right_video_timestamps.empty:
        print(f"{left_trajectory_path} or {right_trajectory_path} trajectory empty!!!!!!!")
        return
    if not get_video_state(left_video_path) or not get_video_state(right_video_path):
        print(f"{left_video_path} or {right_video_path} video empty!!!!!!!")
        return
    
    left_box = get_box(left_trajectory)
    print(left_box, num, judge_box(get_box(left_trajectory), get_box(right_trajectory), task_id="2"))
    # get_box(right_trajectory, num)
    # print(judge_box(get_box(left_trajectory, num), get_box(right_trajectory, num), task_id="1"), num)
    # return 0    
    if not judge_box(get_box(left_trajectory), get_box(right_trajectory), task_id="2"):
        print(f"Box size is too large, skipping sequence {num}")
        return    
    
    
    for _, row in tqdm(left_downsampled_timestamps.iterrows(), desc=f'Extracting Images (Left - {num})'):
        frame_idx = row['Frame Index']
        left_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = left_video_cap.read()
        if ret:
            for cam_name in cfg['camera_names']:
                data_dict[f'/robot_0/observations/images/{cam_name}'].append(frame)
    left_video_cap.release()
    
    for idx, row in tqdm(left_downsampled_timestamps.iterrows(), desc=f'Extracting States (Left - {num})'):
        closest_idx = (np.abs(left_trajectory['Timestamp'] - row['Timestamp'])).argmin()
        closest_row = left_trajectory.iloc[closest_idx]
        pos_quat = [
            closest_row['Pos X'], closest_row['Pos Y'], closest_row['Pos Z'],
            closest_row['Q_X'], closest_row['Q_Y'], closest_row['Q_Z'], closest_row['Q_W']
        ]
        data_dict['/robot_0/observations/qpos'].append(pos_quat)
        data_dict['/robot_0/action'].append(pos_quat)
        

    for _, row in tqdm(right_downsampled_timestamps.iterrows(), desc=f'Extracting Images (Right - {num})'):
        frame_idx = row['Frame Index']
        right_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = right_video_cap.read()
        if ret:
            for cam_name in cfg['camera_names']:
                data_dict[f'/robot_1/observations/images/{cam_name}'].append(frame)
    right_video_cap.release()
    
    for idx, row in tqdm(right_downsampled_timestamps.iterrows(), desc=f'Extracting States (Right - {num})'):
        closest_idx = (np.abs(right_trajectory['Timestamp'] - row['Timestamp'])).argmin()
        closest_row = right_trajectory.iloc[closest_idx]
        pos_quat = [
            closest_row['Pos X'], closest_row['Pos Y'], closest_row['Pos Z'],
            closest_row['Q_X'], closest_row['Q_Y'], closest_row['Q_Z'], closest_row['Q_W']
        ]
        data_dict['/robot_1/observations/qpos'].append(pos_quat)
        data_dict['/robot_1/action'].append(pos_quat)
    hdf5_path = os.path.join(HDF5, f'episode_{num}.hdf5')
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs['sim'] = False
        for i in range(2):
            robot_grp = root.create_group(f'robot_{i}')
            for cam_name in cfg['camera_names']:
                obs_grp = robot_grp.create_group('observations')
                image_grp = obs_grp.create_group('images')
                for cam_name in cfg['camera_names']:
                    image_grp.create_dataset(
                        cam_name,
                        data=np.array(data_dict[f'/robot_{i}/observations/images/{cam_name}'], dtype=np.uint8),
                        compression='gzip',
                        compression_opts=4
                    )
                root.create_dataset(f'/robot_{i}/observations/qpos', data=np.array(data_dict[f'/robot_{i}/observations/qpos']))
                root.create_dataset(f'/robot_{i}/action', data=np.array(data_dict[f'/robot_{i}/action']))


def data2hdf5(data_path, num_workers=4):  # 添加 num_workers 参数
    CAMERA = os.path.join(data_path, 'camera')
    CSV = os.path.join(data_path, 'csv')
    HDF5 = os.path.join(data_path, 'hdf5')
    if not os.path.exists(HDF5):
        os.makedirs(HDF5)
    
    # Count the number of files in the directory
    mp4_file_names = os.listdir(CAMERA)
    csv_file_names = os.listdir(CSV)
    mp4_sequence_numbers = extract_sequence_numbers(mp4_file_names, LEFT_VIDEO_PATH_TEMP)
    trajectory_csv_sequence_numbers = extract_sequence_numbers(csv_file_names, LEFT_TRAJECTORY_PATH_TEMP)
    video_timestamp_csv_sequence_numbers = extract_sequence_numbers(csv_file_names, LEFT_VIDEO_TIMESTAMP_PATH_TEMP)
    
    sequence_numbers = list(mp4_sequence_numbers & trajectory_csv_sequence_numbers & video_timestamp_csv_sequence_numbers)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:  # 创建线程池
        futures = [executor.submit(process_sequence, num, CAMERA, CSV, HDF5) for num in sequence_numbers]  # 提交任务
        for future in concurrent.futures.as_completed(futures):  # 监控任务完成情况
            try:
                future.result()  # 获取任务结果，如果任务抛出异常，这里会捕获
            except Exception as e:
                print(f"任务执行出错: {e}")

    print("Data conversion to HDF5 completed.")

if __name__ == "__main__":
    path = "/Users/shuchenye/Desktop/onestar/FastUMI-dual/task10"  
    list_task = []
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                list_task.append(item)
    print(list_task)

    for task in list_task:
        data_path = os.path.join(path, task)
        print(data_path)
        data2hdf5(data_path, 32)