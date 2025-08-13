# FastUMI_replay_duelARM
This is the repo for FastUMI data dual arms replay

# Step 1: Data Processing (Dual) → HDF5

本步骤将 **raw 数据文件夹中的 mp4 与 csv** 同步、对齐、降采样后，整理为 **HDF5 episode** 文件，供后续训练/评估使用。

- 脚本：`data_process_dual/data2hdf5_multi_process.py`  
- 功能：读取 `camera/` 下的视频与 `csv/` 下的轨迹/时间戳，按时间对齐后写入 `hdf5/episode_{num}.hdf5`  
- 并行：使用线程池并发处理多个序列  
- 备注：当前脚本 **不会** 读取/写入 gripper width（只有位姿 `x,y,z,qx,qy,qz,qw`）

---

## 1) 配置文件 `config/config.json`  

脚本只会用到 `task_config.camera_names`（例如 `["front"]`），用于确定写入 HDF5 内图像数据集的名称。

---

## 2) 原始数据目录与命名规范

每个任务子目录（例如 `task1/1-2_3_0402`）应包含：

```python
<path>/<task_subdir>/
├─ camera/
│ ├─ left_temp_video_{num}.mp4
│ └─ right_temp_video_{num}.mp4
└─ csv/
├─ left_temp_trajectory_{num}.csv
├─ right_temp_trajectory_{num}.csv
├─ left_temp_video_timestamps_{num}.csv
└─ right_temp_video_timestamps_{num}.csv
```

CSV 列：
- 轨迹：`Timestamp, Pos X, Pos Y, Pos Z, Q_X, Q_Y, Q_Z, Q_W`
- 视频时间戳：`Frame Index, Timestamp`

---

## 3) 运行步骤

### 3.1 修改数据路径

打开 `data_process_dual/data2hdf5_multi_process.py`，找到 **第 258 行**，将其改为你的数据根目录。例如：

```python
path = "/Users/shuchenye/Desktop/onestar/FastUMI-dual/task10"
```

脚本会遍历 `path` 下的每个任务子目录，并在每个子目录内部创建输出目录 `hdf5/`.

## 4) 执行脚本

```python
python data_process_dual/data2hdf5_multi_process.py
```

## 输出目录与HDF结构

脚本会在每个任务子目录下生成：

```python
<path>/<task_subdir>/
└─ hdf5/
   └─ episode_{num}.hdf5
```

HDF5 内部结构示例：

```python
/
├─ robot_0/
│  └─ observations/
│     └─ images/
│        └─ front
├─ robot_1/
│  └─ observations/
│     └─ images/
│        └─ front
├─ /robot_0/observations/qpos
├─ /robot_0/action
├─ /robot_1/observations/qpos
└─ /robot_1/action
```

## 常见问题（FAQ）

### 序列被跳过的原因？

可能是：视频无效（`ffprobe` 失败）、轨迹/时间戳 CSV 为空，或 **Box 过滤**（运动范围超过阈值）。

### 多相机支持？
把 `config/config.json` 中的 `task_config.camera_names` 改为相机名列表；脚本会为每个相机名写一个数据集。
