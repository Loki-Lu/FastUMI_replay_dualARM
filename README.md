# FastUMI_replay_dualARM
This is the repo for FastUMI data dual arms replay

# Step 1: Data Processing (Dual) → HDF5

In this step, the **mp4 and csv files in the raw data folder** are synchronized, aligned, and downsampled, then organized into **HDF5 episode** files for later training/evaluation.

- **Script**: `data_process_dual/data2hdf5_multi_process.py`  
- **Function**: Reads videos under `camera/` and trajectories/timestamps under `csv/`, aligns them by time, and writes them into `hdf5/episode_{num}.hdf5`  
- **Parallelism**: Uses a thread pool to concurrently process multiple sequences  
- **Note**: The current script **does not** read/write gripper width (only pose `x,y,z,qx,qy,qz,qw`)

---

## 1) Config file `config/config.json`  

The script only uses `task_config.camera_names` (e.g. `["front"]`) to determine the dataset name for images inside the HDF5 file.

---

## 2) Raw data directory and naming convention

Each task subdirectory (e.g. `task1/1-2_3_0402`) should contain:

```plaintext
<path>/<task_subdir>/
├─ camera/
│  ├─ left_temp_video_{num}.mp4
│  └─ right_temp_video_{num}.mp4
└─ csv/
   ├─ left_temp_trajectory_{num}.csv
   ├─ right_temp_trajectory_{num}.csv
   ├─ left_temp_video_timestamps_{num}.csv
   └─ right_temp_video_timestamps_{num}.csv
```

**CSV columns**:
- Trajectory: `Timestamp, Pos X, Pos Y, Pos Z, Q_X, Q_Y, Q_Z, Q_W`
- Video timestamps: `Frame Index, Timestamp`

---

## 3) Running steps

### 3.1 Modify data path

Open `data_process_dual/data2hdf5_multi_process.py`, go to **line 258**, and change it to your root data directory. For example:

```python
path = "/Users/shuchenye/Desktop/onestar/FastUMI-dual/task10"
```

The script will traverse each task subdirectory under `path` and create an output directory `hdf5/` inside each subdirectory.

---

## 4) Run the script

```bash
python data_process_dual/data2hdf5_multi_process.py
```

---

## Output directory and HDF structure

The script will generate the following in each task subdirectory:

```plaintext
<path>/<task_subdir>/
└─ hdf5/
   └─ episode_{num}.hdf5
```

**Example HDF5 internal structure**:

```plaintext
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

---

# Step 2: Data Processing (FastUMI base) → Local base

## Remember to modify the base of both arms according to your local parameters

```python
# ====== Extrinsics of the two arms ======
base0_xyz = [0.4, 0.0, 0.13]
base0_rpy_deg = [179.94725, -89.999981, 0.0]

base1_xyz = [0.4, 0.0, 0.13]      # TODO: 
base1_rpy_deg = [179.94725, -89.999981, 0.0]
# Left arm marker_id=[0,1]; Right arm marker_id=[6,7]
```

---

## Data conversion results in one HDF5 file (internal structure unchanged)

```bash
python3 data_process_dual/coordinate_transform.py
```

---

# Run replay:
In the HDF5 file, `robot_0` is the left arm, and `robot_1` is the right arm

```bash
# Single-arm test (either left or right)
python3 Realman/replay_rm65.py 
```

---

## FAQ

### Why are some sequences skipped?

Possible reasons: invalid video (`ffprobe` failed), empty trajectory/timestamp CSVs, or **Box filtering** (motion range exceeds threshold).

---

### Multi-camera support?
Change `task_config.camera_names` in `config/config.json` to a list of camera names; the script will write one dataset for each camera.
