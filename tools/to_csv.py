import os
import h5py
import pandas as pd

# —— 直接在这里修改 —— 
input_dir = '/home/onestar/FastUMI_replay_duelARM/test/'
output_dir = '/home/onestar/FastUMI_replay_duelARM/test/'

os.makedirs(output_dir, exist_ok=True)

def convert_hdf5_to_csv(input_path: str, output_path: str):
    """
    Reads qpos and action datasets from an HDF5 file and writes them to a single CSV file.
    """
    with h5py.File(input_path, 'r') as f:
        qpos = f['observations/qpos'][:]  # (T, 8)
        action = f['action'][:]           # (T, 7)

    if qpos.shape[0] != action.shape[0]:
        print(f"警告: {os.path.basename(input_path)} qpos 行数 {qpos.shape[0]} 与 action 行数 {action.shape[0]} 不一致。")

    qpos_cols   = ['x','y','z','qx_base', 'qy_base', 'qz_base', 'qw_base', 'gripper']
    action_cols = ['x','y','z','qx_base', 'qy_base', 'qz_base', 'qw_base', 'gripper']

    df = pd.concat([
        pd.DataFrame(qpos,   columns=qpos_cols),
        pd.DataFrame(action, columns=action_cols)
    ], axis=1)

    df.to_csv(output_path, index=False)
    print(f"已生成 CSV: {output_path}")

# 批量处理
for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith('.hdf5'):
        continue
    in_path  = os.path.join(input_dir,  fname)
    out_name = os.path.splitext(fname)[0] + '.csv'
    out_path = os.path.join(output_dir, out_name)
    convert_hdf5_to_csv(in_path, out_path)