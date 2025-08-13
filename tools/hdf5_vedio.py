import os
import h5py
import cv2

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['robot_1/observations/qpos'][()]
        image_dict = dict()
        for cam_name in root[f'robot_1/observations/images/'].keys():
            image_dict[cam_name] = root[f'robot_1/observations/images/{cam_name}'][()]
        action = root['robot_1/action'][()]

    return qpos, action, image_dict

def play_video_cv2(image_list, fps=60):
    for image in image_list:
        # bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        # print(bgr_image.shape)
        cv2.imshow('Video', image)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def write_video_cv2(image_list, output_file, fps=60):
    height, width, _ = image_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for mp4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in image_list:
        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    out.release()

# Define task and load the dataset
task = 'open_drawer_new'  # open_fridge, open_drawer, pick_place_pot
index = 1
data_file = f'/home/onestar/FastUMI_replay_duelARM/test/episode_18.hdf5'
qpos, action, image_dict = load_hdf5(dataset_path=data_file)

# Play and write videos using OpenCV
for cam_name, image_list in image_dict.items():
    print(f"Playing video from camera: {cam_name}")
    play_video_cv2(image_list, fps=1)
    write_video_cv2(image_list, f'test_video_{cam_name}.mp4', fps=0.1)