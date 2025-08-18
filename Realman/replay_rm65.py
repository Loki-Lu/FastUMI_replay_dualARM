from Robotic_Arm.rm_robot_interface import *
import h5py
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        """
        Initialize and connect to the robotic arm.

        Args:
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("\nFailed to connect to the robot arm\n")
            exit(1)
        else:
            print(f"\nSuccessfully connected to the robot arm: {self.handle.id}\n")

    def disconnect(self):
        """
        Disconnect from the robot arm.

        Returns:
            None
        """
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("\nSuccessfully disconnected from the robot arm\n")
        else:
            print("\nFailed to disconnect from the robot arm\n")

    def get_arm_model(self):
        """Get robotic arm mode.
        """
        res, model = self.robot.rm_get_robot_info()
        if res == 0:
            return model["arm_model"]
        else:
            print("\nFailed to get robot arm model\n")
    def movel(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movel motion.

        Args:
            pose (list of float): End position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("\nmovel motion succeeded\n")
        else:
            print("\nmovel motion failed, Error code: ", movel_result, "\n")

    def movej_p(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movej_p motion.

        Args:
            pose (list of float): Position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("\nmovej_p motion succeeded\n")
        else:
            print("\nmovej_p motion failed, Error code: ", movej_p_result, "\n")

    def set_gripper_pick_on(self, speed, force, block=True, timeout=30):
        """
        Perform continuous force-controlled gripping with the gripper.

        Args:
            speed (int): Speed of the gripper.
            force (int): Force applied by the gripper.
            block (bool, optional): Whether the function is blocking. Defaults to True.
            timeout (int, optional): Timeout duration. Defaults to 30.

        Returns:
            None
        """
        gripper_result = self.robot.rm_set_gripper_pick_on(speed, force, block, timeout)
        if gripper_result == 0:
            print("\nGripper continuous force control gripping succeeded\n")
        else:
            print("\nGripper continuous force control gripping failed, Error code: ", gripper_result, "\n")
        time.sleep(2)

    def set_gripper_release(self, speed, block=True, timeout=30):
        """
        Release the gripper.

        Args:
            speed (int): Speed of the gripper release.
            block (bool, optional): Whether the function is blocking. Defaults to True.
            timeout (int, optional): Timeout duration. Defaults to 30.

        Returns:
            None
        """
        gripper_result = self.robot.rm_set_gripper_release(speed, block, timeout)
        if gripper_result == 0:
            print("\nGripper release succeeded\n")
        else:
            print("\nGripper release failed, Error code: ", gripper_result, "\n")
        time.sleep(2)

    def set_lift_height(self, speed, height, block=True):
        """
        Set the lift height of the robot.

        Args:
            speed (int): Speed of the lift.
            height (int): Target height of the lift.
            block (bool, optional): Whether the function is blocking. Defaults to True.

        Returns:
            None
        """
        lift_result = self.robot.rm_set_lift_height(speed, height, block)
        if lift_result == 0:
            print("\nLift motion succeeded\n")
        else:
            print("\nLift motion failed, Error code: ", lift_result, "\n")
def calculate_new_pose(x, y, z, quaternion, distance):
    """
    基于给定的6D位姿 (x, y, z, 四元数), 计算沿着 z 轴“负方向”平移 distance 后的新位姿。
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]        # 取出姿态矩阵的 z 轴 (第三列)
    new_position = np.array([x, y, z]) - distance * z_axis
    return new_position[0], new_position[1], new_position[2], quaternion

if __name__ == "__main__":

    # Create a robot arm controller instance and connect to the robot arm
    robot = RobotArmController("192.168.1.18", 8080, 3)
        # Get API version
    print("\nAPI Version: ", rm_api_version(), "\n")

    ret = robot.robot.rm_change_work_frame("Base")
    print("\nChange work frame: ", ret, "\n")

        # Get basic arm information
    # robot.get_arm_software_info()

    arm_model = robot.get_arm_model()
    # 准备数据
    input_file = '/home/onestar/FastUMI_replay_duelARM/test_tcp_1/episode_18.hdf5'
    with h5py.File(input_file, 'r') as f_in:
        # 读取位置与姿态
        xyz_data = f_in['robot_0/action'][:, :3]
        q_data = f_in['robot_0/action'][:, 3:7]
        gripper_data = f_in['robot_0/action'][:, 7]

    # 将四元数转换为欧拉角（XYZ顺序，单位：度）
    rotation = R.from_quat(q_data)
    euler_angles_data = rotation.as_euler('xyz', degrees=False)

    # 主循环，依次执行轨迹中的动作
    for i in range(xyz_data.shape[0]):
        xyz_action = xyz_data[i]           # [x, y, z]
        euler_action = euler_angles_data[i]  # [roll, pitch, yaw] in degrees
        gripper_raw = gripper_data[i]      # 原始 gripper 值 (0~1)

        # 计算 gripper 的整数值 (0~255)，并保持与原逻辑一致
        gripper_cmd = int((1 - gripper_raw) * 255)

        # 计算与 gripper 线性关联的位移距离
        # 当 gripper_cmd=0 -> distance=0.09; 当 gripper_cmd=255 -> distance=0.105
        current_distance = -0.82 + 0.015 * (gripper_cmd / 255.0)

        # 计算新的平移后坐标
        quat_action = q_data[i]
        x_new, y_new, z_new, _ = calculate_new_pose(
            xyz_action[0], xyz_action[1], xyz_action[2],
            quat_action, current_distance
        )
        
        # 设置机器人状态 (注意：此方法根据你的机器人接口可自行调整)
        # goal_pose = [x_new, y_new, z_new, euler_action[0], euler_action[1], euler_action[2]]
        goal_pose = [xyz_action[0], xyz_action[1], xyz_action[2], euler_action[0], euler_action[1], euler_action[2]]
        # print("Final pose", goal_pose)
        # 如果不是最后一个点 → 暂存 (trajectory_connect=1, block=False)
        # 最后一个点 → 立即执行 (trajectory_connect=0, block=True)
        if i < xyz_data.shape[0] - 1:
            robot.movej_p(goal_pose, v = 100, r = 100, connect=1, block=0)
            # print("1")
        else:
            robot.movej_p(goal_pose, v = 100, r = 100, connect=0, block=1)
            # print("0")

        # 控制抓手
        # bestman.gripper_goto(gripper_cmd)
        # bestman.gripper_goto_robotiq(gripper_cmd)

        # 可视需要在此插入合适的延时
        # time.sleep(0.5)