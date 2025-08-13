# coding:UTF-8
import numpy as np
import time
from ik_rbtdef import *
from ik_rbtutils import *
from robotic_arm_package.robotic_arm import *
from ik_qp import *

if __name__ == '__main__':

    dT = 0.02 # 单位:sec

    # 实例化逆解库             
    qp = QPIK("RM65B", dT)    
    qp.set_install_angle([90, 180, 0], 'deg')

    # 限制肘部朝外
    # qp.set_elbow_min_angle(3, 'deg')

    # 设置运行过程中的关节速度约束
    qp.set_dq_max_weight([1.0, 1.0, 1.0, 0.1, 1.0, 1.0])

    # 连接机器人, 此案例为real_robot遥操作sim_robot
    # right_robot = Arm(RM75, "192.168.1.19")
    # sim_robot.Movej_Cmd([0, 0, 90, 0, 90, 0,0], 20, 0, 0, True)

    left_robot = Arm(RM65, "192.168.1.18")
    # real_robot.Movej_Cmd([0, 0, 90, 0, 90, 0,0], 20, 0, 0, True)

    # 读取当前机械臂位姿数据
    left_pose = left_robot.Get_Current_Arm_State()
    # q = np.array(ret[1])*deg2rad
    pose_1 = left_pose[2]

    # right_pose = right_robot.Get_Current_Arm_State()
    # pose_2 = right_pose[2]

    # pose0 = [0.31490200757980347, 0.589264988899231, 0.23061299324035645, -1.184000015258789, 1.187000036239624, 0.3149999976158142]
    # sim_robot.Movej_P_Cmd( pose, 10, 0, trajectory_connect=0, block=True)
    left_pose = [-0.08784399926662445, 0.07203999906778336, 0.825626015663147, 0.06800000369548798, 0.4690000116825104, 1.687000036239624]
    # pose_init_left = [-0.4000050127506256, 0.39998599886894226, 0.4000129997730255, -1.5700000524520874, 1.5700000524520874, 0.0]
    # pose_init_right = [0.40001600980758667, 0.3999899923801422, 0.3999899923801422, -1.5700000524520874, 1.5700000524520874, 0.0]
    # pose = [-0.10331699997186661, 0.5100749731063843, 0.029618000611662865, -1.5700000524520874, 1.5700000524520874, 0.0]
    # right_robot.Movej_P_Cmd(pose_init_right, 10, 0, trajectory_connect=0, block=True)
    left_robot.Movej_P_Cmd(left_pose, 10, 0, trajectory_connect=0, block=True)
    time.sleep(5)


