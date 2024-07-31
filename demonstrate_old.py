import os
import pyrealsense2 as rs
import cv2
import json
import rtde_receive
import rtde_control
import numpy as np
import time
import keyboard
import math as m
import IPython
from tool import robotiq_gripper

e = IPython.embed


class Args:
    def __init__(self):
        self.NAME = "Left"
        self.hostname = "127.0.0.1"
        self.right_robot_ip = "192.168.3.196"
        self.init_ur_pose_deg = [-180, -180, 0, -90, 0, 0]
        self.velocity = 0.05
        self.acceleration = 0.05
        self.dt = 1 / 500  # 2ms
        self.lookahead_time = 0.03
        self.fps = 30
        self.gain = 100
        self.resolution = (640, 480)
        self.get_camera_intrinsics = False
        self.get_cap_positions = False
        self.num_positions = 30
        self.cap_positions = [[-2.978, -2.766, -1.782, 0.017, -0.084, -0.135],
                                [-2.991, -2.765, -1.691, -0.009, -0.032, -0.135],
                                [-3.027, -2.812, -1.516, -0.235, -0.144, -0.13],
                                [-3.079, -2.913, -1.171, -0.392, -0.028, -0.133],
                                [-3.081, -3.018, -0.807, -0.858, -0.071, -0.133],
                                [-3.17, -3.054, -0.553, -1.059, 0.045, -0.133],
                                [-3.277, -3.054, -0.553, -1.059, 0.045, -0.133],
                                [-3.277, -3.054, -0.553, -1.059, 0.045, -0.133],
                                [-2.978, -2.766, -1.782, 0.017, -0.084, -0.135],]
        self.gripper_states = [0, 0, 0, 0, 0, 0, 0, 100, 0]


def main():
    args_instance = Args()

     #连接右机械臂
    ROBOT_RIP = args_instance.right_robot_ip
    rtde_R = rtde_receive.RTDEReceiveInterface(ROBOT_RIP)

    
    #gripper = robotiq_gripper.RobotiqGripper()
    #gripper.connect(ROBOT_RIP, 63352)
    print("Gripper Connected!!!")
    print("Gripper Activated!!!")
    rtde_C = rtde_control.RTDEControlInterface(ROBOT_RIP)
    cap_positions = args_instance.cap_positions
    gripper_states = args_instance.gripper_states
    
    six_dof = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    rtde_C.moveL(six_dof)


if __name__ == '__main__':
    main()
