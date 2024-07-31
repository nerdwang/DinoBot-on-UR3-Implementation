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

e = IPython.embed


class Args:
    def __init__(self):
        self.NAME = "Right"
        self.hostname = "127.0.0.1"
        self.right_robot_ip = "192.168.3.195"
        self.init_ur_pose_deg = [-180, -180, 0, -90, 0, 0]
        self.velocity = 0.05
        self.acceleration = 0.05
        self.dt = 1 / 500  # 2ms
        self.lookahead_time = 0.03
        self.fps = 30
        self.gain = 100
        self.resolution = (640, 480)
        self.get_camera_intrinsics = True
        self.get_cap_positions = False
        self.num_positions = 30
        self.cap_positions = [[-1.427, -3.124, 0.503, -0.464, -1.318, -1.634],
                                [-2.929, -3.827, 0.011, -0.698, -0.886, 1.353],
                                [-3.142, -3.813, -0.005, -0.598, -1.012, 1.353],
                                [-3.339, -3.823, -0.012, -0.788, -1.165, 1.539],
                                [-3.02, -3.545, -0.022, -0.802, -0.847, 1.539],
                                [-2.872, -3.314, 0.014, -0.682, -0.669, 1.539],
                                [-3.021, -3.13, -0.061, -0.685, -0.697, 1.539],
                                [-2.853, -1.269, -0.295, -2.256, 0.748, -1.56],
                                [-2.851, -0.992, -0.295, -2.467, 0.833, -1.56],
                                [-2.698, -0.552, -0.951, -2.352, 0.686, -1.56],]

def main():
    args_instance = Args()
    # 对于不同相机，需要在采集数据之前获取相机内部参数（单个相机的代码），需要获取内参请把参数get_camera_intrinsics改为True
    if args_instance.get_camera_intrinsics == True:
        devices = initialize_camera_device()
        config = rs.config()
        config.enable_stream(rs.stream.depth, args_instance.resolution[0], args_instance.resolution[1], rs.format.z16,
                             args_instance.fps)
        config.enable_stream(rs.stream.color, args_instance.resolution[0], args_instance.resolution[1], rs.format.bgr8,
                             args_instance.fps)
        pipeline_1 = rs.pipeline()
        config.enable_device(devices[0])
        cfg = pipeline_1.start(config)
        profile = cfg.get_stream(rs.stream.depth)
        profile1 = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        intr1 = profile1.as_video_stream_profile().get_intrinsics()
        extrinsics = profile1.get_extrinsics_to(profile)
        print(extrinsics)
        print("深度传感器内参：", intr)
        print("RGB相机内参:", intr1)
        exit()

    #连接右机械臂
    ROBOT_RIP = args_instance.right_robot_ip
    rtde_R = rtde_receive.RTDEReceiveInterface(ROBOT_RIP)
    # 采数据之前请演示一下采集数据的位置（手动移动机械臂）
    if args_instance.get_cap_positions == True:
        cap_positions = []
        print("按下回车键开始/继续记录相机位置！")
        for i in range(args_instance.num_positions):
            # 等待回车键按下
            keyboard.wait('enter')
            qpos = rtde_R.getActualQ()
            qpos = [round(elem, 3) for elem in qpos]
            cap_positions.append(qpos)
            # 执行循环体
            print(f"Loop {i + 1}/{args_instance.num_positions}")
        print("完成相机位置记录！")
        with open('cap_positions.txt', mode='w') as file:
            for sublist in cap_positions:
                # 将子列表转换为字符串并写入文件
                sublist_str = str(sublist)
                file.write(sublist_str + '\n')
        print("机械臂关节角数据已经保存至 cap_positions.txt")
        print(cap_positions)
        rtde_R.disconnect()
        exit()

    #初始化相机们
    devices = initialize_camera_device()

    # 配置第一个摄像机
    config = rs.config()
    config.enable_stream(rs.stream.depth, args_instance.resolution[0], args_instance.resolution[1], rs.format.z16,
                         args_instance.fps)
    config.enable_stream(rs.stream.color, args_instance.resolution[0], args_instance.resolution[1], rs.format.bgr8,
                         args_instance.fps)
    # config.enable_stream(rs.stream.pose)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)

    pipeline_1 = rs.pipeline()
    config.enable_device(devices[0])
    pipeline_1.start(config)
    folder_name = create_folder()

    rtde_C = rtde_control.RTDEControlInterface(ROBOT_RIP)
    init_ur_pose_rad = np.deg2rad(args_instance.init_ur_pose_deg)

    # 初始化机械臂
    rtde_C.moveJ(init_ur_pose_rad)
    cap_positions = args_instance.cap_positions
    print("按下回车开始采集数据!")
    accels = []
    gyros = []
    for i in range(len(cap_positions)):
        keyboard.wait('enter')
        rtde_C.moveJ(cap_positions[i])
        frames1 = pipeline_1.wait_for_frames()
        frames1_align = rs.align(rs.stream.color).process(frames1)
        color_frame1 = frames1_align.get_color_frame()
        depth_frame1 = frames1_align.get_depth_frame()
        accel_frame = frames1_align.first_or_default(rs.stream.accel)
        gyro_frame = frames1_align.first_or_default(rs.stream.gyro)

        if depth_frame1 and color_frame1:            # 将彩色图像数据转换为OpenCV格式
            color_data1 = np.asanyarray(color_frame1.get_data())
            color_filename = os.path.join(f"{folder_name}/images/rgb", f"{i}_color.jpg")
            depth_data1 = np.asanyarray(depth_frame1.get_data())
            depth_filename = os.path.join(f"{folder_name}/images/depth", f"{i}_depth.jpg")
            accel = np.array(accel_frame.as_motion_frame().get_motion_data())
            gyro = np.array(gyro_frame.as_motion_frame().get_motion_data())
            accels.append(accel)
            gyros.append(gyro)

            cv2.imwrite(depth_filename, depth_data1)
            cv2.imwrite(color_filename, color_data1)
        print(f'采集数据进度{i + 1}/{len(cap_positions)}')
    with open(f'{folder_name}/accels.txt', 'w') as f:
        f.write(str(accels))
    with open(f'{folder_name}/gyros.txt', 'w') as f:
        f.write(str(gyros))

    rtde_C.disconnect()
    rtde_R.disconnect()

def create_folder():
    folder_number = 0
    folder_base = "episode_"
    while True:
        if not os.path.exists(f"{folder_base}{folder_number}"):
            break
        folder_number += 1
    folder_name = f"{folder_base}{folder_number}"
    os.makedirs(folder_name)
    images_folder = os.path.join(folder_name, "images")
    os.makedirs(images_folder)
    cam_1_folder = os.path.join(images_folder, "rgb")
    os.makedirs(cam_1_folder)
    cam_2_folder = os.path.join(images_folder, "depth")
    os.makedirs(cam_2_folder)
    return folder_name

def initialize_camera_device():
    devices = []
    for device in rs.context().devices:
        print('Found device: ', device.get_info(rs.camera_info.name), ' ',
              device.get_info(rs.camera_info.serial_number))
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            devices.append(device.get_info(rs.camera_info.serial_number))
    return devices

if __name__ == '__main__':
    main()
