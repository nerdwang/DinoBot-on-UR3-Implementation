import os
import pyrealsense2 as rs
import cv2
import json
import rtde_receive
import rtde_control
import numpy as np
import keyboard
import math as m
import IPython
from tool import robotiq_gripper
import torch
import matplotlib.pyplot as plt 
from torchvision import transforms,utils
from PIL import Image
import torchvision.transforms as T
import warnings 
import glob
import time
from scipy.spatial.transform import Rotation as Rt
warnings.filterwarnings("ignore")

#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from correspondences import find_correspondences, draw_correspondences
e = IPython.embed

#Hyperparameters for DINO correspondences extraction
num_pairs = 8 
load_size = 224
layer = 9 
facet = 'key' 
bin=True 
thresh=0.05 
model_type='dino_vits8' 
stride=4 

#Deployment hyperparameters    
ERR_THRESHOLD = 0.1 #A generic error between the two sets of points


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


def demonstrate():
    args_instance = Args()

     #连接右机械臂
    ROBOT_RIP = args_instance.right_robot_ip
    rtde_R = rtde_receive.RTDEReceiveInterface(ROBOT_RIP)

    
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_RIP, 63352)
    print("Gripper Connected!!!")
    print("Gripper Activated!!!")
    rtde_C = rtde_control.RTDEControlInterface(ROBOT_RIP)
    cap_positions = args_instance.cap_positions
    gripper_states = args_instance.gripper_states
    
    for i in range(len(cap_positions)):
        keyboard.wait('enter')
        rtde_C.moveJ(cap_positions[i])
        gripper.move(gripper_states[i], 255, 10)
        print(rtde_R.getActualQ())
        # 获取当前末端执行器的位姿
        tcp_pose = rtde_R.getActualTCPPose()
        print("Current End Effector Pose: ", tcp_pose)

        # 末端执行器的位姿包含 [X, Y, Z, Rx, Ry, Rz]
        x, y, z, rx, ry, rz = tcp_pose
        print(f"Position: X={x}, Y={y}, Z={z}")
        print(f"Orientation: Rx={rx}, Ry={ry}, Rz={rz}")
def initialize_camera_device():
    devices = []
    for device in rs.context().devices:
        print('Found device: ', device.get_info(rs.camera_info.name), ' ',
              device.get_info(rs.camera_info.serial_number))
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            devices.append(device.get_info(rs.camera_info.serial_number))
    return devices

    
def add_depth(indexs, depth, LorR):
    """
    Inputs: indexs: list of [x,y] pixel coordinates, 
            depth (H,W,1) observations from camera.
    Outputs: point_with_depth: list of [x,y,z] coordinates.
    
    Adds the depth value/channel to the list of pixels by
    extracting the corresponding depth value from depth.Hence, you 
    can get 3d coordinates in the camera frame.
    """
    
    points = []

    for index in indexs:
        v, u = index
        if LorR == "Left":
            cx = 321.366
            cy = 251.662
            fx = 609.287
            fy = 609.187
        if LorR == "Right":
            cx = 316.09
            cy = 244.494
            fx = 606.304
            fy = 605.679
        z = depth[index] / 1000
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy      
        point = (x, y, z)
        points.append(point)

    return np.array(points)
    

def find_transformation(X, Y):
    """
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    print("det for rotation matrix is", np.linalg.det(R))
    if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
    print("det for rotation matrix is", np.linalg.det(R))
    # Determine translation vector
    t = cY - np.dot(R, cX)

    return R, t

def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))

def activate_gripper():
    args_instance = Args()

    #连接右机械臂
    ROBOT_RIP = args_instance.right_robot_ip
    rtde_R = rtde_receive.RTDEReceiveInterface(ROBOT_RIP)
    rtde_C = rtde_control.RTDEControlInterface(ROBOT_RIP)

    #连接夹爪
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_RIP, 63352)
    print("Gripper Connected!!!")
    gripper.activate()
    print("Gripper Activated!!!")

def main():
    args_instance = Args()

    #连接右机械臂
    ROBOT_RIP = args_instance.right_robot_ip
    rtde_R = rtde_receive.RTDEReceiveInterface(ROBOT_RIP)
    rtde_C = rtde_control.RTDEControlInterface(ROBOT_RIP)

    #连接夹爪
    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ROBOT_RIP, 63352)
    print("Gripper Connected!!!")
    #gripper.activate()
    print("Gripper Activated!!!")
    
    devices = initialize_camera_device()

    config = rs.config()
    config.enable_stream(rs.stream.depth, args_instance.resolution[0], args_instance.resolution[1], rs.format.z16,
                            args_instance.fps)
    config.enable_stream(rs.stream.color, args_instance.resolution[0], args_instance.resolution[1], rs.format.bgr8,
                            args_instance.fps)
    pipeline_1 = rs.pipeline()
    config.enable_device(devices[0])
    cfg = pipeline_1.start(config)
    print("please be ready to press Enter before initializing:")
    keyboard.wait('enter')
   #rtde_C.moveJ([-3.424551073704855, -2.4996212164508265, -0.8613818327533167, -2.8750675360309046, -0.3003757635699671, 1.4724793434143066])
    rtde_C.moveJ([-2.62, -2.91, -0.08, -2.933, 0.53, 1.17])
    print("Current End Effector Pose: ", rtde_R.getActualTCPPose())
    time.sleep(1)
    old_pose = rtde_R.getActualTCPPose()
    frames1 = pipeline_1.wait_for_frames()
    depth_frame0 = frames1.get_depth_frame()
    depth_bn_unaligned = np.asanyarray(depth_frame0.get_data())
    print("depth_bn_unaligned's shape:", depth_bn_unaligned.shape)
    depth_unaligned = cv2.normalize(depth_bn_unaligned, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image_old = Image.fromarray(depth_unaligned)
    depth_image_old.save('depth_image_old.png')
    print("unaligned Depth image saved as depth_image_old.png")

    frames1_align = rs.align(rs.stream.color).process(frames1)
  
  
    
    # 假设已经获取了 frames1_align
    color_frame1 = frames1_align.get_color_frame()
    depth_frame1 = frames1_align.get_depth_frame()

    # 将帧数据转换为 numpy 数组
    rgb_bn = np.asanyarray(color_frame1.get_data())
    depth_bn = np.asanyarray(depth_frame1.get_data())
    print("depth_bn's shape:", depth_bn.shape)
    print("rgb_bn's shape:", rgb_bn.shape)
    # 保存 RGB 图像为 PNG
    rgb_image = Image.fromarray(rgb_bn)
    rgb_image.save('rgb_image.png')
    print("RGB image saved as rgb_image.png")

    # 将深度数据标准化到 0-255 范围内，然后保存为 PNG
    depth_normalized = cv2.normalize(depth_bn, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image = Image.fromarray(depth_normalized)
    depth_image.save('depth_image.png')
    print("Depth image saved as depth_image.png")

    # 将 RGB 图像和深度图像叠加
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    combined_image = cv2.addWeighted(rgb_bn, 0.6, depth_colored, 0.4, 0)

    # 保存叠加后的图像
    combined_image_pil = Image.fromarray(combined_image)
    combined_image_pil.save('combined_image.png')
    print("Combined image saved as combined_image.png")
    print("please be ready to press Enter before test depolyment:")
    keyboard.wait('enter')
    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    #rtde_C.moveJ([-3.424551073704855, -2.4996212164508265, -0.8613818327533167, -2.8750675360309046, -0.3003757635699671, 0])
    rtde_C.moveJ([-2.62, -2.91, -0.08, -2.933, 0.53, 1.8])
    
    print("Current End Effector Pose: ", rtde_R.getActualTCPPose())
    signal = 0
    error = 100000
    while error > ERR_THRESHOLD:
        #Collect observations at the current pose.
        frames1 = pipeline_1.wait_for_frames()
        frames1_align = rs.align(rs.stream.color).process(frames1)
        color_frame1 = frames1_align.get_color_frame()
        depth_frame1 = frames1_align.get_depth_frame()
        rgb_live = np.asanyarray(color_frame1.get_data())
        depth_live = np.asanyarray(depth_frame1.get_data())
        print("now the correspondences compute begins", signal)
        #Compute pixel correspondences between new observation and bottleneck observation.
        with torch.no_grad():
                    points1, points2, image1_pil, image2_pil = find_correspondences(rgb_bn, rgb_live, num_pairs, load_size, layer,
                                                                                    facet, bin, thresh, model_type, stride)
                
                    print("key points:", points1, points2)
                    print("it's crazy to see image_pil's shape",np.array(image1_pil).shape, np.array(image2_pil).shape)
                    rgb_bn_image = Image.fromarray(rgb_bn)
                    rgb_live_image = Image.fromarray(rgb_live)  
                    fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
                    # 获取 Figure 对象的尺寸（以英寸为单位）
                    fig_size_inches = fig1.get_size_inches()

                    # 获取 Figure 对象的 DPI（每英寸点数）
                    fig_dpi = fig1.get_dpi()

                    # 计算图形的像素大小
                    fig_width_px = int(fig_size_inches[0] * fig_dpi)
                    fig_height_px = int(fig_size_inches[1] * fig_dpi)

                    print(f"Figure shape: (width: {fig_width_px}px, height: {fig_height_px}px)")
                    # 获取 Figure 对象中的所有 Axes 对象
                    axes = fig1.get_axes()

                    # 输出每个 Axes 对象的形状
                    for ax in axes:
                        print("examine the shape for axes object again", ax.get_position().bounds)  # 返回 (left, bottom, width, height) 以 Figure 的比例为单位
                    scale_factor = 480 / 224
                    points1 = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points1]
                    points2 = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points2]
                    print("points1",points1)
                    print("points2",points2)
                    fig3, fig4 = draw_correspondences(points1, points2, rgb_bn_image, rgb_live_image)
                    fig1.savefig('E:/figure1.png')
                    # 打开已经保存的图像文件
                    image_path = 'E:/figure1.png'
                    image = Image.open(image_path)

                    # 将 Image 转换为 numpy array
                    image_array = np.array(image)

                    # 获取图像的形状
                    image_shape = image_array.shape

                    print(f"Image shape: {image_shape}")
                    fig2.savefig('E:/figure2.png')
                    fig3.savefig('E:/figure3.png')
                    fig4.savefig('E:/figure4.png')
        #Given the pixel coordinates of the correspondences, add the depth channel.
        
        print("finish it")
        points1 = add_depth(points1, depth_bn, args_instance.NAME)
        points2 = add_depth(points2, depth_live, args_instance.NAME)
        print("3d points1",points1)
        print("3d points2",points2)        
        #Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)
        
        print("rotation matrix is", R)
        tcp_pose = rtde_R.getActualTCPPose()
        rot_vec_now = tcp_pose[3:]
        rot_mat_now = Rt.from_rotvec(rot_vec_now).as_matrix()
        rot_mat_next = np.dot(rot_mat_now, R)
        
        rotation = Rt.from_matrix(rot_mat_next).as_rotvec()
        #A function to convert pixel distance into meters based on calibration of camera.
        
        #profile = cfg.get_stream(rs.stream.depth)
        #profile1 = cfg.get_stream(rs.stream.color)
        #extrinsics = profile1.get_extrinsics_to(profile)
    
        #rot_mat = np.reshape(extrinsics.rotation, (3, 3))
        #tran_vec = np.reshape(extrinsics.translation, (3, 1))

        #t = np.reshape(t, (3, 1))
        
        delta = np.dot(rot_mat_now, t)
        #reverse_mat = rot_mat.T
        #t_meters = np.dot(reverse_mat, t - tran_vec)
        print("Current TCP Pose:", tcp_pose)
        print("Current Rotation Vector:", rot_vec_now)
        print("Current Rotation Matrix:\n", rot_mat_now)
        print("Next Rotation Vector:", rotation)
        print("Next Rotation Matrix:\n", rot_mat_next)
        print("true rotation matrix:",Rt.from_rotvec(old_pose[3:]).as_matrix())
        xyz_pos_now = tcp_pose[:3] + delta
        six_dof = np.concatenate((xyz_pos_now, rotation))
        six_dof = six_dof.tolist()

        #Move robot
        print("please be ready to press Enter:")
        keyboard.wait('enter')
        rtde_C.moveL(six_dof)
        error = compute_error(points1, points2)
        print("current error is:", error)
        time.sleep(1)
        frames1 = pipeline_1.wait_for_frames()
        frames1_align = rs.align(rs.stream.color).process(frames1)
        color_frame1 = frames1_align.get_color_frame()
        rgb_now = np.asanyarray(color_frame1.get_data())
        save_path = 'E:/now.png'
        cv2.imwrite(save_path, cv2.cvtColor(rgb_now, cv2.COLOR_RGB2BGR))
        print(f"Image saved to {save_path}")
        print("the current joint angles:",rtde_R.getActualQ())
        time.sleep(1)





if __name__ == '__main__':
    main()