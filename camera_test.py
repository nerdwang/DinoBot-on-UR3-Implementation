import pyrealsense2 as rs
import numpy as np

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

args_instance = Args()


def initialize_camera_device():
    devices = []
    for device in rs.context().devices:
        print('Found device: ', device.get_info(rs.camera_info.name), ' ',
              device.get_info(rs.camera_info.serial_number))
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            devices.append(device.get_info(rs.camera_info.serial_number))
    return devices

devices = initialize_camera_device()

config = rs.config()
config.enable_stream(rs.stream.depth, args_instance.resolution[0], args_instance.resolution[1], rs.format.z16,
                        args_instance.fps)
config.enable_stream(rs.stream.color, args_instance.resolution[0], args_instance.resolution[1], rs.format.bgr8,
                        args_instance.fps)
pipeline_1 = rs.pipeline()
config.enable_device(devices[0])
cfg = pipeline_1.start(config)

frames1 = pipeline_1.wait_for_frames()
frames1_align = rs.align(rs.stream.color).process(frames1)
color_frame1 = frames1_align.get_color_frame()
depth_frame1 = frames1_align.get_depth_frame()
rgb_bn = np.asanyarray(color_frame1.get_data())
depth_bn = np.asanyarray(depth_frame1.get_data())

print(depth_bn.max())