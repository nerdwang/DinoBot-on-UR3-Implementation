    
import pyrealsense2 as rs
print("pls do not be empty",rs.context().devices)

devices = []
resolution = (640, 480)
#初始化相机们
for device in rs.context().devices:
    print('Found device: ', device.get_info(rs.camera_info.name), ' ',
            device.get_info(rs.camera_info.serial_number))
    print("show me what can you get")
    if device.get_info(rs.camera_info.name).lower() != 'platform camera':
        devices.append(device.get_info(rs.camera_info.serial_number))

config = rs.config()
config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)