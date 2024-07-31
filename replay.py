import rtde_control
import csv
import time

# 连接到机器人
robot_ip = "192.168.3.195"
rtde_c = rtde_control.RTDEControlInterface(robot_ip)

# 读取记录的关节数据
with open("recorded_joint_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头
    for row in reader:
        timestamp, actual_q, actual_qd, target_qd_acceleration = row
        joint_positions = [float(val) for val in actual_q.strip('[]').split(',')]
        joint_speeds = [float(val) for val in actual_qd.strip('[]').split(',')]
        
        # 使用 speedJ 函数进行速度控制
        rtde_c.speedJ(joint_speeds, 0.3, 0.008)  # 设置速度，加速度和时间参数
        time.sleep(0.008)  # 控制时间间隔

rtde_c.disconnect()



