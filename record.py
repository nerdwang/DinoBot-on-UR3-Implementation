import rtde_receive
import rtde_control
import csv
import time

# 连接到机器人
robot_ip = "192.168.3.195"
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
rtde_c = rtde_control.RTDEControlInterface(robot_ip)

# 打开 CSV 文件准备写入数据
with open("recorded_joint_data.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['timestamp', 'actual_q', 'actual_qd', 'target_qd_acceleration'])

    try:
        while True:
            # 获取当前时间戳
            timestamp = time.time()
            # 获取关节位置
            actual_q = rtde_r.getActualQ()
            # 获取关节速度
            actual_qd = rtde_r.getActualQd()
            target_qd_acceleration = [0.0] * 6

            # 将数据写入 CSV 文件
            writer.writerow([timestamp, actual_q, actual_qd, target_qd_acceleration])
            time.sleep(0.008)  # 控制数据记录频率
    except KeyboardInterrupt:
        pass
