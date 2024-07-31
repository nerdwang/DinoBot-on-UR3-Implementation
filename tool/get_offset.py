from tool.dynamixel_driver import DynamixelDriver

import json
import numpy as np


def get_arm_offset(port):
    """获取Leader机械臂的各个关节角和实机UR关节角之间的关系。返回offset (list)。
    因为Leader的机械臂在初始位置的时候，误差一定是np.pi/2的整数倍，所以只需计算Leader与Follower机械臂的关节角差值是np.pi/2的几倍。
    ids是DYNAMIXEL电机的id数，从base link到tool，从1到6编码。夹爪为7。
            Args:
                port (str): DYNAMIXEL舵机与电脑通信的端口，Windows一般为COM开头。例如COM4。
            """
    ids = [1, 2, 3, 4, 5, 6, 7]
    driver = DynamixelDriver(ids, port=port)
    driver.set_torque_mode(False)
    joint_angles = driver.get_joints()
    raw_joint_angles = [f"{angle:.4f}" for angle in joint_angles]
    raw_joints = [float(angle) for angle in raw_joint_angles[:6]]
    joint_offsets = [joint / (np.pi / 2) for joint in raw_joints]
    arm_offset = [round(joint_offset) for joint_offset in joint_offsets]
    print("六轴机械臂的offset值为：", arm_offset)
    driver.close()
    return arm_offset


def get_gripper_offset(port):
    """获取夹爪舵机的与实际夹爪的偏移值offset。
                Args:
                    port (str): DYNAMIXEL舵机与电脑通信的端口，Windows一般为COM开头。例如COM4。
                """
    gripper_id = [7]
    driver = DynamixelDriver(gripper_id, port=port)
    driver.set_torque_mode(False)
    gripper_rad = driver.get_joints()  # 弧度
    gripper_angle = int(np.rad2deg(gripper_rad[0]))
    gripper_open = gripper_angle - 1
    gripper_close = gripper_angle - 42
    gripper_offset = [gripper_open, gripper_close]
    print("夹爪的offset值为：", gripper_offset)
    driver.close()
    return gripper_offset


def main():
    port = "COM9"
    arm_offset = get_arm_offset(port)
    gripper_offset = get_gripper_offset(port)
    data = {
        "arm_offset": arm_offset,
        "gripper_offset": gripper_offset
    }
    with open('offsets.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
