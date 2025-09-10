# pick_place_demo.py
import time, math
import numpy as np
import pybullet as p
import pybullet_data as pd
from panda_sim_grasp import pandaEndEffectorIndex, pandaNumDofs, PandaSim  # 直接用你已有的
# ↑ panda_sim_grasp.py 里定义了 end-effector index=11、7个关节，以及 PandaSim 类

FPS = 240.
DT = 1.0 / FPS

def connect():
    p.connect(p.GUI)  # 可改 p.DIRECT
    p.setAdditionalSearchPath(pd.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setTimeStep(DT)
    p.setGravity(0, -9.8, 0)

def set_friction(body_uid, link_idx=-1, mu=1.2):
    p.changeDynamics(body_uid, link_idx,
                     lateralFriction=mu,
                     rollingFriction=0.001,
                     spinningFriction=0.001,
                     restitution=0.0)

def move_cartesian_line(robot_uid, ee_link, start, goal, orn, steps=120):
    """ 直线插值一条笛卡尔路径；每步做一次 IK + 关节位置伺服 """
    for k in range(steps):
        alpha = (k + 1) / float(steps)
        tgt = [start[i] * (1 - alpha) + goal[i] * alpha for i in range(3)]
        q = p.calculateInverseKinematics(
            robot_uid, ee_link, tgt, orn,
            maxNumIterations=50, residualThreshold=1e-4
        )
        # 只发 0..6 的 7 个关节
        for j in range(pandaNumDofs):
            p.setJointMotorControl2(robot_uid, j, p.POSITION_CONTROL,
                                    targetPosition=q[j],
                                    positionGain=0.2, velocityGain=1.0,
                                    force=5 * 240.)
        p.stepSimulation()
        time.sleep(DT)

def set_gripper(robot_uid, opening, steps=60):
    """ 控制两指对称关节 (9,10) 的开度；opening ∈ [0.0 ~ 0.04] """
    for _ in range(steps):
        for j in [9, 10]:
            p.setJointMotorControl2(robot_uid, j, p.POSITION_CONTROL,
                                    targetPosition=opening, force=20)
        p.stepSimulation()
        time.sleep(DT)

def wait_until_contact(robot_uid, target_uid, timeout=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        cps = p.getContactPoints(bodyA=robot_uid, bodyB=target_uid)
        if len(cps) > 0:
            return True
        p.stepSimulation()
        time.sleep(DT)
    return False

def main():
    connect()
    sim = PandaSim(p, [0, 0, 0])  # 你的 PandaSim：会加载托盘、乐高、平面等
    robot = sim.panda
    lego = sim.legos[0]

    # 调摩擦：手指、目标
    for finger_link in [9, 10]:
        set_friction(robot, finger_link, mu=1.2)
    set_friction(lego, -1, mu=1.0)

    # 准备常用姿态：末端朝下（注意：Y 轴为“上”）
    down_orn = p.getQuaternionFromEuler([math.pi/2., 0., 0.])

    # 读取目标 pose & 顶面高度
    (ox, oy, oz), _ = p.getBasePositionAndOrientation(lego)
    aabb_min, aabb_max = p.getAABB(lego)
    top_y = aabb_max[1]

    # 关键点（世界坐标, Y 是高度）：
    pre_grasp = [ox, top_y + 0.12, oz]
    grasp     = [ox, top_y + 0.01, oz]
    lift_high = [ox, top_y + 0.20, oz]
    place_xy  = [0.10, -0.55]              # 你想放置到托盘内的某点 (x, z)
    place_high= [place_xy[0], top_y + 0.20, place_xy[1]]
    place     = [place_xy[0], top_y + 0.02, place_xy[1]]

    # 1) 张开夹爪
    set_gripper(robot, opening=0.04, steps=90)

    # 2) 移动到 pre-grasp
    cur_pos = p.getLinkState(robot, pandaEndEffectorIndex)[0]
    move_cartesian_line(robot, pandaEndEffectorIndex, cur_pos, pre_grasp, down_orn, steps=150)

    # 3) 下降到抓取高度
    move_cartesian_line(robot, pandaEndEffectorIndex, pre_grasp, grasp, down_orn, steps=120)

    # 4) 闭合夹爪并等到接触
    set_gripper(robot, opening=0.01, steps=180)
    wait_until_contact(robot, lego, timeout=0.8)

    # 5) 抬起
    move_cartesian_line(robot, pandaEndEffectorIndex, grasp, lift_high, down_orn, steps=180)

    # 6) 水平移动到放置上方
    move_cartesian_line(robot, pandaEndEffectorIndex, lift_high, place_high, down_orn, steps=220)

    # 7) 下降放置
    move_cartesian_line(robot, pandaEndEffectorIndex, place_high, place, down_orn, steps=120)

    # 8) 松开并回撤
    set_gripper(robot, opening=0.04, steps=120)
    move_cartesian_line(robot, pandaEndEffectorIndex, place, place_high, down_orn, steps=120)

    print("Done. 按 Ctrl+C 退出。")
    while True:
        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
