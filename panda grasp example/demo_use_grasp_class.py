
"""
demo_use_grasp_class.py
-----------------------
演示如何在现有 Panda 场景中调用 PandaGrasp 一次性完成 Pick→Place。
"""

import time, math
import pybullet as p
import pybullet_data as pd
import numpy as np

import panda_sim_grasp as panda_sim
from grasp_class import PandaGrasp

FPS=240.0; DT=1.0/FPS

def connect():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setTimeStep(DT)
    p.setGravity(0,-9.8,0)

def main():
    connect()
    sim = panda_sim.PandaSim(p, [0,0,0])
    robot = sim.panda
    ee = panda_sim.pandaEndEffectorIndex
    arm = list(range(panda_sim.pandaNumDofs))  # 0..6
    fingers = (9,10)

    # 构造抓取器
    grasp = PandaGrasp(p, robot, ee, arm_joints=arm, finger_joints=fingers, dt=DT)

    # 提升摩擦，减少打滑
    for f in fingers:
        grasp.set_friction(robot, f, mu=1.2)
    grasp.set_friction(sim.legos[0], -1, mu=1.0)

    # 一行搞定：选择控制链
    target = sim.legos[0]
    # 方案1：笛卡尔 + OSC（更顺应）
    grasp.pick_and_place(target, place_xy=(0.12,-0.55), mode="cartesian_osc")
    # 方案2：关节 + 计算力矩法（可替换上一行）
    # grasp.pick_and_place(target, place_xy=(0.12,-0.55), mode="joint_ctc")

    print("Done. Ctrl+C 退出。")
    while True:
        p.stepSimulation(); time.sleep(DT)

if __name__ == "__main__":
    main()
