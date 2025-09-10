# -*- coding: utf-8 -*-
"""
grasp_control.py (clean)
Franka Panda 在 PyBullet 中的最小抓取流程：
  - IK 规划到预抓取
  - 笛卡尔直线下探
  - 闭合夹爪
  - 抬起

依赖：
  pip install pybullet numpy

运行：
  python grasp_control.py
"""

from __future__ import annotations

# -------------------- 0) 彻底屏蔽 ExampleBrowser --------------------
import sys, os
# 有些环境会把 demo 参量塞进 argv，这里直接清空只保留脚本名
sys.argv = [sys.argv[0]]
# 常见会触发示例浏览器的环境变量
for k in ("PYBULLET_USE_EXAMPLE_BROWSER", "B3_USE_GUI_EXAMPLE_BROWSER"):
    os.environ.pop(k, None)

# -------------------- 1) 标准依赖 --------------------
import time
from typing import Tuple, List
import numpy as np
import pybullet as p
import pybullet_data as pd

# -------------------- 2) 工具函数 --------------------
def quintic(q0: np.ndarray, qf: np.ndarray, T: float, t: float):
    """五次多项式插值（端点速度/加速度为零）→ (q, qd, qdd)"""
    a0 = q0
    a1 = 0.0
    a2 = 0.0
    a3 = 10.0 * (qf - q0) / (T**3)
    a4 = -15.0 * (qf - q0) / (T**4)
    a5 =  6.0 * (qf - q0) / (T**5)
    tt = t
    q   = a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5
    qd  = a1 + 2*a2*tt + 3*a3*tt**2 + 4*a4*tt**3 + 5*a5*tt**4
    qdd = 2*a2 + 6*a3*tt + 12*a4*tt**2 + 20*a5*tt**3
    return q, qd, qdd

def slerp(q0, q1, t):
    """四元数球面插值（xyzw 顺序）"""
    q0 = q0/np.linalg.norm(q0); q1 = q1/np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        q = q0 + t*(q1 - q0)
        return q/np.linalg.norm(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta)/sin_0
    s1 = np.sin(theta)/sin_0
    return s0*q0 + s1*q1

def pose_lerp(p0, q0, p1, q1, t):
    """位姿插值：位置线性 + 姿态 slerp"""
    return (1-t)*p0 + t*p1, slerp(q0, q1, t)

# -------------------- 3) Panda 助手类 --------------------
# 若遇到“没动”，先将 USE_TORQUE=False 用位置控制验证流程
USE_TORQUE = True

class PandaHelper:
    def __init__(self, gui: bool=True, dt: float=1/240.0):
        self.dt = dt

        # 只在这里连接一次（GUI 失败自动回退到 DIRECT）
        cid = p.connect(p.GUI if gui else p.DIRECT)
        if cid < 0:
            print("[WARN] GUI connection failed; falling back to DIRECT.")
            cid = p.connect(p.DIRECT)

        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 世界
        p.loadURDF("plane.urdf")

        # 机器人
        self.robot = p.loadURDF("franka_panda/panda.urdf",
                                [0,0,0], [0,0,0,1], useFixedBase=True)

        self.arm_joints: List[int] = []
        self.gripper_joints: List[int] = []
        self.ee_link = 11  # 兜底；下方按链路名再自动匹配

        nJ = p.getNumJoints(self.robot)
        for j in range(nJ):
            info = p.getJointInf
