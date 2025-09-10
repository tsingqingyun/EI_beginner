# -*- coding: utf-8 -*-
"""
grasp_minimal.py
A minimal, self-contained PyBullet script that demonstrates a complete
pick-and-lift grasp pipeline on a Franka Panda arm using *traditional*
robotics building blocks:
  - Basic kinematics (FK via getLinkState; IK via calculateInverseKinematics)
  - Simple joint-space interpolation (quintic, zero vel/accel at endpoints)
  - PD position control with gravity compensation (lightweight)
  - A clear grasp sequence: pregrasp -> approach -> close -> lift

Dependencies:
  pip install pybullet numpy

Run:
  python grasp_minimal.py
"""
from __future__ import annotations
import time
from typing import Tuple, List
import numpy as np
import pybullet as p
import pybullet_data as pd

# ----------------------------- Utils -----------------------------

def quintic(q0: np.ndarray, qf: np.ndarray, T: float, t: float):
    """Fifth-order (zero vel/accel boundary). Returns (q, qd, qdd)."""
    a0 = q0
    a1 = 0.0
    a2 = 0.0
    a3 = 10.0 * (qf - q0) / (T**3)
    a4 = -15.0 * (qf - q0) / (T**4)
    a5 = 6.0 * (qf - q0) / (T**5)
    tt = t
    q   = a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5
    qd  = a1 + 2*a2*tt + 3*a3*tt**2 + 4*a4*tt**3 + 5*a5*tt**4
    qdd = 2*a2 + 6*a3*tt + 12*a4*tt**2 + 20*a5*tt**3
    return q, qd, qdd

def slerp(q0, q1, t):
    """Quaternion slerp, xyzw order."""
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
    return (1-t)*p0 + t*p1, slerp(q0, q1, t)

# -------------------------- Panda Helpers ------------------------

class PandaHelper:
    def __init__(self, gui: bool=True, dt: float=1/240.0):
        self.dt = dt
        p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.81)

        # World
        p.loadURDF("plane.urdf")

        # Robot
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
        self.arm_joints: List[int] = []
        self.gripper_joints: List[int] = []
        self.ee_link = 11  # default for Panda hand; we'll also search by name

        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            name = info[1].decode()
            jtype = info[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if "finger" in name:
                    self.gripper_joints.append(j)
                else:
                    self.arm_joints.append(j)
            if "panda_hand" in name or "tool0" in name or "tcp" in name:
                self.ee_link = j

        # Default PD gains (joint space) + gravity comp
        self.kp = 50.0*np.ones(len(self.arm_joints))
        self.kd = 2*np.sqrt(self.kp)

        # Slightly open the gripper
        self.open_gripper(0.06)

    # --- State & FK ---
    def get_state(self):
        q, qd = [], []
        for j in self.arm_joints:
            js = p.getJointState(self.robot, j)
            q.append(js[0]); qd.append(js[1])
        return np.array(q), np.array(qd)

    def ee_fk(self):
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        return np.array(ls[4]), np.array(ls[5])

    # --- IK ---
    def ik(self, pos, quat):
        sol = p.calculateInverseKinematics(self.robot, self.ee_link, list(map(float,pos)), list(map(float,quat)),
                                           maxNumIterations=200, residualThreshold=1e-4)
        # Only first n arm joints
        n = len(self.arm_joints)
        return np.array(sol)[:n]

    # --- Control steps ---
    def step_pd_with_gravity(self, q_ref, qd_ref=None):
        q, qd = self.get_state()
        if qd_ref is None: qd_ref = np.zeros_like(q)
        e  = q_ref - q
        ed = qd_ref - qd
        tau = self.kp*e + self.kd*ed

        # Gravity compensation (inverse dynamics with zero desired accel/vel)
        full_q, full_qd = [], []
        ctrl_idx = []
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            jt = info[2]
            if jt in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                ctrl_idx.append(j)
                js = p.getJointState(self.robot, j)
                full_q.append(js[0]); full_qd.append(js[1])
        full_q = np.array(full_q); full_qd = np.array(full_qd)

        # Map arm targets into the full vector
        arm_indices = [ctrl_idx.index(j) for j in self.arm_joints]
        qdd = np.zeros_like(full_q)
        tau_id = np.array(p.calculateInverseDynamics(self.robot, list(full_q), list(full_qd), list(qdd)))
        # take only arm part
        tau_g = tau_id[arm_indices]

        tau += tau_g

        # apply to arm joints
        for i, j in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=float(tau[i]))
        p.stepSimulation()

    # --- Convenience ---
    def go_to_q(self, q_goal, T=2.0):
        q0, _ = self.get_state()
        steps = max(10, int(T/self.dt))
        for i in range(steps):
            t = i/(steps-1) * T
            q_ref, qd_ref, _ = quintic(q0, q_goal, T, t)
            self.step_pd_with_gravity(q_ref, qd_ref)
            time.sleep(self.dt)

    def move_lincart(self, p0, q0, p1, q1, T=2.0):
        steps = max(10, int(T/self.dt))
        for i in range(steps):
            s = i/(steps-1)
            pd, qd = pose_lerp(p0, q0, p1, q1, s)
            q_cmd = self.ik(pd, qd)
            self.step_pd_with_gravity(q_cmd, np.zeros_like(q_cmd))
            time.sleep(self.dt)

    # --- Gripper ---
    def open_gripper(self, width=0.08, force=40):
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=width/2.0, force=force)
        for _ in range(120):
            p.stepSimulation(); time.sleep(self.dt)

    def close_gripper(self, width=0.0, force=80):
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=width/2.0, force=force)
        for _ in range(240):
            p.stepSimulation(); time.sleep(self.dt)

    # --- Objects ---
    def spawn_box(self, size=(0.05,0.05,0.05), mass=0.2, pose=((0.6,0.0,0.025),(0,0,0,1))):
        half = [s/2 for s in size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.9,0.4,0.2,1])
        box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                basePosition=pose[0], baseOrientation=pose[1])
        return box

    # --- Grasp (pre-approach-close-lift) ---
    def grasp(self, obj_pose: Tuple[np.ndarray, np.ndarray],
              pre_offset=np.array([0,0,0.18]),   # above object
              lift_offset=np.array([0,0,0.20]),  # lift up
              approach_T=1.5):
        obj_p, obj_q = obj_pose
        pre_p = obj_p + pre_offset
        pre_q = obj_q  # (simple: same orientation)

        # 1) open
        self.open_gripper(0.08)

        # 2) go pregrasp (joint quintic to IK target)
        q_pre = self.ik(pre_p, pre_q)
        self.go_to_q(q_pre, T=2.0)

        # 3) descend straight
        ee_p, ee_q = self.ee_fk()
        self.move_lincart(ee_p, ee_q, obj_p, obj_q, T=approach_T)

        # 4) close
        self.close_gripper(0.0, force=100)

        # 5) lift
        lift_p = obj_p + lift_offset
        self.move_lincart(obj_p, obj_q, lift_p, obj_q, T=1.5)


if __name__ == "__main__":
    dt = 1/240.0
    panda = PandaHelper(gui=True, dt=dt)

    # Home posture (comfortable)
    home = np.array([0, -0.6, 0, -2.4, 0, 1.8, 0.8])
    panda.go_to_q(home, T=2.5)

    # Place a box and fetch its pose
    box_id = panda.spawn_box(size=(0.05,0.05,0.05), mass=0.2,
                             pose=((0.6, 0.0, 0.025), p.getQuaternionFromEuler([np.pi, 0, 0])))
    pos, ori = p.getBasePositionAndOrientation(box_id)
    obj_pose = (np.array(pos), np.array(ori))

    # Execute grasp
    panda.grasp(obj_pose, pre_offset=np.array([0,0,0.18]), lift_offset=np.array([0,0,0.20]), approach_T=1.5)

    # Idle so user can observe
    for _ in range(5*240):
        p.stepSimulation()
        time.sleep(dt)
