
"""
control_classic.py
------------------
Classic robot controllers (PyBullet-backed):
- Joint-space PD with gravity compensation
- Computed-torque (inverse dynamics) control
- Operational Space Control (OSC, Khatib-style) for end-effector tasks

These controllers depend on PyBullet-provided dynamics: mass matrix, inverse dynamics, Jacobians.
"""

import numpy as np

class JointSpacePD:
    def __init__(self, pb, robot_id, joint_indices, kp, kd, gravity_comp=True):
        self.pb = pb
        self.rid = robot_id
        self.idx = list(joint_indices)
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.gc = gravity_comp

    def step(self, q_des, qd_des=None):
        q  = np.array([self.pb.getJointState(self.rid, j)[0] for j in self.idx])
        qd = np.array([self.pb.getJointState(self.rid, j)[1] for j in self.idx])
        if qd_des is None:
            qd_des = np.zeros_like(q)
        e  = q_des - q
        ed = qd_des - qd
        tau = self.kp*e + self.kd*ed
        if self.gc:
            g = np.array(self.pb.calculateInverseDynamics(self.rid, q.tolist(),
                                                          [0.0]*len(self.idx),
                                                          [0.0]*len(self.idx)))
            tau = tau + g
        for t, j in zip(tau, self.idx):
            self.pb.setJointMotorControl2(self.rid, j, self.pb.TORQUE_CONTROL, force=float(t))

class ComputedTorque:
    """
    tau = M(q) * (qdd_des + Kd e_dot + Kp e) + h(q, qd), where h = C + g.
    Use PyBullet mass matrix and inverse dynamics to approximate h.
    """
    def __init__(self, pb, robot_id, joint_indices, kp, kd):
        self.pb = pb
        self.rid = robot_id
        self.idx = list(joint_indices)
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)

    def step(self, q_des, qd_des=None, qdd_des=None):
        q  = np.array([self.pb.getJointState(self.rid, j)[0] for j in self.idx])
        qd = np.array([self.pb.getJointState(self.rid, j)[1] for j in self.idx])
        n = len(self.idx)
        if qd_des is None:  qd_des  = np.zeros(n)
        if qdd_des is None: qdd_des = np.zeros(n)
        e  = q_des - q
        ed = qd_des - qd
        v  = qdd_des + self.kd*ed + self.kp*e  # desired "virtual" acceleration

        M = np.array(self.pb.calculateMassMatrix(self.rid, q.tolist()))
        # h(q,qd) approximated via inverse dynamics with zero qdd:
        h = np.array(self.pb.calculateInverseDynamics(self.rid, q.tolist(), qd.tolist(),
                                                      [0.0]*n))
        tau = M @ v + h
        for t, j in zip(tau, self.idx):
            self.pb.setJointMotorControl2(self.rid, j, self.pb.TORQUE_CONTROL, force=float(t))

class OperationalSpaceControl:
    """
    6D OSC with mass-weighted task-space inertia (Lambda) and simple PD in task space.
    tau = J^T [ Lambda (xdd_des + Kd xe_dot + Kp xe) + mu + p ]
    Here we approximate mu+p (Cor+grav) with bias forces projected to task space.
    """
    def __init__(self, pb, robot_id, joint_indices, ee_link):
        self.pb = pb
        self.rid = robot_id
        self.idx = list(joint_indices)
        self.ee  = ee_link

    def step(self, xd, xRd, Kp, Kd, local_pos=(0,0,0)):
        """
        xd: desired ee position (3,)
        xRd: desired ee rotation matrix (3,3)
        """
        n = len(self.idx)
        # states
        q  = np.array([self.pb.getJointState(self.rid, j)[0] for j in self.idx])
        qd = np.array([self.pb.getJointState(self.rid, j)[1] for j in self.idx])

        # Jacobian (world frame)
        zero = [0.0]*n
        jlin, jang = self.pb.calculateJacobian(self.rid, self.ee, local_pos, q.tolist(), zero, zero)
        J = np.vstack([np.array(jlin), np.array(jang)])  # 6xn

        # Mass matrix and its inverse
        M = np.array(self.pb.calculateMassMatrix(self.rid, q.tolist()))
        Minv = np.linalg.inv(M)

        # Task-space inertia
        Lambda_inv = J @ Minv @ J.T
        # Regularize in case of singularity
        reg = 1e-6*np.eye(6)
        Lambda = np.linalg.inv(Lambda_inv + reg)

        # Current EE pose
        x, orn = self.pb.getLinkState(self.rid, self.ee)[0], self.pb.getLinkState(self.rid, self.ee)[1]
        R = np.array(self.pb.getMatrixFromQuaternion(orn)).reshape(3,3)

        # Pose error (world): position + orientation (axis-angle)
        ex = np.array(xd) - np.array(x)
        Rerr = xRd @ R.T
        # orientation error as rotation vector
        angle = np.arccos(np.clip((np.trace(Rerr)-1)/2, -1, 1))
        if angle < 1e-6:
            eR = np.zeros(3)
        else:
            eR = angle/(2*np.sin(angle)) * np.array([Rerr[2,1]-Rerr[1,2], Rerr[0,2]-Rerr[2,0], Rerr[1,0]-Rerr[0,1]])

        e = np.r_[ex, eR]

        # Bias forces h = C+g
        h = np.array(self.pb.calculateInverseDynamics(self.rid, q.tolist(), qd.tolist(), [0.0]*n))
        # task-space bias: mu+p ≈ Lambda * (J M^{-1} h)
        mu_p = Lambda @ (J @ (Minv @ h))

        # Desired task wrench
        xdot = J @ qd
        F = Lambda @ (-Kd @ xdot - Kp @ e) + mu_p

        tau = J.T @ F
        for t, j in zip(tau, self.idx):
            self.pb.setJointMotorControl2(self.rid, j, self.pb.TORQUE_CONTROL, force=float(t))
