
"""
grasp_class.py
---------------
可复用的 Panda 抓取类（PyBullet）。
特点：
- 统一接口：pick_and_place(target_uid, place_xy, mode="cartesian_osc"|"joint_ctc")
- 两套传统控制链任选：
  1) 关节空间五次多项式 + 计算力矩法（Computed Torque）
  2) 笛卡尔 SE(3) 直线 + 操作空间控制（OSC）
- 自带夹爪开合、接触判据、摩擦设置、末端“朝下”约定（Y 轴向上）

依赖：control_classic.py, trajectories.py, robot_math.py（已随前置步骤生成）
"""

import time, math
import numpy as np
import pybullet as p

from control_classic import ComputedTorque, OperationalSpaceControl
from trajectories import joint_quintic_segment, cartesian_segment
from robot_math import make_T

class PandaGrasp:
    def __init__(self, pb, robot_id, ee_link, arm_joints, finger_joints=(9,10), dt=1/240.0):
        self.pb = pb
        self.rid = robot_id
        self.ee  = ee_link
        self.arm = list(arm_joints)
        self.fingers = list(finger_joints)
        self.dt = dt

        # 默认控制器（可在 configure_* 覆盖）
        self.ctc = ComputedTorque(pb, robot_id, self.arm, kp=[180]*len(self.arm), kd=[30]*len(self.arm))
        self.osc = OperationalSpaceControl(pb, robot_id, self.arm, ee_link)

        # 默认 OSC 增益（位置/姿态）
        self.Kp_osc = np.diag([500,500,500, 60,60,60])
        self.Kd_osc = np.diag([70,70,70, 10,10,10])

        # 末端“朝下”（Y-up）
        self.down_quat = self.pb.getQuaternionFromEuler([math.pi/2, 0, 0])
        self.down_R = np.array(self.pb.getMatrixFromQuaternion(self.down_quat)).reshape(3,3)

    # ---------- 配置与基础 ----------
    def set_dt(self, dt):
        self.dt = float(dt)

    def configure_ctc(self, kp, kd):
        self.ctc = ComputedTorque(self.pb, self.rid, self.arm, kp=kp, kd=kd)

    def configure_osc(self, Kp, Kd):
        self.Kp_osc = np.array(Kp, float)
        self.Kd_osc = np.array(Kd, float)

    def torque_mode(self):
        # 将臂的关节切换为 TORQUE_CONTROL（把位置/速度电机力清 0）
        for j in self.arm:
            self.pb.setJointMotorControl2(self.rid, j, self.pb.VELOCITY_CONTROL, force=0.0)

    def set_friction(self, body_uid, link_idx=-1, mu=1.2):
        self.pb.changeDynamics(body_uid, link_idx,
                               lateralFriction=float(mu),
                               rollingFriction=0.001,
                               spinningFriction=0.001,
                               restitution=0.0)

    # ---------- 夹爪 ----------
    def set_gripper(self, opening, steps=90, force=20):
        """opening≈[0.0, 0.04]；对称手指 9/10（URDF 里已齿轮约束）。"""
        opening = float(opening)
        for _ in range(int(steps)):
            for j in self.fingers:
                self.pb.setJointMotorControl2(self.rid, j, self.pb.POSITION_CONTROL,
                                              targetPosition=opening, force=force)
            self.pb.stepSimulation(); time.sleep(self.dt)

    def wait_until_contact(self, other_uid, timeout=1.0):
        t = 0.0
        while t < timeout:
            if self.pb.getContactPoints(self.rid, other_uid):
                return True
            self.pb.stepSimulation(); time.sleep(self.dt); t += self.dt
        return False

    # ---------- 末端姿态/IK ----------
    def _ik(self, pos, quat=None, rest=None):
        if quat is None: quat = self.down_quat
        qstar = self.pb.calculateInverseKinematics(self.rid, self.ee, pos, quat,
                                                   maxNumIterations=100, residualThreshold=1e-4,
                                                   restPoses=rest if rest is not None else [0]*len(self.arm))
        return np.array(qstar[:len(self.arm)])

    # ---------- Waypoints 生成（Y-up） ----------
    def waypoints_from_target(self, target_uid, place_xy, pre_h=0.12, grasp_h=0.01, lift_h=0.20, place_h=0.02):
        """根据目标 AABB 顶面自动生成关键点（pre, grasp, lift, phigh, place）。"""
        (ox, oy, oz), _ = self.pb.getBasePositionAndOrientation(target_uid)
        aabb_min, aabb_max = self.pb.getAABB(target_uid)
        top_y = aabb_max[1]
        pre   = [ox, top_y + pre_h,   oz]
        grasp = [ox, top_y + grasp_h, oz]
        lift  = [ox, top_y + lift_h,  oz]
        phigh = [place_xy[0], top_y + lift_h,  place_xy[1]]
        place = [place_xy[0], top_y + place_h, place_xy[1]]
        return pre, grasp, lift, phigh, place

    # ---------- 执行：关节空间 + CTC ----------
    def _run_joint_ctc(self, q_path, durations):
        for k in range(len(q_path)-1):
            q0 = q_path[k]; q1 = q_path[k+1]; T = durations[k]
            seg, Ttot = joint_quintic_segment(q0, q1, T)
            t=0.0
            while t < Ttot:
                qd, qd_dot, qd_ddot = seg(t)
                self.ctc.step(qd, qd_des=qd_dot, qdd_des=qd_ddot)
                self.pb.stepSimulation(); time.sleep(self.dt); t += self.dt

    # ---------- 执行：笛卡尔 SE(3) + OSC ----------
    def _run_cartesian_osc(self, T_list, durations):
        Tprev = T_list[0]
        for i in range(1, len(T_list)):
            Ttarget = T_list[i]
            f, Ttot = cartesian_segment(Tprev, Ttarget, durations[i-1], sampler="quintic")
            t=0.0
            while t < Ttot:
                Tref = f(t)
                xd  = Tref[:3,3]; xRd = Tref[:3,:3]
                self.osc.step(xd, xRd, self.Kp_osc, self.Kd_osc, local_pos=(0,0,0))
                self.pb.stepSimulation(); time.sleep(self.dt); t += self.dt
            Tprev = Ttarget

    # ---------- 统一接口 ----------
    def pick_and_place(self, target_uid, place_xy=(0.12,-0.55), mode="cartesian_osc",
                       pre_h=0.12, grasp_h=0.01, lift_h=0.20, place_h=0.02,
                       open_w=0.04, close_w=0.01,
                       durations=(1.2, 0.8, 1.0, 1.2, 0.9, 0.8)):
        """
        一行调用完成抓取：
        - 自动从 AABB 顶面推断高度
        - 生成关键点：pre→grasp→lift→phigh→place→phigh（最后回撤到放置高点）
        - 根据 mode 选择控制链并执行
        durations: 每一段的时长（秒），长度需为 6（含起点到 pre 的第一段）
        """
        assert mode in ("cartesian_osc", "joint_ctc")
        self.torque_mode()

        # 1) 关键点
        pre, grasp, lift, phigh, place = self.waypoints_from_target(
            target_uid, place_xy, pre_h, grasp_h, lift_h, place_h
        )

        # 2) 张开夹爪
        self.set_gripper(open_w, steps=90)

        # 3) 执行到 pre / grasp / lift / phigh / place / phigh
        if mode == "joint_ctc":
            # 用 IK 求各 waypoint 的 q*，各段五次多项式 + CTC
            qcur = np.array([self.pb.getJointState(self.rid, j)[0] for j in self.arm])
            q_pre   = self._ik(pre,   self.down_quat, rest=qcur)
            q_grasp = self._ik(grasp, self.down_quat, rest=q_pre)
            q_lift  = self._ik(lift,  self.down_quat, rest=q_grasp)
            q_phigh = self._ik(phigh, self.down_quat, rest=q_lift)
            q_place = self._ik(place, self.down_quat, rest=q_phigh)
            q_path = [qcur, q_pre, q_grasp, q_lift, q_phigh, q_place, q_phigh]
            self._run_joint_ctc(q_path, durations)

        else:  # cartesian_osc
            # 直接在 SE(3) 直线插值 + OSC 跟踪
            pos0, orn0 = self.pb.getLinkState(self.rid, self.ee)[:2]
            R0 = np.array(self.pb.getMatrixFromQuaternion(orn0)).reshape(3,3)
            T0 = make_T(R0, pos0)
            T_pre   = make_T(self.down_R, pre)
            T_grasp = make_T(self.down_R, grasp)
            T_lift  = make_T(self.down_R, lift)
            T_phigh = make_T(self.down_R, phigh)
            T_place = make_T(self.down_R, place)
            T_list = [T0, T_pre, T_grasp, T_lift, T_phigh, T_place, T_phigh]
            self._run_cartesian_osc(T_list, durations)

        # 4) 抓取夹紧 + 接触确认（在 grasp 段末已到位）
        self.set_gripper(close_w, steps=160)
        self.wait_until_contact(target_uid, timeout=0.8)

        # 5) 放置后张开
        self.set_gripper(open_w, steps=120)

        return True  # 可扩展更严格的成功判据
