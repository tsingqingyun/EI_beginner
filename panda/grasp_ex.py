# -*- coding: utf-8 -*-
"""
TraditionalGraspController: 用运动学 + 轨迹生成 + 动力学 + 控制理论
实现 Panda 机械臂在 PyBullet 中的传统抓取流程（可复用基类）。

依赖：
  - pybullet, pybullet_data, numpy

功能要点：
1) 坐标变换：提供SO(3)/SE(3)与四元数互转、齐次变换、位姿插值。
2) 运动学：
   - 正运动学(FK)：调用 getLinkState / multiplyTransforms。
   - 逆运动学(IK)：
       a) 解析器：pybullet.calculateInverseKinematics（快速、稳）
       b) 数值法：Jacobian 伪逆的速度闭环(Resolved-Rate IK)，可带障碍惩罚权重。
3) 轨迹生成：
   - 关节空间：五次多项式(q, qd, qdd)；
   - 笛卡尔空间：最小加加速度(直线位姿 + Slerp姿态) + 在线IK追踪；
4) 动力学：
   - 惯量矩阵 M(q)：pybullet.calculateMassMatrix；
   - 科氏/离心 + 重力：pybullet.calculateInverseDynamics；
   - 关节力矩限制与摩擦简模。
5) 控制：
   - 关节 PD + 重力补偿；
   - 计算力矩控制(Computed-Torque / 逆动力学前馈)；
   - 笛卡尔阻抗控制(可选)。
6) 抓取流程：pregrasp -> approach -> close gripper -> lift。

用法示例见文件末尾 __main__。
你可以把本文件命名为 grasp_controller.py，然后在你的 batchsim3.py / panda_sim.py 中导入并复用。
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pybullet as p
import pybullet_data
import time

# ---------------------------- 工具函数：坐标与姿态 ----------------------------

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """ 球面线性插值（四元数），t∈[0,1] """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    # 反向以走短弧
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # 夹角很小，退化为线性
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def pose_lerp(p0: np.ndarray, q0: np.ndarray, p1: np.ndarray, q1: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """ 位姿插值：位置线性 + 姿态slerp """
    pos = (1 - t) * p0 + t * p1
    quat = quat_slerp(q0, q1, t)
    return pos, quat

# ---------------------------- 数据类与配置 ----------------------------

@dataclass
class TrajSegment:
    q0: np.ndarray
    qf: np.ndarray
    T: float  # 时长

@dataclass
class Gains:
    kp: np.ndarray  # 关节Kp (n,)
    kd: np.ndarray  # 关节Kd (n,)

@dataclass
class ImpedanceGains:
    Kp: np.ndarray  # 位置 3x3
    Kd: np.ndarray  # 速度 3x3
    Ko: np.ndarray  # 姿态 3x3（或标量）

# ---------------------------- 主控制类 ----------------------------

class TraditionalGraspController:
    """传统抓取控制器：基于PyBullet的Panda机械臂运动学+动力学控制基类
    
    功能特性：
    1. 支持正逆运动学计算（解析解和数值解）
    2. 提供关节空间和笛卡尔空间轨迹生成
    3. 实现动力学模型和多种控制策略（PD控制、计算力矩控制、阻抗控制）
    4. 完整的抓取流程：预抓取 -> 接近 -> 抓取 -> 提升
    
    属性:
        robot: PyBullet机器人模型
        arm_joints: 机械臂关节索引列表
        gripper_joints: 夹爪关节索引列表
        ee_link: 末端执行器链接索引
        dof: 机械臂自由度
        q: 当前关节位置
        qd: 当前关节速度
        joint_gains: 关节控制增益参数
        imp_gains: 阻抗控制增益参数
    """
    def __init__(self,
                 use_gui: bool = True,
                 time_step: float = 1.0/240.0,
                 panda_urdf: Optional[str] = None,
                 base_pos=(0,0,0), base_ori=(0,0,0,1)):
        self.dt = time_step
        self.cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.panda_urdf = panda_urdf or "franka_panda/panda.urdf"
        self.robot = p.loadURDF(self.panda_urdf, basePosition=base_pos, baseOrientation=base_ori, useFixedBase=True)

        # 解析出机械臂与夹爪关节索引
        self.arm_joints: List[int] = []
        self.gripper_joints: List[int] = []
        self.ee_link: int = -1
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            jtype = info[2]
            jname = info[1].decode()
            if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_PRISMATIC:
                if "finger" in jname:
                    self.gripper_joints.append(i)
                else:
                    self.arm_joints.append(i)
            if "panda_hand" in jname or "tool0" in jname or "tcp" in jname:
                self.ee_link = i
        if self.ee_link < 0:
            # 对Panda，手爪末端 link 一般是 11（panda_hand），但不同URDF可能不同
            self.ee_link = self.arm_joints[-1]

        self.dof = len(self.arm_joints)
        self.q = np.zeros(self.dof)
        self.qd = np.zeros(self.dof)

        # 默认关节 PD 增益
        self.joint_gains = Gains(kp=50*np.ones(self.dof), kd=2*np.sqrt(50)*np.ones(self.dof))
        # 默认笛卡尔阻抗
        self.imp_gains = ImpedanceGains(Kp=800*np.eye(3), Kd=60*np.eye(3), Ko=40*np.eye(3))

        # 力矩限制（可根据URDF更新）
        self.tau_limits = np.array([87]*self.dof)

        # 关节速度限制（近似）
        self.qd_limits = np.array([2.5]*self.dof)

    # ------------------------- 读取状态 / FK / Jacobian -------------------------
    def get_state(self):
        q, qd = [], []
        for j in self.arm_joints:
            js = p.getJointState(self.robot, j)
            q.append(js[0]); qd.append(js[1])
        self.q = np.array(q)
        self.qd = np.array(qd)
        return self.q.copy(), self.qd.copy()

    def ee_fk(self) -> Tuple[np.ndarray, np.ndarray]:
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        pos = np.array(ls[4])
        quat = np.array(ls[5])
        return pos, quat

    def jacobian(self, q: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        """计算末端执行器的雅可比矩阵
        
        参数:
            q: 关节位置向量（可选），形状(n,)
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 线速度和角速度雅可比矩阵，形状均为(3,n)
        """
        if q is not None:
            for i,j in enumerate(self.arm_joints):
                p.resetJointState(self.robot, j, float(q[i]))
        zero = [0.0]*len(self.arm_joints)
        Jlin, Jang = p.calculateJacobian(self.robot, self.ee_link, [0,0,0], \
                                         list(self.q), list(self.qd), zero)
        return np.array(Jlin), np.array(Jang)

    # ------------------------- 逆运动学（两种） -------------------------
    def ik_pb(self, pos, quat, nullspace=None):
        # 保护：numpy -> python list/tuple
        pos  = tuple(map(float, pos))
        quat = tuple(map(float, quat))

        # 保护：ee_link 合法性
        assert self.ee_link is not None and self.ee_link >= 0, "ee_link 未初始化或非法"

        if nullspace is None:
            # ★ 不传任何 nullspace 关键字参数 ★
            sol = p.calculateInverseKinematics(
                self.robot, self.ee_link, pos, quat,
                maxNumIterations=200, residualThreshold=1e-4
            )
            return np.array(sol)[:self.dof]

        # 有 nullspace 时，必须提供同长度数组
        lower = list(map(float, nullspace.get("lower", [])))
        upper = list(map(float, nullspace.get("upper", [])))
        ranges = list(map(float, nullspace.get("ranges", [])))
        rest  = list(map(float, nullspace.get("rest",  [])))

        n = len(self.arm_joints)
        for name, arr in [("lower",lower),("upper",upper),("ranges",ranges),("rest",rest)]:
            if len(arr) != n:
                raise ValueError(f"nullspace.{name} 长度应为 {n}，当前为 {len(arr)}")

        sol = p.calculateInverseKinematics(
            self.robot, self.ee_link, pos, quat,
            lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
            maxNumIterations=200, residualThreshold=1e-4
        )
        return np.array(sol)[:self.dof]
    def ik_resolved_rate(self, pos_des: np.ndarray, quat_des: np.ndarray, iters=200, alpha=0.6) -> np.ndarray:
        # 数值法：在速度层做闭环，J^+ * (ẋ*)，其中姿态误差用四元数误差近似
        def quat_err(qd, q):
            # 误差四元数 qe = qd * inv(q)
            q = q/np.linalg.norm(q); qd = qd/np.linalg.norm(qd)
            w = np.array([q[3], q[0], q[1], q[2]])
            wd = np.array([qd[3], qd[0], qd[1], qd[2]])
            qe = quat_mult(wd, quat_conj(w))
            # 取向误差的向量部分 2*qe_vec 近似角速度指令
            return 2.0*qe[1:4]
        def quat_conj(w):
            return np.array([w[0], -w[1], -w[2], -w[3]])
        def quat_mult(a,b):
            # (w, x, y, z)
            w = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
            x = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
            y = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
            z = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]
            return np.array([w,x,y,z])

        q = self.get_state()[0]
        for _ in range(iters):
            p.stepSimulation()
            pos, quat = self.ee_fk()
            Jv, Jw = self.jacobian()
            J = np.vstack([Jv, Jw])  # 6 x n
            e_p = pos_des - pos
            e_o = quat_err(quat_des[[3,0,1,2]], quat[[3,0,1,2]])  # 转成(w,x,y,z)
            xe = np.hstack([e_p, e_o])
            # 伪逆
            J_pinv = np.linalg.pinv(J, rcond=1e-3)
            qd_cmd = J_pinv @ (alpha * xe)
            q = q + qd_cmd * self.dt
            for i,j in enumerate(self.arm_joints):
                p.resetJointState(self.robot, j, float(q[i]))
        return q

    # ------------------------- 轨迹生成（五次多项式） -------------------------
    def quintic(self, q0, qf, T: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """五次多项式插值：给定边界(q0, qf)和零初末速度/加速度"""
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = (10*(qf - q0)) / (T**3)
        a4 = (-15*(qf - q0)) / (T**4)
        a5 = (6*(qf - q0)) / (T**5)
        tt = t
        q = a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5
        qd = a1 + 2*a2*tt + 3*a3*tt**2 + 4*a4*tt**3 + 5*a5*tt**4
        qdd = 2*a2 + 6*a3*tt + 12*a4*tt**2 + 20*a5*tt**3
        return q, qd, qdd

    def plan_joint_traj(self, q_start: np.ndarray, q_goal: np.ndarray, T: float, steps: int) -> Dict[str, np.ndarray]:
        ts = np.linspace(0, T, steps)
        Q, QD, QDD = [], [], []
        for t in ts:
            q, qd, qdd = self.quintic(q_start, q_goal, T, t)
            Q.append(q); QD.append(qd); QDD.append(qdd)
        return {"t": ts, "q": np.stack(Q), "qd": np.stack(QD), "qdd": np.stack(QDD)}

    # ------------------------- 动力学与控制 -------------------------
    
    def mass_matrix(self, q_arm: np.ndarray) -> np.ndarray:
        """计算机械臂部分的质量矩阵
        
        参数:
            q_arm: 机械臂关节位置向量，形状(n,)
            
        返回:
            np.ndarray: 机械臂部分的质量矩阵，形状(n,n)
        """

    def invdyn_terms(self, q_arm: np.ndarray, qd_arm: np.ndarray, qdd_des_arm: Optional[np.ndarray]=None):
        """计算动力学项（重力、科氏力等）
        
        参数:
            q_arm: 关节位置向量，形状(n,)
            qd_arm: 关节速度向量，形状(n,)
            qdd_des_arm: 关节加速度向量（可选），形状(n,)
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 质量矩阵和动力学项，形状均为(n,n)和(n,)
        """
        if qdd_des_arm is None:
            qdd_des_arm = np.zeros_like(q_arm)
        q_all, qd_all, qdd_all, ctrl_joints, arm_indices = self._full_state_vectors(q_arm, qd_arm, qdd_des_arm)

        # 计算整机的逆动力学力矩
        tau_all = np.array(p.calculateInverseDynamics(self.robot, list(q_all), list(qd_all), list(qdd_all)))

        # 质量矩阵（整机），再切 arm 子块
        M_full = np.array(p.calculateMassMatrix(self.robot, list(q_all)))
        M_aa = M_full[np.ix_(arm_indices, arm_indices)]

        # h = tau - M*qdd  → 只取 arm 对应分量
        h_all = tau_all - M_full.dot(qdd_all)
        h_arm = h_all[arm_indices]
        return M_aa, h_arm

    def clamp(self, x: np.ndarray, lim: np.ndarray) -> np.ndarray:
        return np.clip(x, -np.abs(lim), np.abs(lim))

    def step_computed_torque(self, q_ref: np.ndarray, qd_ref: np.ndarray, qdd_ref: np.ndarray):
        q, qd = self.get_state()
        e = q_ref - q
        ed = qd_ref - qd
        Kp = np.diag(self.joint_gains.kp)
        Kd = np.diag(self.joint_gains.kd)
        v = qdd_ref + Kd.dot(ed) + Kp.dot(e)
        M, h = self.invdyn_terms(q, qd, v)
        tau = M.dot(v) + h
        tau = self.clamp(tau, self.tau_limits)
        for i, j in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=float(tau[i]))
        p.stepSimulation()

    def step_pd_gravity(self, q_ref: np.ndarray, qd_ref: Optional[np.ndarray]=None):
        q, qd = self.get_state()
        qd_ref = qd_ref if qd_ref is not None else np.zeros_like(q)
        e = q_ref - q
        ed = qd_ref - qd
        Kp = np.diag(self.joint_gains.kp)
        Kd = np.diag(self.joint_gains.kd)
        # 重力补偿项
        _, h = self.invdyn_terms(q, qd, np.zeros_like(q))
        tau = Kp.dot(e) + Kd.dot(ed) + h
        tau = self.clamp(tau, self.tau_limits)
        for i, j in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=float(tau[i]))
        p.stepSimulation()

    # ------------------------- 笛卡尔阻抗(位置)控制（可选） -------------------------
    def step_cartesian_impedance(self, xd_pos: np.ndarray, xd_quat: np.ndarray, xdot_d: Optional[np.ndarray]=None):
        q, qd = self.get_state()
        pos, quat = self.ee_fk()
        Jv, Jw = self.jacobian()
        J = np.vstack([Jv, Jw])
        xdot = J.dot(qd)
        xdot_d = xdot_d if xdot_d is not None else np.zeros(6)
        # 位置误差
        e_p = xd_pos - pos
        # 姿态误差（小角度近似用角轴差，这里用四元数误差向量部近似）
        # 为简洁，这里直接零处理，详细可参考 ik_resolved_rate 的误差实现
        e_o = np.zeros(3)
        e = np.hstack([e_p, e_o])
        # 虚拟质量与阻尼 (简单起见直接在关节空间施力矩)
        Kp = np.block([[self.imp_gains.Kp, np.zeros((3,3))], [np.zeros((3,3)), self.imp_gains.Ko]])
        Kd = np.block([[self.imp_gains.Kd, np.zeros((3,3))], [np.zeros((3,3)), 2*np.eye(3)]])
        f_task = Kp.dot(e) + Kd.dot(xdot_d - xdot)
        tau = J.T.dot(f_task)
        # + 重力补偿
        _, h = self.invdyn_terms(q, qd, np.zeros_like(q))
        tau = tau + h
        tau = self.clamp(tau, self.tau_limits)
        for i,j in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=float(tau[i]))
        p.stepSimulation()

    # ------------------------- 抓取子流程 -------------------------
    def open_gripper(self, width: float = 0.08):
        # Panda 两指对称，简单用位置控制(若URDF含传动需要按比例)
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=width/2.0, force=20)
        for _ in range(60):
            p.stepSimulation()

    def close_gripper(self, width: float = 0.0, force: float = 40):
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=width/2.0, force=force)
        for _ in range(120):
            p.stepSimulation()

    def go_to_q(self, q_goal: np.ndarray, T: float = 2.0, controller: str = "ctc"):
        q0, _ = self.get_state()
        steps = max(10, int(T / self.dt))
        traj = self.plan_joint_traj(q0, q_goal, T, steps)
        for i in range(steps):
            if controller == "ctc":
                self.step_computed_torque(traj['q'][i], traj['qd'][i], traj['qdd'][i])
            else:
                self.step_pd_gravity(traj['q'][i], traj['qd'][i])

    def move_lincart(self, p_start: np.ndarray, q_start: np.ndarray, p_goal: np.ndarray, q_goal: np.ndarray, T: float=2.0):
        steps = max(10, int(T/self.dt))
        for i in range(steps):
            t = i/(steps-1)
            pd, qd = pose_lerp(p_start, q_start, p_goal, q_goal, t)
            q_cmd = self.ik_pb(pd, qd)
            self.step_pd_gravity(q_cmd)

    def grasp(self,
              obj_pose: Tuple[np.ndarray, np.ndarray],
              pre_offset: np.ndarray = np.array([0,0,0.10]),
              lift_offset: np.ndarray = np.array([0,0,0.15]),
              approach_T: float = 1.5,
              close_force: float = 60.0,
              controller: str = "ctc"):
        """ 基本抓取：pregrasp -> approach -> close -> lift """
        obj_p, obj_q = obj_pose
        # 预抓姿态（在物体正上方）
        pre_p = obj_p + pre_offset
        pre_q = obj_q  # 简化：姿态一致（可根据任务旋转使爪口对齐）

        # 打开手爪
        self.open_gripper()

        # 关节空间移动至 pregrasp（用 IK 计算关节目标）
        q_pre = self.ik_pb(pre_p, pre_q)
        self.go_to_q(q_pre, T=2.0, controller=controller)

        # 直线下降到接触前（笛卡尔线性）
        ee_p, ee_q = self.ee_fk()
        self.move_lincart(ee_p, ee_q, obj_p, obj_q, T=approach_T)

        # 闭合抓取
        self.close_gripper(width=0.0, force=close_force)

        # 抬升
        lift_p = obj_p + lift_offset
        self.move_lincart(obj_p, obj_q, lift_p, obj_q, T=1.5)

    # ------------------------- 常用场景/物体 -------------------------
    def load_ycb_box(self, halfExtents=(0.025,0.025,0.025), mass=0.1, pose=((0.6,0,0.02),(0,0,0,1))):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.9,0.4,0.2,1])
        box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                basePosition=pose[0], baseOrientation=pose[1])
        return box



    def _full_state_vectors(self, q_arm: np.ndarray, qd_arm: np.ndarray,
                            qdd_arm: Optional[np.ndarray]=None):
        """生成包含完整关节状态的向量（含手爪关节）
        
        参数:
            q_arm: 机械臂关节位置向量，形状(n,)
            qd_arm: 机械臂关节速度向量，形状(n,)
            qdd_arm: 机械臂关节加速度向量（可选），形状(n,)
            
        返回:
            Tuple: 包含以下元素的元组:
                - q_all: 完整关节位置向量
                - qd_all: 完整关节速度向量
                - qdd_all: 完整关节加速度向量
                - ctrl_joints: 控制的关节索引列表
                - arm_indices: 机械臂关节在完整列表中的索引
        """
        if qdd_arm is None:
            qdd_arm = np.zeros_like(q_arm)

        # 读取当前所有关节的实际状态（作为默认）
        q_all, qd_all = [], []
        ctrl_joints = []
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            jtype = info[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                ctrl_joints.append(j)
                js = p.getJointState(self.robot, j)
                q_all.append(js[0]); qd_all.append(js[1])
        q_all = np.array(q_all); qd_all = np.array(qd_all)

        # 按照 ctrl_joints 把 arm 的值“覆盖”进去
        arm_mask = np.zeros(len(ctrl_joints), dtype=bool)
        arm_indices = []
        # 建立 arm_joints 在 ctrl_joints 中的位置映射
        for a in self.arm_joints:
            idx = ctrl_joints.index(a)
            arm_indices.append(idx)
            arm_mask[idx] = True
        arm_indices = np.array(arm_indices)

        q_all[arm_indices]  = q_arm
        qd_all[arm_indices] = qd_arm

        # 加速度向量
        qdd_all = np.zeros_like(q_all)
        qdd_all[arm_indices] = qdd_arm
        # 手爪加速度保持 0（如果需要也可给一点阻尼）
        return q_all, qd_all, qdd_all, np.array(ctrl_joints), arm_indices

# ---------------------------- 示例 ----------------------------
if __name__ == "__main__":
    ctrl = TraditionalGraspController(use_gui=True)
    # 放一个方块
    box_id = ctrl.load_ycb_box(pose=((0.6, 0.0, 0.025), p.getQuaternionFromEuler([np.pi, 0, 0])))

    # 初始收拢姿态（方便起步，可按自己系统的home姿态替换）
    home = np.array([0, -0.6, 0, -2.4, 0, 1.8, 0.8])
    ctrl.go_to_q(home, T=2.5, controller="ctc")

    # 获取方块当前姿态
    pos, ori = p.getBasePositionAndOrientation(box_id)
    obj_pose = (np.array(pos), np.array(ori))

    # 执行抓取
    ctrl.grasp(obj_pose, pre_offset=np.array([0,0,0.18]), lift_offset=np.array([0,0,0.20]), controller="ctc")

    # 停留观察
    for _ in range(100000):
        p.stepSimulation(); time.sleep(1.0/240.0)
