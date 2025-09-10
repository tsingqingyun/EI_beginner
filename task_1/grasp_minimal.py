# -*- coding: utf-8 -*-
"""
grasp_minimal_posctl.py
模仿 test 的处理：IK + POSITION_CONTROL（不使用扭矩PD）。
抓取序列：pre_grasp -> descend -> close(with contact) -> lift -> place

Run:
  python grasp_minimal_posctl.py
"""
import math
import time
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data


def s_curve(alpha: float) -> float:
    """平滑插值函数：实现3次多项式插值，生成S形速度曲线
    参数:
        alpha - 插值系数(0~1)
    返回: 平滑后的插值结果"""
    a = max(0.0, min(1.0, alpha))  # 确保alpha在0~1范围内
    return 3 * a * a - 2 * a * a * a  # 3次多项式公式


def quat_down():
    """生成末端执行器掌心向下的四元数姿态
    返回: 表示姿态的四元数"""
    return p.getQuaternionFromEuler([0.0, -math.pi, 0.0])  # 绕Y轴旋转180度的欧拉角转换为四元数


class PandaPosctl:
    def __init__(self, dt: float = 1.0 / 240.0):
        self.dt = dt
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(1.6, 50, -35, [0.5, 0.0, 0.15])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # 物理
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep=self.dt)

        # 场景
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.62])
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # 物体：小方块（提高摩擦，利于夹持）
        self.cube = p.loadURDF("cube_small.urdf", basePosition=[0.55, 0.0, 0.02])
        p.changeDynamics(self.cube, -1,
                         lateralFriction=1.2, rollingFriction=0.001, spinningFriction=0.001)

        # 关节索引
        self.arm_joints, self.finger_joints, self.ee_link = self._introspect_panda()

        # 初始位姿 & 夹爪张开
        home = [0.0, -0.6, 0.0, -1.8, 0.0, 1.8, 0.8]
        self._goto_joint_positions_posctl(home, steps=240)
        self.set_gripper(0.04, steps=180)  # 单指 0.04 ≈ 双指 0.08

    # ---------------- 基本工具 ----------------
    def _introspect_panda(self) -> Tuple[List[int], List[int], int]:
        n = p.getNumJoints(self.robot)
        arm, fingers, ee = [], [], 11
        for j in range(n):
            info = p.getJointInfo(self.robot, j)
            jname = info[1].decode()
            linkname = info[12].decode()
            jtype = info[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if jname.startswith("panda_joint") and jname[-1].isdigit() and int(jname[-1]) <= 7:
                    arm.append(j)
                if jname in ("panda_finger_joint1", "panda_finger_joint2"):
                    fingers.append(j)
            if linkname == "panda_hand":
                ee = j
        arm.sort()
        fingers.sort()
        return arm, fingers, ee

    def _step(self, steps: int = 1):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(self.dt)

    # ---------------- 位置控制（内置PD） ----------------
    def _goto_joint_positions_posctl(self, q_target: List[float], steps: int = 480, force: float = 200.0):
        assert len(q_target) == len(self.arm_joints)
        forces = [force] * len(self.arm_joints)
        for _ in range(steps):
            p.setJointMotorControlArray(self.robot, self.arm_joints, p.POSITION_CONTROL,
                                        targetPositions=q_target, forces=forces)
            self._step()

    # ---------------- 末端直线：笛卡尔 -> IK -> 位置控制 ----------------
    def ee_line(self, pos_start, pos_end, orn=None, duration=1.5, samples=360):
        if orn is None:
            orn = quat_down()
        p.addUserDebugLine(pos_start, pos_end, [0, 1, 0], lineWidth=1.5, lifeTime=5)

        per = max(1, int(duration / (samples * self.dt)))
        for i in range(samples):
            a = s_curve(i / (samples - 1))
            px = pos_start[0] + (pos_end[0] - pos_start[0]) * a
            py = pos_start[1] + (pos_end[1] - pos_start[1]) * a
            pz = pos_start[2] + (pos_end[2] - pos_start[2]) * a

            sol = p.calculateInverseKinematics(self.robot, self.ee_link, [px, py, pz], orn,
                                               maxNumIterations=200, residualThreshold=1e-4)
            q_tar = list(sol[:len(self.arm_joints)])
            self._goto_joint_positions_posctl(q_tar, steps=per)

    # ---------------- 夹爪（POSITION_CONTROL） ----------------
    def set_gripper(self, single_finger: float, force: float = 60.0, steps: int = 120):
        """单指位移（0~0.04），两指会同步设置"""
        target = max(0.0, min(0.04, single_finger))
        for _ in range(steps):
            for j in self.finger_joints:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                        targetPosition=target, force=force)
            self._step()

    def close_until_contact(self, obj_id: int, start=0.028, step=0.003, force=80.0, max_steps=360) -> bool:
        """逐步闭合并检测接触。接触后再小幅闭合定型。"""
        width = float(start)
        for _ in range(max_steps):
            self.set_gripper(width, force=force, steps=12)
            if p.getContactPoints(self.robot, obj_id):
                self.set_gripper(0.0, force=force, steps=120)
                return True
            width -= step
            if width <= 0.0:
                break
        return False

    # ---------------- 抓取流程 ----------------
    def run(self):
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
        cube_z_top = cube_pos[2] + 0.02  # cube_small 高约 0.04m，顶面 z
        ee_orn = quat_down()

        pre_grasp = [cube_pos[0], cube_pos[1], cube_z_top + 0.18]
        grasp_pos = [cube_pos[0], cube_pos[1], cube_z_top + 0.012]
        lift_pos  = [cube_pos[0], cube_pos[1], cube_z_top + 0.18]
        place_pos = [cube_pos[0] + 0.20, cube_pos[1] - 0.20, cube_z_top + 0.18]

        self._draw_frame(pre_grasp)
        self._draw_frame(grasp_pos)
        self._draw_frame(place_pos)

        # 张开 → 上方 → 下探 → 闭合(接触感知) → 抬起 → 平移放置上方 → 下降松开 → 回撤
        self.set_gripper(0.04, steps=180)

        ee_now = p.getLinkState(self.robot, self.ee_link)[0]
        self.ee_line(ee_now, pre_grasp, ee_orn, duration=1.6, samples=360)
        self.ee_line(pre_grasp, grasp_pos, ee_orn, duration=1.2, samples=300)

        ok = self.close_until_contact(self.cube, start=0.028, step=0.003, force=100.0)
        if not ok:
            print("[WARN] 未检测到稳定接触，尝试直接闭合")
            self.set_gripper(0.0, force=100.0, steps=240)

        self.ee_line(grasp_pos, lift_pos, ee_orn, duration=1.2, samples=300)
        hover_place = [place_pos[0], place_pos[1], lift_pos[2]]
        self.ee_line(lift_pos, hover_place, ee_orn, duration=1.6, samples=360)

        down_place = [place_pos[0], place_pos[1], grasp_pos[2]]
        self.ee_line(hover_place, down_place, ee_orn, duration=1.2, samples=300)
        self.set_gripper(0.04, steps=180)
        self.ee_line(down_place, hover_place, ee_orn, duration=1.0, samples=240)

        print("[INFO] 任务完成，可继续观察窗口，关闭窗口即可退出。")
        # 只在连接仍存在时继续渲染，避免 Not connected to physics server
        while p.isConnected():
            try:
                p.stepSimulation()
                time.sleep(self.dt)
            except Exception:
                break

    # ---------------- 可视化 ----------------
    def _draw_frame(self, pos, size: float = 0.06, life: float = 8.0):
        x = [pos[0] + size, pos[1], pos[2]]
        y = [pos[0], pos[1] + size, pos[2]]
        z = [pos[0], pos[1], pos[2] + size]
        p.addUserDebugLine(pos, x, [1, 0, 0], 2, lifeTime=life)
        p.addUserDebugLine(pos, y, [0, 1, 0], 2, lifeTime=life)
        p.addUserDebugLine(pos, z, [0, 0, 1], 2, lifeTime=life)


if __name__ == "__main__":
    PandaPosctl().run()
