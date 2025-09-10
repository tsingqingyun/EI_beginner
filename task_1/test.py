import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pybullet as p
import pybullet_data


@dataclass
class PDGains:
    kp: float = 120.0
    kd: float = 6.0
    gravity_comp: bool = True  # 仅在 --mode pd 时生效


@dataclass
class GraspPlan:
    approach_height: float = 0.18     # 物体上方预抓高度
    descend_clearance: float = 0.012  # 下降到物体上表面的余量
    lift_height: float = 0.18         # 抬起高度
    place_offset: Tuple[float, float] = (0.20, -0.20)  # 相对放置位移 (x,y)
    gripper_open: float = 0.04        # Panda 夹爪最大开度（单指位移）
    gripper_close: float = 0.0        # 夹爪闭合目标


class PandaGrasper:
    def __init__(self, mode: str = "ik", gains: PDGains = PDGains(), record_path: Optional[str] = None):
        assert mode in ("ik", "pd")
        self.mode = mode
        self.gains = gains
        self.record_path = record_path
        self.log_id = None

        # 连接 GUI, 配置可视化
        self.cid = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(
            cameraDistance=1.6, cameraYaw=50, cameraPitch=-35,
            cameraTargetPosition=[0.5, 0.0, 0.15]
        )
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        # 物理参数
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(
            numSolverIterations=150, fixedTimeStep=1.0 / 240.0
        )

        # 若指定录制，启动视频记录（MP4）
        if self.record_path:
            self.log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.record_path)

        # 环境
        self.plane = p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.62])
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # 目标物体（小方块）
        self.cube = p.loadURDF("cube_small.urdf", basePosition=[0.55, 0.0, 0.02])
        # 提高摩擦，利于夹持
        p.changeDynamics(self.cube, -1, lateralFriction=1.2, rollingFriction=0.001, spinningFriction=0.001)

        # 关节和末端信息（通过名称解析，避免硬编码）
        self.arm_joints, self.finger_joints, self.ee_link = self._introspect_panda()

        # 初始姿态：一个安全的“准备抓取”关节配置
        home = [0.0, -0.6, 0.0, -1.8, 0.0, 1.8, 0.8]
        self._goto_joint_positions_posctl(home, steps=240)
        self._set_gripper(width=0.04)  # 张开夹爪

        # 持续渲染一点 HUD
        p.addUserDebugText(f"Mode: {self.mode.upper()} (press Ctrl+C to quit)",
                           [0.05, -0.5, 0.5], textColorRGB=[0.9, 0.9, 0.9], textSize=1.3, lifeTime=0)

    # ---------- 机器人模型解析 ----------
    def _introspect_panda(self) -> Tuple[List[int], List[int], int]:
        n = p.getNumJoints(self.robot)
        arm, fingers, ee = [], [], 11  # ee 默认 11（Panda 手爪基座），与常用示例一致
        for j in range(n):
            ji = p.getJointInfo(self.robot, j)
            jname = ji[1].decode("utf-8")
            linkname = ji[12].decode("utf-8")
            jtype = ji[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if jname.startswith("panda_joint") and jname[-1].isdigit() and int(jname[-1]) <= 7:
                    arm.append(j)
                if jname in ("panda_finger_joint1", "panda_finger_joint2"):
                    fingers.append(j)
            if linkname == "panda_hand":
                ee = j  # 以连到手爪的关节索引作为末端 link index
        arm.sort()
        fingers.sort()
        return arm, fingers, ee

    # ---------- 常用工具 ----------
    @staticmethod
    def _s_curve(alpha: float) -> float:
        # 平滑插值：3t^2 - 2t^3
        return 3 * alpha ** 2 - 2 * alpha ** 3

    @staticmethod
    def _quat_down():
        # 末端朝下的姿态（绕Y翻转pi），常见抓取姿态
        return p.getQuaternionFromEuler([0.0, -math.pi, 0.0])

    def _step(self, steps: int = 1):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    # ---------- 关节空间控制 ----------
    def _goto_joint_positions_posctl(self, q_target: List[float], steps: int = 480):
        """利用 POSITION_CONTROL 伺服到目标关节角（内部 PD），视觉效果平滑"""
        assert len(q_target) == len(self.arm_joints)
        forces = [200.0] * len(self.arm_joints)
        for t in range(steps):
            p.setJointMotorControlArray(
                self.robot, self.arm_joints, p.POSITION_CONTROL,
                targetPositions=q_target, forces=forces
            )
            self._step()

    def _goto_joint_positions_torque_pd(self, q_target: List[float], steps: int = 480):
        """手写扭矩 PD（可选重力补偿），更贴近“传统低层伺服”"""
        assert len(q_target) == len(self.arm_joints)
        # 关闭默认电机
        for j in self.arm_joints + self.finger_joints:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0.0)

        for _ in range(steps):
            q = []
            dq = []
            for j in self.arm_joints:
                js = p.getJointState(self.robot, j)
                q.append(js[0])
                dq.append(js[1])

            # PD 力矩
            tau = [self.gains.kp * (qd - qi) - self.gains.kd * dqi
                   for qd, qi, dqi in zip(q_target, q, dq)]

            # 重力补偿（只需在扭矩模式下考虑）
            if self.gains.gravity_comp:
                g = p.calculateInverseDynamics(self.robot, q, [0.0] * len(q), [0.0] * len(q))
                tau = [ti + gi for ti, gi in zip(tau, g)]

            # 施加扭矩
            for j, tj in zip(self.arm_joints, tau):
                p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=tj)

            self._step()

    # ---------- 末端直线轨迹（笛卡尔空间 -> IK -> 关节） ----------
    def _ee_line_to(self, pos_start, pos_end, orn=None, duration=1.5, steps=360):
        if orn is None:
            orn = self._quat_down()

        # 画个目标路径，增强可视化
        p.addUserDebugLine(pos_start, pos_end, [0, 1, 0], lineWidth=1.5, lifeTime=5)

        joint_targets_series = []
        for i in range(steps):
            a = self._s_curve(i / (steps - 1))
            px = pos_start[0] + (pos_end[0] - pos_start[0]) * a
            py = pos_start[1] + (pos_end[1] - pos_start[1]) * a
            pz = pos_start[2] + (pos_end[2] - pos_start[2]) * a

            sol = p.calculateInverseKinematics(
                self.robot, self.ee_link, [px, py, pz], orn,
                maxNumIterations=200, residualThreshold=1e-4
            )
            # 只取前 7 个臂关节
            joint_targets_series.append(list(sol[:len(self.arm_joints)]))

        # 执行（根据模式选择控制器）
        per_step = max(1, int(duration / (steps * (1.0 / 240.0))))
        for q_tar in joint_targets_series:
            if self.mode == "ik":
                self._goto_joint_positions_posctl(q_tar, steps=per_step)
            else:
                self._goto_joint_positions_torque_pd(q_tar, steps=per_step)

    # ---------- 夹爪 ----------
    def _set_gripper(self, width: float, force: float = 40.0, steps: int = 240):
        # Panda 夹爪是两指等距开合，单指目标位移约等于 width
        target = max(0.0, min(0.04, width))
        for _ in range(steps):
            p.setJointMotorControl2(self.robot, self.finger_joints[0], p.POSITION_CONTROL,
                                    targetPosition=target, force=force)
            # 部分 URDF 会配置 mimic，这里同步设置两指更稳妥
            if len(self.finger_joints) > 1:
                p.setJointMotorControl2(self.robot, self.finger_joints[1], p.POSITION_CONTROL,
                                        targetPosition=target, force=force)
            self._step()

    def _close_gripper_until_contact(self, max_steps=360, force=80.0):
        """逐步闭合，检测与方块接触；接触后再略微加力以稳定抓取"""
        width = 0.028  # 从半开开始
        for _ in range(max_steps):
            self._set_gripper(width, force=force, steps=12)
            cps = p.getContactPoints(self.robot, self.cube)
            if len(cps) > 0:
                # 夹持到位后再小幅闭合
                self._set_gripper(0.0, force=force, steps=120)
                return True
            width -= 0.003
            if width <= 0.0:
                break
        return False

    # ---------- 主流程 ----------
    def run(self, plan: GraspPlan = GraspPlan()):
        # 读取目标物体位姿（在本教学中相当于“感知结果”）
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
        cube_z_top = cube_pos[2] + 0.02  # cube_small 高约 0.04，顶面 z

        # 目标姿态：末端朝下
        ee_orn = self._quat_down()

        # 关键位姿（世界坐标）
        pre_grasp = [cube_pos[0], cube_pos[1], cube_z_top + plan.approach_height]
        grasp_pos = [cube_pos[0], cube_pos[1], cube_z_top + plan.descend_clearance]
        lift_pos = [cube_pos[0], cube_pos[1], cube_z_top + plan.lift_height]
        place_pos = [cube_pos[0] + plan.place_offset[0],
                     cube_pos[1] + plan.place_offset[1],
                     cube_z_top + plan.lift_height]

        # 可视化目标帧
        self._draw_frame(pre_grasp)
        self._draw_frame(grasp_pos)
        self._draw_frame(place_pos)

        # 动作序列
        self._set_gripper(plan.gripper_open)                        # 张开
        ee_now = p.getLinkState(self.robot, self.ee_link)[0]

        # 1) 移动到物体上方
        self._ee_line_to(ee_now, pre_grasp, ee_orn, duration=1.6, steps=360)

        # 2) 垂直下降到抓取高度
        self._ee_line_to(pre_grasp, grasp_pos, ee_orn, duration=1.2, steps=300)

        # 3) 闭合夹爪（带接触检测）
        ok = self._close_gripper_until_contact()
        if not ok:
            print("[WARN] 未检测到稳定接触，尝试直接闭合")
            self._set_gripper(plan.gripper_close, steps=240)

        # 4) 垂直抬起
        self._ee_line_to(grasp_pos, lift_pos, ee_orn, duration=1.2, steps=300)

        # 5) 平移到放置上方
        hover_place = [place_pos[0], place_pos[1], lift_pos[2]]
        self._ee_line_to(lift_pos, hover_place, ee_orn, duration=1.6, steps=360)

        # 6) 下降、松开、回撤
        down_place = [place_pos[0], place_pos[1], grasp_pos[2]]
        self._ee_line_to(hover_place, down_place, ee_orn, duration=1.2, steps=300)
        self._set_gripper(plan.gripper_open, steps=240)
        self._ee_line_to(down_place, hover_place, ee_orn, duration=1.0, steps=240)

        print("[INFO] 任务完成，窗口可继续观察；按 Ctrl+C 结束。")
        # 若在录制，任务完成后立即停止录制并提示保存位置
        if self.log_id is not None:
            p.stopStateLogging(self.log_id)
            print(f"[INFO] 视频已保存到: {self.record_path}")
        while True:
            self._step()

    # ---------- 可视化辅助 ----------
    def _draw_frame(self, pos, size: float = 0.06, life: float = 10.0):
        x = [pos[0] + size, pos[1], pos[2]]
        y = [pos[0], pos[1] + size, pos[2]]
        z = [pos[0], pos[1], pos[2] + size]
        p.addUserDebugLine(pos, x, [1, 0, 0], 2, lifeTime=life)
        p.addUserDebugLine(pos, y, [0, 1, 0], 2, lifeTime=life)
        p.addUserDebugLine(pos, z, [0, 0, 1], 2, lifeTime=life)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ik", "pd"], default="ik",
                    help="ik: IK + 位置伺服（视觉更平滑）; pd: 扭矩PD + 可选重力补偿")
    ap.add_argument("--record", type=str, default=None,
                    help="将 GUI 视图录制为 MP4 文件，例如: --record out.mp4")
    args = ap.parse_args()

    # 适度的增益，PD 模式更“硬”；IK 模式该参数不影响
    gains = PDGains(kp=120.0, kd=6.0, gravity_comp=True if args.mode == "pd" else False)
    grasper = PandaGrasper(mode=args.mode, gains=gains, record_path=args.record)
    grasper.run(GraspPlan())


if __name__ == "__main__":
    main()
