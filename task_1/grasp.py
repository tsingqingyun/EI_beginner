import pybullet as p
import pybullet_data
import time
import numpy as np  # 用于矩阵运算和坐标变换

# 步骤1: 初始化仿真环境
p.connect(p.GUI)  # GUI 模式，便于可视化
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)  # 设置重力
p.setTimeStep(1./240.)  # 时间步长：每次仿真迭代所代表的实际时间长度，影响仿真的精度和稳定性

# 步骤2: 加载机械臂和物体
# 加载地面
planeId = p.loadURDF("plane.urdf")

# 加载 6-DOF KUKA iiwa 机械臂
armStartPos = [0, 0, 0]
armStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
armId = p.loadURDF("kuka_iiwa/model.urdf", armStartPos, armStartOrientation)

# 获取关节信息（假设末端执行器链接索引为 6）
numJoints = p.getNumJoints(armId)
endEffectorIndex = 6  # KUKA iiwa 的末端链接link6

# 加载物体：一个立方体作为抓取目标
boxPos = [0.5, 0.5, 0.5]  # 物体初始位置
boxId = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.04]),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1]),
    basePosition=boxPos
)

# 步骤3: 基础坐标变换示例
# 获取机械臂基坐标到世界坐标的变换（这里基坐标即世界坐标）
basePos, baseOrn = p.getBasePositionAndOrientation(armId)
print("基坐标位置:", basePos, "姿态:", baseOrn)

# 转换欧拉角到四元数(避免万向锁)
targetEuler = [0, np.pi/4, np.pi/4]  # 目标姿态（欧拉角）

targetQuat = p.getQuaternionFromEuler(targetEuler)

# 步骤4: 正向运动学 (FK) 示例
# 设置关节角度，计算末端位置
jointAngles = [0, 0, 0, np.pi/2, 0, np.pi/4, 0]  # 示例关节角度
for i in range(numJoints):
    p.resetJointState(armId, i, jointAngles[i])

# 获取末端执行器状态（FK 输出）
linkState = p.getLinkState(armId, endEffectorIndex)
endPos = linkState[4]  # 世界位置
endOrn = linkState[5]  # 世界姿态（四元数）
print("FK 计算末端位置:", endPos, "姿态:", endOrn)

# 步骤5: 逆向运动学 (IK) 和轨迹规划
# 定义抓取目标：物体上方位置
targetPos = [boxPos[0], boxPos[1], boxPos[2] + 0.2]  # 物体上方 20cm
targetOrn = p.getQuaternionFromEuler([0, np.pi, 0])  # 向下姿态（抓取）

# 使用 PyBullet 的 IK 求解器计算关节角度
ikJointAngles = p.calculateInverseKinematics(
    armId, endEffectorIndex, targetPos, targetOrientation=targetOrn,
    jointDamping=[0.01] * numJoints  # 阻尼参数，提高求解稳定性
)
print("IK 计算关节角度:", ikJointAngles)

# 简单线性轨迹规划：从当前到目标，插值 100 步
currentJointStates = [p.getJointState(armId, i)[0] for i in range(numJoints)]
trajectory = []
for t in np.linspace(0, 1, 100):
    interpAngles = (1 - t) * np.array(currentJointStates) + t * np.array(ikJointAngles)
    trajectory.append(interpAngles)

# 步骤6: 动力学和控制理论（PID 控制）
# PID 参数（经验值，可调优）
Kp = 200.0  # 比例增益
Ki = 0.0    # 积分增益（避免积分饱和）
Kd = 50.0   # 微分增益

# 控制循环：跟踪轨迹，实现抓取
for step in trajectory:
    for i in range(numJoints):
        # 获取当前关节状态
        jointState = p.getJointState(armId, i)
        currentPos = jointState[0]
        currentVel = jointState[1]
        
        # 计算误差
        error = step[i] - currentPos
        error_deriv = -currentVel  # 近似微分
        
        # PID 计算控制扭矩（动力学输入）
        torque = Kp * error + Kd * error_deriv  # 简化无积分
        
        # 施加扭矩（考虑动力学：质量、摩擦等由 PyBullet 处理）
        p.setJointMotorControl2(
            armId, i, p.TORQUE_CONTROL, force=torque
        )
    
    p.stepSimulation()
    time.sleep(1./240.)  # 同步实时

# 步骤7: 抓取逻辑（简单夹持模拟）
# 移动到物体位置后，模拟抓取：创建约束
graspConstraint = p.createConstraint(
    armId, endEffectorIndex, boxId, -1,  # -1 表示物体基链接
    p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, -0.1]  # 相对位置偏移
)
print("物体已抓取")

# 保持抓取并移动（示例：抬起）
for _ in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

# 步骤8: 释放物体并关闭
p.removeConstraint(graspConstraint)
p.disconnect()