import pybullet as p
import pybullet_data
import time
import numpy as np

# 步骤1: 初始化仿真环境
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1./240.)

# 步骤2: 加载机械臂、地面和多个物体
planeId = p.loadURDF("plane.urdf")
armStartPos = [0, 0, 0]
armStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
armId = p.loadURDF("kuka_iiwa/model.urdf", armStartPos, armStartOrientation)
numJoints = p.getNumJoints(armId)
endEffectorIndex = 6

# 加载多个立方体
object_positions = [[0.5, 0.5, 0.5], [0.7, 0.5, 0.5], [0.5, 0.7, 0.5]]
object_ids = []
for pos in object_positions:
    obj_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1]),
        basePosition=pos
    )
    object_ids.append(obj_id)

# 步骤3: 视觉引导 - 设置相机
width, height = 640, 480
fov = 60
aspect = width / height
near, far = 0.1, 10.0
projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[1, 1, 2],  # 相机位置
    cameraTargetPosition=[0.5, 0.5, 0],  # 观察点
    cameraUpVector=[0, 0, 1]  # 上向量：相机视角下“上”在哪里
)

# 获取相机图像，检测物体位置
def detect_objects():
    _, _, rgb, depth, seg = p.getCameraImage(
        width, height, viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    object_centers = []
    for obj_id in object_ids:
        # 提取分割图像中对应物体的像素
        seg_mask = (seg == obj_id)
        if np.sum(seg_mask) > 0:  # 确保物体可见
            # 计算像素中心（平均坐标）
            y, x = np.where(seg_mask)
            u, v = np.mean(x), np.mean(y)
            # 获取深度值
            z = depth[int(v), int(u)] * (far - near) + near
            # 逆投影：像素到世界坐标
            uv_hom = np.array([u, v, 1])
            K = np.array(projectionMatrix).reshape(4, 4)[:3, :3]
            K_inv = np.linalg.inv(K)
            R_t = np.array(viewMatrix).reshape(4, 4)
            world_hom = R_t @ np.append(K_inv @ uv_hom * z, 1)
            world_pos = world_hom[:3] / world_hom[3]
            object_centers.append(world_pos)
        else:
            object_centers.append(None)  # 物体不可见
    return object_centers

# 步骤4: 多物体抓取 - 轨迹规划和控制
Kp, Ki, Kd = 200.0, 0.0, 50.0
place_position = [0.3, 0.3, 0.5]  # 放置点

for obj_idx, obj_id in enumerate(object_ids):
    # 检测物体位置
    object_centers = detect_objects()
    target_pos = object_centers[obj_idx]
    if target_pos is None:
        print(f"物体 {obj_id} 不可见，跳过")
        continue

    # 计算抓取位置（物体上方 20cm）
    grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.2]
    target_orn = p.getQuaternionFromEuler([0, np.pi, 0])

    # 逆向运动学
    ik_joint_angles = p.calculateInverseKinematics(
        armId, endEffectorIndex, grasp_pos, targetOrientation=target_orn,
        jointDamping=[0.01] * numJoints
    )

    # 轨迹规划
    current_joint_states = [p.getJointState(armId, i)[0] for i in range(numJoints)]
    trajectory = []
    for t in np.linspace(0, 1, 100):
        interp_angles = (1 - t) * np.array(current_joint_states) + t * np.array(ik_joint_angles)
        trajectory.append(interp_angles)

    # PID 控制跟踪轨迹
    for step in trajectory:
        for i in range(numJoints):
            joint_state = p.getJointState(armId, i)
            current_pos, current_vel = joint_state[0], joint_state[1]
            error = step[i] - current_pos
            error_deriv = -current_vel
            torque = Kp * error + Kd * error_deriv
            p.setJointMotorControl2(armId, i, p.TORQUE_CONTROL, force=torque)
        p.stepSimulation()
        time.sleep(1./240.)

    # 抓取物体
    grasp_constraint = p.createConstraint(
        armId, endEffectorIndex, obj_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, -0.1]
    )

    # 移动到放置点
    place_pos = [place_position[0], place_position[1], place_position[2] + 0.2]
    ik_joint_angles = p.calculateInverseKinematics(
        armId, endEffectorIndex, place_pos, targetOrientation=target_orn,
        jointDamping=[0.01] * numJoints
    )
    trajectory = []
    current_joint_states = [p.getJointState(armId, i)[0] for i in range(numJoints)]
    for t in np.linspace(0, 1, 100):
        interp_angles = (1 - t) * np.array(current_joint_states) + t * np.array(ik_joint_angles)
        trajectory.append(interp_angles)

    for step in trajectory:
        for i in range(numJoints):
            joint_state = p.getJointState(armId, i)
            current_pos, current_vel = joint_state[0], joint_state[1]
            error = step[i] - current_pos
            error_deriv = -current_vel
            torque = Kp * error + Kd * error_deriv
            p.setJointMotorControl2(armId, i, p.TORQUE_CONTROL, force=torque)
        p.stepSimulation()
        time.sleep(1./240.)

    # 释放物体
    p.removeConstraint(grasp_constraint)
    time.sleep(1)  # 等待物体稳定

# 步骤5: 关闭仿真
p.disconnect()