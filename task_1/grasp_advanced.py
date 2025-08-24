import pybullet as p
import pybullet_data
import time
import numpy as np

def init_simulation(gravity=(0, 0, -9.81), time_step=1./240.):
    """
    初始化仿真环境。
    返回：仿真客户端 ID。
    """
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*gravity)
    p.setTimeStep(time_step)
    return client_id

def load_ground():
    """
    加载地面模型。
    返回：地面 ID。
    """
    return p.loadURDF("plane.urdf")

def load_arm(base_pos=[0, 0, 0], base_orn=[0, 0, 0]):
    """
    加载机械臂模型。
    返回：臂 ID, 关节数, 末端执行器索引。
    """
    base_orientation = p.getQuaternionFromEuler(base_orn)
    arm_id = p.loadURDF("kuka_iiwa/model.urdf", base_pos, base_orientation)
    num_joints = p.getNumJoints(arm_id)
    end_effector_index = 6
    return arm_id, num_joints, end_effector_index

def load_objects(positions, mass=1.0, half_extents=[0.05, 0.05, 0.05], color=[1, 0, 0, 1]):
    """
    加载多个物体。
    返回：物体 ID 列表。
    """
    object_ids = []
    for pos in positions:
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        obj_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape,
                                   baseVisualShapeIndex=visual_shape, basePosition=pos)
        object_ids.append(obj_id)
    return object_ids

def setup_camera(width=640, height=480, fov=60, near=0.1, far=10.0,
                 eye_pos=[1, 1, 2], target_pos=[0.5, 0.5, 0], up_vector=[0, 0, 1]):
    """
    设置相机参数。
    返回：投影矩阵, 视图矩阵。
    """
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    view_matrix = p.computeViewMatrix(eye_pos, target_pos, up_vector)
    return projection_matrix, view_matrix, width, height, near, far

def detect_objects(object_ids, view_matrix, projection_matrix, width, height, near, far):
    """
    使用相机检测物体位置。
    返回：物体中心位置列表（或 None）。
    """
    _, _, rgb, depth, seg = p.getCameraImage(
        width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    object_centers = []
    for obj_id in object_ids:
        seg_mask = (seg == obj_id)
        if np.sum(seg_mask) > 0:
            y, x = np.where(seg_mask)
            u, v = np.mean(x), np.mean(y)
            z = depth[int(v), int(u)] * (far - near) + near
            uv_hom = np.array([u, v, 1])
            K = np.array(projection_matrix).reshape(4, 4)[:3, :3]
            K_inv = np.linalg.inv(K)
            R_t = np.array(view_matrix).reshape(4, 4)
            world_hom = R_t @ np.append(K_inv @ uv_hom * z, 1)
            world_pos = world_hom[:3] / world_hom[3]
            object_centers.append(world_pos)
        else:
            object_centers.append(None)
    return object_centers

def calculate_ik(arm_id, end_effector_index, target_pos, target_orn, num_joints, joint_damping=0.01):
    """
    计算逆向运动学。
    返回：关节角度列表。
    """
    return p.calculateInverseKinematics(
        arm_id, end_effector_index, target_pos, targetOrientation=target_orn,
        jointDamping=[joint_damping] * num_joints
    )

def generate_trajectory(current_joints, target_joints, steps=100):
    """
    生成线性轨迹。
    返回：轨迹列表（每个元素是关节角度数组）。
    """
    trajectory = []
    for t in np.linspace(0, 1, steps):
        interp_angles = (1 - t) * np.array(current_joints) + t * np.array(target_joints)
        trajectory.append(interp_angles)
    return trajectory

def get_current_joints(arm_id, num_joints):
    """
    获取当前关节状态。
    返回：关节位置列表。
    """
    return [p.getJointState(arm_id, i)[0] for i in range(num_joints)]

def apply_pid_control(arm_id, num_joints, target_step, kp=200.0, ki=0.0, kd=50.0):
    """
    应用 PID 控制到关节。
    无返回值（直接施加扭矩）。
    """
    for i in range(num_joints):
        joint_state = p.getJointState(arm_id, i)
        current_pos, current_vel = joint_state[0], joint_state[1]
        error = target_step[i] - current_pos
        error_deriv = -current_vel
        torque = kp * error + kd * error_deriv
        p.setJointMotorControl2(arm_id, i, p.TORQUE_CONTROL, force=torque)

def perform_grasp_and_place(arm_id, num_joints, end_effector_index, obj_id, grasp_pos, place_pos,
                            target_orn, steps=100, sleep_time=1./240.):
    """
    执行抓取和放置操作。
    返回：无（执行仿真步骤）。
    """
    # 移动到抓取位置
    ik_grasp = calculate_ik(arm_id, end_effector_index, grasp_pos, target_orn, num_joints)
    current_joints = get_current_joints(arm_id, num_joints)
    trajectory_grasp = generate_trajectory(current_joints, ik_grasp, steps)
    for step in trajectory_grasp:
        apply_pid_control(arm_id, num_joints, step)
        p.stepSimulation()
        time.sleep(sleep_time)

    # 抓取
    grasp_constraint = p.createConstraint(
        arm_id, end_effector_index, obj_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, -0.1]
    )

    # 移动到放置位置
    ik_place = calculate_ik(arm_id, end_effector_index, place_pos, target_orn, num_joints)
    current_joints = get_current_joints(arm_id, num_joints)
    trajectory_place = generate_trajectory(current_joints, ik_place, steps)
    for step in trajectory_place:
        apply_pid_control(arm_id, num_joints, step)
        p.stepSimulation()
        time.sleep(sleep_time)

    # 释放
    p.removeConstraint(grasp_constraint)
    time.sleep(0.5)

def start_video_recording(file_name="simulation_video.mp4"):
    """
    启动视频录制。
    返回：日志 ID。
    """
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file_name)
    print(f"视频录制已启动：{file_name}")
    return log_id

def stop_video_recording(log_id):
    """
    停止视频录制。
    无返回值。
    """
    p.stopStateLogging(log_id)
    print("视频录制已完成")

def main():
    """
    主函数：协调所有步骤。
    """
    # 初始化
    init_simulation()

    # 加载模型
    load_ground()
    arm_id, num_joints, end_effector_index = load_arm()
    object_positions = [[0.5, 0.5, 0.5], [0.7, 0.5, 0.5], [0.5, 0.7, 0.5]]
    object_ids = load_objects(object_positions)

    # 启动视频录制
    video_log_id = start_video_recording()

    # 设置相机
    projection_matrix, view_matrix, width, height, near, far = setup_camera()

    # 多物体抓取
    place_position = [0.3, 0.3, 0.5]
    target_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    for obj_idx, obj_id in enumerate(object_ids):
        object_centers = detect_objects(object_ids, view_matrix, projection_matrix, width, height, near, far)
        target_pos = object_centers[obj_idx]
        if target_pos is None:
            print(f"物体 {obj_id} 不可见，跳过")
            continue
        grasp_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.2]
        place_pos = [place_position[0], place_position[1], place_position[2] + 0.2]
        perform_grasp_and_place(arm_id, num_joints, end_effector_index, obj_id, grasp_pos, place_pos, target_orn)

    # 停止录制并关闭
    stop_video_recording(video_log_id)
    p.disconnect()

if __name__ == "__main__":
    main()