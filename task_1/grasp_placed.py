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
    p.setRealTimeSimulation(1)  # 修改：启用实时仿真，确保与真实时间同步
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

def get_joint_limits(arm_id, num_joints):
    """
    新增：获取关节下限和上限。
    返回：lower_limits, upper_limits 列表。
    """
    lower_limits = []
    upper_limits = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(arm_id, i)
        lower_limits.append(joint_info[8])
        upper_limits.append(joint_info[9])
    return lower_limits, upper_limits

def calculate_ik(arm_id, end_effector_index, target_pos, target_orn, num_joints, joint_damping=0.01):
    """
    计算逆向运动学。
    返回：关节角度列表。
    """
    lower_limits, upper_limits = get_joint_limits(arm_id, num_joints)  # 修改：添加关节限制，提高稳定性
    return p.calculateInverseKinematics(
        arm_id, end_effector_index, target_pos, targetOrientation=target_orn,
        lowerLimits=lower_limits, upperLimits=upper_limits,
        jointDamping=[joint_damping] * num_joints
    )

def generate_trajectory(current_joints, target_joints, steps=500):  # 修改：步数增加到500，使运动更慢
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

def apply_pid_control(arm_id, num_joints, target_step, kp=100.0, ki=1.0, kd=100.0):  # 修改：调整参数，增加Ki，降低Kp，提高Kd
    """
    应用 PID 控制到关节。
    无返回值（直接施加扭矩）。
    """
    integral_errors = [0.0] * num_joints  # 新增：积分项累积（简化版，避免全局变量）
    for i in range(num_joints):
        joint_state = p.getJointState(arm_id, i)
        current_pos, current_vel = joint_state[0], joint_state[1]
        error = target_step[i] - current_pos
        integral_errors[i] += error  # 累积积分
        error_deriv = -current_vel
        torque = kp * error + ki * integral_errors[i] + kd * error_deriv
        p.setJointMotorControl2(arm_id, i, p.TORQUE_CONTROL, force=torque)

def check_stability(arm_id, end_effector_index, timeout=1.0):
    """
    新增：检查末端执行器是否稳定（速度接近0）。
    返回：True 如果稳定。
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        link_state = p.getLinkState(arm_id, end_effector_index, computeLinkVelocity=1)
        linear_vel, angular_vel = link_state[6], link_state[7]
        if np.linalg.norm(linear_vel) < 0.01 and np.linalg.norm(angular_vel) < 0.01:
            return True
        time.sleep(0.1)
    return False


def adjust_physics_parameters(body_id, friction=1.0, damping=0.5, stiffness=1000.0):
    """
    新增：调整物体或臂的物理参数，提高摩擦、阻尼和刚度，防止吸附。
    """
    p.changeDynamics(body_id, -1, lateralFriction=friction, linearDamping=damping, 
                     contactStiffness=stiffness, contactDamping=damping * 10)
    print(f"调整物理参数 for ID {body_id}: friction={friction}, damping={damping}, stiffness={stiffness}")

# ... (load_objects 函数后调用此函数)

def perform_grasp_and_place(arm_id, num_joints, end_effector_index, obj_id, grasp_pos, place_pos,
                            target_orn, steps=1000, sleep_time=1./480.):  # 修改：步数增加，sleep_time 减小（更精确）
    """
    执行抓取和放置操作。
    返回：无（执行仿真步骤）。
    """
    # 移动到抓取位置（逻辑不变）
    ik_grasp = calculate_ik(arm_id, end_effector_index, grasp_pos, target_orn, num_joints)
    current_joints = get_current_joints(arm_id, num_joints)
    trajectory_grasp = generate_trajectory(current_joints, ik_grasp, steps)
    for step in trajectory_grasp:
        apply_pid_control(arm_id, num_joints, step, kp=50.0, ki=1.0, kd=150.0)  # 修改：进一步调优 PID
        p.stepSimulation()
        time.sleep(sleep_time)

    time.sleep(1.0)
    if not check_stability(arm_id, end_effector_index):
        print("臂不稳定，重新尝试...")

    # 抓取（调整偏移，避免嵌入）
    grasp_constraint = p.createConstraint(
        arm_id, end_effector_index, obj_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, -0.05]  # 修改：减小偏移到 -0.05，减少嵌入
    )
    time.sleep(1.0)

    # 移动到放置位置（逻辑不变）
    ik_place = calculate_ik(arm_id, end_effector_index, place_pos, target_orn, num_joints)
    current_joints = get_current_joints(arm_id, num_joints)
    trajectory_place = generate_trajectory(current_joints, ik_place, steps)
    for step in trajectory_place:
        apply_pid_control(arm_id, num_joints, step, kp=50.0, ki=1.0, kd=150.0)
        p.stepSimulation()
        time.sleep(sleep_time)

    # 释放并施加小力推动物体脱离
    p.removeConstraint(grasp_constraint)
    p.applyExternalForce(obj_id, -1, [0, 0, -0.1], [0, 0, 0], p.LINK_FRAME)  # 新增：施加向下小力（-0.1 N），帮助分离
    time.sleep(2.0)  # 修改：延长等待到2秒，确保分离
import pybullet as p
import pybullet_data
import time
import numpy as np
from threading import Thread, Lock

def init_simulation(gravity=(0, 0, -9.81), time_step=1./240.):
    """带异常处理的初始化"""
    try:
        client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity)
        p.setTimeStep(time_step)
        p.setRealTimeSimulation(0)  # 关闭实时仿真，使用精确步进控制
        return client_id
    except Exception as e:
        print(f"初始化错误: {e}")
        return None

def safe_simulation_step():
    """带锁的仿真步进控制"""
    lock = Lock()
    with lock:
        try:
            p.stepSimulation()
            time.sleep(1./480.)
        except Exception as e:
            print(f"仿真步进错误: {e}")

def main():
    client_id = init_simulation()
    if client_id is None:
        return
    
    try:
        # 带错误处理的仿真主体
        load_ground()
        arm_id, num_joints, end_effector_index = load_arm()
        object_positions = [[0.5, 0.5, 0.5], [0.7, 0.5, 0.5], [0.5, 0.7, 0.5]]
        object_ids = load_objects(object_positions)
        
        adjust_physics_parameters(arm_id, friction=2.0, damping=0.8, stiffness=2000.0)
        for obj_id in object_ids:
            adjust_physics_parameters(obj_id, friction=2.0, damping=0.8, stiffness=2000.0)
        
        # 主仿真循环
        for _ in range(10000):
            safe_simulation_step()
            time.sleep(0.001)  # 防止CPU占用过高
            
    except Exception as e:
        print(f"主函数错误: {e}")
    finally:
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    Thread(target=main, daemon=True).start()