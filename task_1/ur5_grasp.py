import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os

class UR5Simulation:
    def __init__(self):
        """初始化仿真环境"""
        # 启动仿真引擎的GUI
        self.physicsClient = p.connect(p.GUI)
        
        # 设置重力加速度
        p.setGravity(0, 0, -9.81)
        
        # 设置附加搜索路径
        self.setup_search_paths()
        
        # 设置视角
        self.set_camera_view()
        
        # 加载环境对象
        self.load_environment()
        
        # 加载机械臂
        self.load_robot()
        
        # 初始化控制参数
        self.setup_control_parameters()
        
        # 初始化状态变量
        self.button_state_prev = 0
        
    def setup_search_paths(self):
        """设置URDF文件搜索路径"""
        # 添加当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(current_dir)
        print(f"添加搜索路径: {current_dir}")
        
        # 尝试添加pybullet_data路径
        try:
            data_path = pybullet_data.getDataPath()
            if os.path.exists(data_path):
                p.setAdditionalSearchPath(data_path)
                print(f"添加pybullet_data路径: {data_path}")
        except:
            print("pybullet_data路径不可用")
    
    def set_camera_view(self, distance=3, yaw=120, pitch=-45, target=[0, 0, 0]):
        """设置相机视角"""
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target
        )
    
    def load_environment(self):
        """加载环境对象"""
        # 加载平面模型作为地面
        try:
            self.planeId = p.loadURDF("plane.urdf")
            print("地面加载完成")
        except:
            print("地面模型未找到，创建简单地面")
            self.create_simple_ground()
        
        # 尝试加载托盘
        try:
            self.trayId = p.loadURDF("tray/traybox.urdf", basePosition=[0.6, 0, 0.63])
            print("托盘加载完成")
        except:
            print("托盘模型未找到，跳过加载")
            self.trayId = None
        
        # 尝试加载桌子
        try:
            self.tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
            print("桌子加载完成")
        except:
            print("桌子模型未找到，跳过加载")
            self.tableId = None
        
        print("环境加载完成")
    
    def create_simple_ground(self):
        """创建简单的地面"""
        ground_shape = p.createCollisionShape(p.GEOM_PLANE)
        ground_visual = p.createVisualShape(p.GEOM_PLANE, planeNormal=[0, 0, 1], rgbaColor=[0.6, 0.8, 0.6, 1])
        self.planeId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, baseVisualShapeIndex=ground_visual)
    
    def load_robot(self):
        """加载机械臂"""
        # 尝试加载不同的URDF文件
        urdf_files = [
            "ur5_simple.urdf",  # 我们创建的简单URDF
            "ur5/ur5.urdf",     # pybullet内置的UR5
            "kuka_iiwa/model.urdf",  # 其他机器人
            "franka_panda/panda.urdf"
        ]
        
        self.robotId = None
        for urdf_file in urdf_files:
            try:
                print(f"尝试加载: {urdf_file}")
                self.robotId = p.loadURDF(urdf_file, useFixedBase=True, basePosition=[0.0, 0, 0.62])
                print(f"成功加载: {urdf_file}")
                break
            except Exception as e:
                print(f"加载失败 {urdf_file}: {e}")
                continue
        
        if self.robotId is None:
            print("所有URDF文件加载失败，创建简单几何体")
            self.create_simple_robot()
            self.numJoints = 3  # 简单机器人有3个关节
            self.endEffectorIndex = 2
        else:
            # 获取关节信息
            self.numJoints = p.getNumJoints(self.robotId)
            self.endEffectorIndex = self.numJoints - 1  # 末端通常是最后一个关节
            print(f"机械臂关节数量: {self.numJoints}")
            self.print_joint_info()
            self.set_initial_position()
    
    def create_simple_robot(self):
        """创建简单的机器人几何体"""
        print("创建简单机器人...")
        
        # 底座
        base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1])
        self.robotId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_shape, baseVisualShapeIndex=base_visual, basePosition=[0, 0, 0.62])
        
        # 第一个关节（旋转）
        joint1_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.2)
        joint1_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, height=0.2, rgbaColor=[1, 0, 0, 1])
        joint1_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=joint1_shape, baseVisualShapeIndex=joint1_visual, basePosition=[0, 0, 0.72])
        p.createConstraint(self.robotId, -1, joint1_id, -1, p.JOINT_REVOLUTE, [0, 0, 1], [0, 0, 0.1], [0, 0, 0], [0, 0, 1])
        
        # 第二个关节（俯仰）
        joint2_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.2])
        joint2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.2], rgbaColor=[0, 1, 0, 1])
        joint2_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=joint2_shape, baseVisualShapeIndex=joint2_visual, basePosition=[0, 0, 0.92])
        p.createConstraint(joint1_id, -1, joint2_id, -1, p.JOINT_REVOLUTE, [0, 1, 0], [0, 0, 0.1], [0, 0, -0.2], [0, 1, 0])
        
        # 末端执行器
        end_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        end_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
        end_id = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=end_shape, baseVisualShapeIndex=end_visual, basePosition=[0, 0, 1.12])
        p.createConstraint(joint2_id, -1, end_id, -1, p.JOINT_REVOLUTE, [1, 0, 0], [0, 0, 0.2], [0, 0, 0], [1, 0, 0])
        
        print("简单机器人创建完成")
    
    def print_joint_info(self):
        """打印关节信息"""
        print("关节信息:")
        for joint_index in range(self.numJoints):
            joint_info = p.getJointInfo(self.robotId, joint_index)
            joint_name = joint_info[1].decode('utf-8') if isinstance(joint_info[1], bytes) else joint_info[1]
            print(f"  关节 {joint_index}: {joint_name}")
    
    def set_initial_position(self):
        """设置机械臂初始位置"""
        if self.numJoints >= 6:
            restingPosition = [0, 3.14, -1.57, 1.57, 1.57, 1.57]
            for jointNumber in range(min(self.numJoints, 6)):
                p.resetJointState(self.robotId, jointNumber, restingPosition[jointNumber])
        print("机械臂初始位置设置完成")
    
    def setup_control_parameters(self):
        """设置控制参数"""
        # 目标点控制滑块
        self.target_x = p.addUserDebugParameter("Target X", -2, 2, 0.5)
        self.target_y = p.addUserDebugParameter("Target Y", -2, 2, 0)
        self.target_z = p.addUserDebugParameter("Target Z", 0, 2, 0.8)
        
        # 开始按钮
        self.button_start = p.addUserDebugParameter("Start", 1, 0, 1)
        
        # 创建目标点球
        self.create_target_sphere()
        
        print("控制参数设置完成")
    
    def create_target_sphere(self):
        """创建目标点可视化球"""
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        self.sphere_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=-1, 
            baseVisualShapeIndex=sphere_visual, 
            basePosition=[0.5, 0, 0.8]
        )
    
    def run_simulation(self):
        """运行仿真"""
        print("仿真开始...")
        print("使用滑块调整目标位置，按下开始按钮移动")
        
        try:
            while True:
                # 获取目标位置
                x = p.readUserDebugParameter(self.target_x)
                y = p.readUserDebugParameter(self.target_y)
                z = p.readUserDebugParameter(self.target_z)
                target_pos = [x, y, z]
                
                # 更新目标球位置
                p.resetBasePositionAndOrientation(self.sphere_id, target_pos, [0, 0, 0, 1])
                
                # 检查按钮
                button_state = p.readUserDebugParameter(self.button_start)
                if button_state != self.button_state_prev and button_state == 0:
                    print("开始移动...")
                    # 这里可以添加移动逻辑
                
                self.button_state_prev = button_state
                
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
                
        except KeyboardInterrupt:
            print("程序中断")
    
def cleanup(self):
    """清理资源"""
    if p.isConnected():
        p.disconnect()
    print("仿真结束")

def main():
    """主函数"""
    simulation = UR5Simulation()
    try:
        simulation.run_simulation()
    except Exception as e:
        print(f"错误: {e}")
    finally:
        simulation.cleanup()

if __name__ == "__main__":
    main()