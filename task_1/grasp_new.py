import pybullet as p
import pybullet_data
import time
import numpy as np
import os

class UR5Simulation:
    def __init__(self):
        # 初始化物理引擎
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载环境
        self.load_environment()
        
        # 加载机械臂
        self.load_robot()
        
        # 设置控制参数
        self.setup_control_parameters()
        
        # 初始化状态变量
        self.prev_button_state = 1
        self.is_sim_running = True
    
    def load_environment(self):
        """加载环境物体"""
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
        
        # 创建目标物体
        cube_size = 0.05
        self.cubeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3)
        self.cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[0, 0, 1, 1])
        self.cubeId = p.createMultiBody(baseMass=0.5, 
                                      baseCollisionShapeIndex=self.cubeId,
                                      baseVisualShapeIndex=self.cube_visual,
                                      basePosition=[0.6, 0.2, 0.7])
    
    def load_robot(self):
        """加载机械臂"""
        # 加载KUKA iiwa机械臂
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0.65], useFixedBase=True)
        
        # 获取关节信息
        self.numJoints = p.getNumJoints(self.robotId)
        self.endEffectorIndex = self.numJoints - 1
        
        print(f"机械臂关节数量: {self.numJoints}")
        for i in range(self.numJoints):
            joint_info = p.getJointInfo(self.robotId, i)
            print(f"关节 {i}: {joint_info[1]}")
        
        # 设置初始位置
        self.set_initial_joint_positions()
    
    def set_initial_joint_positions(self):
        """设置机械臂初始位置"""
        rest_poses = [0, 0, 0, 0.5 * np.pi, 0, -0.5 * np.pi, 0]
        for i in range(self.numJoints):
            p.resetJointState(self.robotId, i, rest_poses[i])
            p.setJointMotorControl2(
                bodyIndex=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=rest_poses[i],
                force=500,
                positionGain=0.3,
                velocityGain=1.0
            )
        print("机械臂初始位置设置完成")
    
    def setup_control_parameters(self):
        """设置控制参数"""
        # 目标位置控制滑块
        self.target_x = p.addUserDebugParameter("Target X", -1, 1, 0.5)
        self.target_y = p.addUserDebugParameter("Target Y", -1, 1, 0)
        self.target_z = p.addUserDebugParameter("Target Z", 0.5, 1.2, 0.8)
        
        # 开始按钮
        self.start_button = p.addUserDebugParameter("Start Move", 1, 0, 1)
        
        # 创建目标可视化球
        self.target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.5])
        self.target_id = p.createMultiBody(baseMass=0, 
                                         baseVisualShapeIndex=self.target_visual,
                                         basePosition=[0.5, 0, 0.8])
    
    def calculate_ik(self, target_pos):
        """计算逆运动学"""
        print(f"\n计算IK，目标位置: {target_pos}")
        
        # 获取当前末端位置
        current_pos = p.getLinkState(self.robotId, self.endEffectorIndex)[0]
        print(f"当前末端位置: {current_pos}")
        
        # 末端执行器朝向（保持垂直向下）
        target_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        
        # 计算IK
        joint_poses = p.calculateInverseKinematics(
            self.robotId, 
            self.endEffectorIndex,
            target_pos,
            target_orientation,
            lowerLimits=[-np.pi]*self.numJoints,
            upperLimits=[np.pi]*self.numJoints,
            jointRanges=[2*np.pi]*self.numJoints,
            restPoses=[0]*self.numJoints,
            jointDamping=[0.1]*self.numJoints,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        
        print(f"计算得到的关节角度: {joint_poses}")
        return joint_poses
    
    def move_to_target(self, target_pos):
        """平滑移动到目标位置"""
        print(f"开始移动到目标位置: {target_pos}")
        
        # 计算IK解
        joint_poses = self.calculate_ik(target_pos)
        
        # 设置关节位置控制器
        for i in range(self.numJoints):
            p.setJointMotorControl2(
                bodyIndex=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
                positionGain=0.3,
                velocityGain=1.0
            )
        
        # 等待到达目标位置
        for step in range(200):
            if not self.is_sim_running:
                break
                
            p.stepSimulation()
            
            # 实时打印末端执行器位置
            if step % 40 == 0:
                end_pos = p.getLinkState(self.robotId, self.endEffectorIndex)[0]
                print(f"步数 {step}: 末端位置 = {end_pos}")
            
            time.sleep(1./240.)
        
        if self.is_sim_running:
            # 最终位置检查
            end_pos = p.getLinkState(self.robotId, self.endEffectorIndex)[0]
            print(f"移动完成，最终末端位置: {end_pos}")
            print(f"与目标位置的距离: {np.linalg.norm(np.array(end_pos)-np.array(target_pos))}")
    
    def check_button_press(self):
        """检查按钮状态并执行相应操作"""
        current_button_state = p.readUserDebugParameter(self.start_button)
        print(f"按钮状态: {current_button_state}")  # 调试输出
        
        if current_button_state == 0 and self.prev_button_state == 1:
            print("\n=== 按钮按下触发 ===")
            target_pos = [
                p.readUserDebugParameter(self.target_x),
                p.readUserDebugParameter(self.target_y),
                p.readUserDebugParameter(self.target_z)
            ]
            print(f"目标位置: {target_pos}")
            self.move_to_target(target_pos)
        
        self.prev_button_state = current_button_state
    
    def run_simulation(self):
        """主仿真循环"""
        print("\n仿真开始，使用说明:")
        print("1. 使用滑块调整目标位置")
        print("2. 按下'Start Move'按钮移动机械臂")
        print("按Ctrl+C退出\n")
        
        try:
            while self.is_sim_running:
                try:
                    # 更新目标可视化球位置
                    target_pos = [
                        p.readUserDebugParameter(self.target_x),
                        p.readUserDebugParameter(self.target_y),
                        p.readUserDebugParameter(self.target_z)
                    ]
                    p.resetBasePositionAndOrientation(self.target_id, target_pos, [0, 0, 0, 1])
                    
                    # 检查按钮状态
                    self.check_button_press()
                    
                    p.stepSimulation()
                    time.sleep(1./240.)
                except p.error as e:
                    if "Not connected to physics server" in str(e):
                        print("检测到物理服务器断开连接")
                        self.is_sim_running = False
                    else:
                        raise
                
        except KeyboardInterrupt:
            print("\n用户中断仿真")
        except Exception as e:
            print(f"\n仿真出错: {str(e)}")
        finally:
            self.is_sim_running = False
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'physicsClient') and p.isConnected():
            try:
                p.disconnect()
            except:
                pass
        print("资源清理完成")

def main():
    sim = UR5Simulation()
    sim.run_simulation()

if __name__ == "__main__":
    main()