import pybullet as p
import pybullet_data
import time

# 1. 初始化物理引擎
# 连接到 GUI 模式（带可视化窗口），也可以用 p.DIRECT（无 GUI）
physicsClient = p.connect(p.GUI)

# 2. 设置仿真环境
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置 PyBullet 数据路径
p.setGravity(0, 0, -9.81)  # 设置重力（x, y, z）
planeId = p.loadURDF("plane.urdf")  # 加载地面模型

# 3. 加载立方体
cubeStartPos = [0, 0, 10]  # 初始位置 (x, y, z)
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])  # 初始朝向（欧拉角转四元数）
cubeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])  # 创建立方体碰撞形状
cubeMass = 1  # 质量
cubeVisualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])  # 红色立方体
cubeBodyId = p.createMultiBody(cubeMass, cubeId, cubeVisualShapeId, cubeStartPos, cubeStartOrientation)

# 4. 运行仿真循环
p.setRealTimeSimulation(0)  # 关闭实时仿真，使用步进仿真
for i in range(1000):
    p.stepSimulation()  # 进行一步物理仿真
    time.sleep(1/240.)  # 模拟 240Hz 的仿真频率

# 5. 断开连接
p.disconnect()