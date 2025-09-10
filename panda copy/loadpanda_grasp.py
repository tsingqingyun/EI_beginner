# 导入 PyBullet 及相关依赖库
import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import panda_sim_grasp as panda_sim


# 帧率
fps = 240.
# 仿真步长
timeStep = 1. / fps

# 连接物理引擎
p.connect(p.GUI)

# 配置可视化参数
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)  # Y 轴朝上
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)        # 关闭 GUI 面板
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)      # 提高物理引擎命令速率
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35, -0.13, 0])
p.setAdditionalSearchPath(pd.getDataPath())            # 设置数据文件搜索路径

# 设置仿真步长和重力
p.setTimeStep(timeStep)
p.setGravity(0, -9.8, 0)

# 创建 Panda 机械臂自动仿真对象
panda = panda_sim.PandaSimAuto(p, [0, 0, 0])
# 设置控制步长
panda.control_dt = timeStep

# 主仿真循环，运行10000步
for i in range(10000):
	panda.step()           # 自动状态机控制机械臂
	p.stepSimulation()     # 推进物理仿真
	time.sleep(timeStep)   # 控制仿真速度



