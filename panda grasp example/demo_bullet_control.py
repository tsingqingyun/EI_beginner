
"""
demo_bullet_control.py
----------------------
Demonstrate classic controllers on the user's Panda scene (panda_sim_grasp.py).

Modes:
- Joint PD + gravity comp to a Cartesian target via IK (resolved motion)
- Computed Torque tracking of a joint-space waypoint path
- Operational Space Control to a Cartesian pose (torque)

Run:
    python demo_bullet_control.py --mode pd_ik
    python demo_bullet_control.py --mode ctc
    python demo_bullet_control.py --mode osc
"""

import time, math, argparse
import numpy as np
import pybullet as p
import pybullet_data as pd

import panda_sim_grasp as panda_sim  # your existing file
from control_classic import JointSpacePD, ComputedTorque, OperationalSpaceControl

FPS = 240.0
DT  = 1.0 / FPS

def connect():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setTimeStep(DT)
    p.setGravity(0, -9.8, 0)

def disable_position_motors(robot, joint_indices):
    """Switch selected joints to TORQUE_CONTROL by setting zero force in position motors."""
    for j in joint_indices:
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0.0)

def ik_to_pose(robot, ee, pos, orn, rest=None):
    q = p.calculateInverseKinematics(robot, ee, pos, orn,
                                     lowerLimits=[-7]*7, upperLimits=[7]*7,
                                     jointDamping=[0.01]*7,
                                     restPoses=rest if rest is not None else [0]*7,
                                     maxNumIterations=100, residualThreshold=1e-4)
    return np.array(q[:7])

def main(mode):
    connect()
    sim = panda_sim.PandaSim(p, [0,0,0])
    robot = sim.panda
    ee = panda_sim.pandaEndEffectorIndex
    dofs = list(range(panda_sim.pandaNumDofs))  # 0..6

    # Switch arm joints to torque control
    disable_position_motors(robot, dofs)

    if mode == "pd_ik":
        ctrl = JointSpacePD(p, robot, dofs, kp=[200]*7, kd=[20]*7, gravity_comp=True)
        # Cartesian goal: directly over first lego
        lego = sim.legos[0]
        (ox, oy, oz), _ = p.getBasePositionAndOrientation(lego)
        aabb_min, aabb_max = p.getAABB(lego)
        top_y = aabb_max[1]
        xd = [ox, top_y + 0.15, oz]
        Rd = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([math.pi/2, 0, 0]))
        # simple loop: recompute IK each step, then PD to q*
        for _ in range(4*FPS):
            qstar = ik_to_pose(robot, ee, xd, p.getQuaternionFromEuler([math.pi/2,0,0]))
            ctrl.step(qstar)
            p.stepSimulation(); time.sleep(DT)

    elif mode == "ctc":
        ctrl = ComputedTorque(p, robot, dofs, kp=[150]*7, kd=[30]*7)
        # plan a simple joint waypoint path: start -> home -> small move
        q0 = np.array([p.getJointState(robot, j)[0] for j in dofs])
        q1 = q0 + np.array([0.1, -0.2, 0.2, -0.1, 0.0, 0.1, -0.1])
        T = 3.0; steps = int(T/DT)
        for k in range(steps):
            s = k/steps
            qd = (1-3*s*s+2*s*s*s)*q0 + (3*s*s-2*s*s*s)*q1  # cubic
            qdd = (6*s-6*s*s)*(q1 - q0)/(T*T)               # approx
            ctrl.step(qd, qd_des=np.zeros(7), qdd_des=qdd)
            p.stepSimulation(); time.sleep(DT)

    elif mode == "osc":
        ctrl = OperationalSpaceControl(p, robot, dofs, ee)
        lego = sim.legos[0]
        (ox, oy, oz), _ = p.getBasePositionAndOrientation(lego)
        aabb_min, aabb_max = p.getAABB(lego)
        top_y = aabb_max[1]
        xd = np.array([ox, top_y + 0.12, oz])
        Rd = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([math.pi/2,0,0]))).reshape(3,3)
        Kp = np.diag([400, 400, 400, 50, 50, 50])
        Kd = np.diag([60, 60, 60, 8, 8, 8])
        for _ in range(int(4*FPS)):
            ctrl.step(xd, Rd, Kp, Kd, local_pos=(0,0,0))
            p.stepSimulation(); time.sleep(DT)

    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="pd_ik",
                    choices=["pd_ik","ctc","osc"])
    args = ap.parse_args()
    main(args.mode)
