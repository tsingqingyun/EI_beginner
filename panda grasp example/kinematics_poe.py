
"""
kinematics_poe.py
-----------------
Product-of-Exponential (PoE) based kinematics utilities:
- Forward kinematics: fkine(M, Slist, q)
- Space & Body Jacobians
- Damped-least-squares (DLS) IK on SE(3) using log-map pose error
- Simple planar examples

Conventions: Slist is 6xn with twists in the SPACE frame; M is home pose of end-effector in space frame.
"""

import numpy as np
from robot_math import se3_exp, make_T, adjoint, pose_error, T_inv

_EPS = 1e-9

def fkine_poe(M, Slist, q):
    """Forward kinematics via product of exponentials (space frame)."""
    T = np.array(M, dtype=float).reshape(4,4)
    for i in range(Slist.shape[1]-1, -1, -1):
        T = se3_exp(Slist[:,i], q[i]) @ T
    return T

def jacobian_space(Slist, q):
    """Compute space Jacobian Js(q) from Slist (6xn) using POE."""
    n = Slist.shape[1]
    Js = np.zeros((6, n))
    T = np.eye(4)
    for i in range(n):
        Js[:, i] = (adjoint(T) @ Slist[:, i])
        T = T @ se3_exp(Slist[:, i], q[i])
    return Js

def jacobian_body(M, Slist, q):
    """Body Jacobian from space Jacobian: Jb = Ad(T^{-1}) Js."""
    T = fkine_poe(M, Slist, q)
    Js = jacobian_space(Slist, q)
    Jb = adjoint(T_inv(T)) @ Js
    return Jb

def ik_dls_poe(M, Slist, q0, T_des, max_iters=100, eps=1e-4, lam=1e-3, alpha=0.5):
    """
    Damped least squares IK in the body frame:
    q_{k+1} = q_k + alpha * Jb^T (Jb Jb^T + lam^2 I)^{-1} * err
    where err = log(T(q)^{-1} T_des) (6,).

    Returns (q, success, iters).
    """
    q = np.array(q0, dtype=float).copy()
    for it in range(max_iters):
        T = fkine_poe(M, Slist, q)
        err = pose_error(T, T_des)  # 6,
        if np.linalg.norm(err) < eps:
            return q, True, it
        Jb = jacobian_body(M, Slist, q)  # 6xn
        # DLS step
        JJt = Jb @ Jb.T
        dq = Jb.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), err)
        q = q + alpha * dq
    return q, False, max_iters

# ---------- simple example (3-DOF planar) ----------

def example_planar_3dof():
    """
    Return (M, Slist) for a planar 3R arm in XY plane about z-axis.
    Link lengths l1, l2, l3.
    """
    l1 = 0.3; l2 = 0.25; l3 = 0.2
    # Home pose: end-effector at (l1+l2+l3, 0, 0), orientation identity
    M = np.array([[1,0,0,l1+l2+l3],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]], dtype=float)
    # Space twists (revolute about z) located at (0,0,0), (l1,0,0), (l1+l2,0,0)
    S1 = np.array([0,0,1, 0,0,0])
    S2 = np.array([0,0,1, 0,-l1,0])
    S3 = np.array([0,0,1, 0,-(l1+l2),0])
    Slist = np.c_[S1,S2,S3]
    return M, Slist
