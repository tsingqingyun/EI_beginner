
"""
robot_math.py
-------------
Essential SO(3)/SE(3) utilities for robotics:
- hat/vee operators
- exp/log on SO(3), SE(3)
- adjoint, composition, inverse
- conversions between rotation representations

All functions are NumPy-based and numerically safe for small angles.
"""

import numpy as np

_EPS = 1e-9

# ---------- basic helpers ----------

def skew(w):
    """Return [w]^ (skew-symmetric) from a 3-vector."""
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def unskew(W):
    """Inverse of skew: R^3 from a 3x3 skew-symmetric matrix."""
    return np.array([W[2,1], W[0,2], W[1,0]], dtype=float)

def rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def rpy_to_R(roll, pitch, yaw, order="xyz"):
    """Roll-pitch-yaw to rotation matrix. Default intrinsic XYZ."""
    if order.lower() == "xyz":
        return rotx(roll) @ roty(pitch) @ rotz(yaw)
    raise NotImplementedError("Only 'xyz' supported for simplicity.")

def R_to_rpy(R):
    """Extract intrinsic XYZ roll/pitch/yaw from rotation matrix."""
    # Avoid gimbal lock edge cases; assumes proper rotation matrix
    sy = -R[2,0]
    cy = np.sqrt(max(0.0, 1 - sy*sy))
    if cy > _EPS:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arcsin(sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        # near gimbal lock
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arcsin(sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw], dtype=float)

# ---------- SO(3) exp/log ----------

def so3_exp(omega, theta=None):
    """
    Rodrigues' formula for exp([omega]^ * theta).
    omega: 3x vector (not necessarily unit).
    theta: optional angle; if None -> theta = ||omega|| and omega normalized.
    """
    w = np.array(omega, dtype=float).reshape(3)
    if theta is None:
        th = np.linalg.norm(w)
        if th < _EPS:
            return np.eye(3)
        wn = w / th
    else:
        th = float(theta)
        wn = w / (np.linalg.norm(w) + _EPS)

    W = skew(wn)
    return np.eye(3) + np.sin(th)*W + (1-np.cos(th))*(W@W)

def so3_log(R):
    
    """
    Matrix log of R \in SO(3). Returns the rotation vector omega*theta (3,).
    """
    R = np.array(R, dtype=float).reshape(3,3)
    cos_th = (np.trace(R) - 1.0) / 2.0
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-7:
        # first-order approximation around I
        return unskew(R - R.T) / 2.0
    else:
        return (th / (2.0*np.sin(th))) * unskew(R - R.T)

# ---------- SE(3) helpers ----------

def make_T(R=np.eye(3), p=np.zeros(3)):
    T = np.eye(4)
    T[:3,:3] = np.array(R, dtype=float).reshape(3,3)
    T[:3, 3] = np.array(p, dtype=float).reshape(3)
    return T

def T_inv(T):
    """Inverse of homogeneous transform."""
    R = T[:3,:3]; p = T[:3,3]
    Rt = R.T
    Ti = np.eye(4)
    Ti[:3,:3] = Rt
    Ti[:3, 3] = -Rt @ p
    return Ti

def adjoint(T):
    """Adjoint of SE(3): 6x6 matrix that maps twists between frames."""
    R = T[:3,:3]
    p = T[:3,3]
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = skew(p) @ R
    return Ad

def se3_hat(xi):
    """hat operator for a twist (6,) -> 4x4 matrix."""
    w = xi[:3]; v = xi[3:]
    Xi = np.zeros((4,4))
    Xi[:3,:3] = skew(w)
    Xi[:3, 3] = v
    return Xi

def se3_vee(Xi):
    """vee operator for a 4x4 se(3) matrix -> (6,) twist."""
    w = unskew(Xi[:3,:3])
    v = Xi[:3,3]
    return np.r_[w, v]

# ---------- SE(3) exp/log ----------

def se3_exp(xi, theta=1.0):
    """
    expm([xi]^ * theta) with closed-form for twists.
    xi = [w; v] (6,), w is angular part, v translational part.
    """
    xi = np.array(xi, dtype=float).reshape(6)
    w = xi[:3]; v = xi[3:]
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-8:
        # Pure translation: R = I, p = v*theta
        R = np.eye(3)
        p = v * theta
        return make_T(R, p)

    wn = w / w_norm
    th = w_norm * theta
    R = so3_exp(wn, th)

    W = skew(wn)
    I = np.eye(3)
    A = np.sin(th) / w_norm
    B = (1 - np.cos(th)) / (w_norm**2)
    C = (th - np.sin(th)) / (w_norm**3)
    V = I*theta + A*W + B*(W@W)  # common but incorrect, see below
    # Correct V per MR: I*theta + (1-cos th)/||w||^2 [w]^ + (th - sin th)/||w||^3 [w]^2
    V = I*theta + (1-np.cos(th))/(w_norm**2)*W + (th - np.sin(th))/(w_norm**3)*(W@W)
    p = V @ v
    return make_T(R, p)

def se3_log(T):
    """Matrix log of T in SE(3). Returns (xi, theta) with xi unit twist."""
    R = T[:3,:3]; p = T[:3,3]
    wth = so3_log(R)
    th = np.linalg.norm(wth)
    if th < 1e-8:
        # Pure translation: xi = [0,0,0, p/||p||], theta = ||p||
        d = np.linalg.norm(p)
        if d < 1e-12:
            return np.zeros(6), 0.0
        xi = np.r_[np.zeros(3), p/d]
        return xi, d
    w = wth / th
    W = skew(w)
    I = np.eye(3)
    A = (1 - np.cos(th)) / (th**2)
    B = (th - np.sin(th)) / (th**3)
    V = I + A*W + B*(W@W)
    v = np.linalg.solve(V, p)
    xi = np.r_[w, v]
    return xi, th

# ---------- utilities ----------

def pose_error(T_current, T_desired):
    """
    6x1 body-frame error (twist) using logmap of T_err = T_c^{-1} T_d.
    Return as (6,) vector: [omega_err; v_err].
    """
    T_err = T_inv(T_current) @ T_desired
    xi, th = se3_log(T_err)
    return xi * th

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)
