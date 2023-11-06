import jax.numpy as np
import jax
from jaxlie import SE3
from franka_config import Tz, Rz, _franka_tf_tree, _franka_inertia_tree

def forward_kinematics(q):
    _L = SE3.identity()
    joint_tfs = []
    com_tfs = []
    for i in range(7):
        _L = _L@_franka_tf_tree[i] @ Rz(q[i])
        joint_tfs.append(_franka_tf_tree[i] @ Rz(q[i]))
        com_tfs.append(_L @ _franka_inertia_tree[i][0])
    return joint_tfs, com_tfs

def get_ee_frame(q):
    joint_tfs, com_tfs = forward_kinematics(q)
    return com_tfs[-1]@Tz(0.12)@Rz(np.pi/4)

def body_twists(q, qdot):
    # assumes revolute joint about z-axis
    xi = np.array([0.,0.,0.,0.,0.,1.0]) 
    vbL = np.zeros(6)
    body_vels = []
    for i in range(7):
        vbL = (_franka_tf_tree[i] @ Rz(q[i])).adjoint() @ vbL + xi * qdot[i]
        vbm = _franka_inertia_tree[i][0].adjoint() @ vbL
        body_vels.append(vbm)
    return body_vels
