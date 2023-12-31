import jax.numpy as np
import jax
from jaxlie import SE3

def Tr(r):
    return SE3.exp(np.array(r+[0.,0.,0.]))
def Tz(d):
    return SE3.exp(np.array([0.,0.,d]+[0.,0.,0.]))
def Tx(a):
    return SE3.exp(np.array([a, 0.,0.]+[0.,0.,0.]))
def Rx(alpha):
    return SE3.exp(np.array([0.,0.,0.]+[alpha,0.,0.]))
def Rz(th):
    return SE3.exp(np.array([0.,0.,0.]+[0.,0.,th]))

_franka_DH_params = [
    {'a': 0.,       'd': 0.333,     'alpha': 0.},
    {'a': 0.,       'd': 0.,        'alpha': -np.pi/2.},
    {'a': 0.,       'd': 0.316,     'alpha': np.pi/2.},
    {'a': 0.0825,   'd': 0.,        'alpha': np.pi/2.},
    {'a': -0.0825,  'd': 0.384,     'alpha': -np.pi/2.},
    {'a': 0.,       'd': 0.,        'alpha': np.pi/2.},
    {'a': 0.088,    'd': 0.,        'alpha': np.pi/2.},
    {'a': 0.,       'd': 0.107,     'alpha': 0.}# no joint after this
]

_franka_inertia_params = [
    { 'm': 4.970684, 
        'r': [3.875e-03, 2.081e-03, 0], 
        'I': [7.03370e-01,  7.06610e-01,  9.11700e-03, -1.39000e-04,  1.91690e-02,  6.77200e-03]},
    { 'm': 0.646926, 
        'r': [-3.141e-03, -2.872e-02, 3.495e-03], 
        'I': [7.96200e-03, 2.81100e-02, 2.59950e-02, -3.92500e-03,  7.04000e-04, 1.02540e-02]},
    { 'm': 3.228604, 
        'r': [2.7518e-02, 3.9252e-02, -6.6502e-02], 
        'I': [3.72420e-02,  3.61550e-02,  1.08300e-02, -4.76100e-03, -1.28050e-02, -1.13960e-02]},
    { 'm': 3.587895, 
        'r': [-5.317e-02, 1.04419e-01, 2.7454e-02], 
        'I': [2.58530e-02,  1.95520e-02,  2.83230e-02,  7.79600e-03,  8.64100e-03, -1.33200e-03]},
    { 'm':  1.225946, 
        'r': [-1.1953e-02, 4.1065e-02, -3.8437e-02], 
        'I': [3.55490e-02,  2.94740e-02,  8.62700e-03, -2.11700e-03,  2.29000e-04, -4.03700e-03]},
    { 'm':  1.666555, 
        'r': [6.0149e-02, -1.4117e-02, -1.0517e-02], 
        'I': [1.96400e-03,  4.35400e-03,  5.43300e-03,  1.09000e-04,  3.41000e-04, -1.15800e-03]},
    { 'm':   7.35522e-01, 
        'r': [1.0517e-02, -4.252e-03, 6.1597e-02 ], 
        'I': [1.25160e-02,  1.00270e-02,  4.81500e-03, -4.28000e-04, -7.41000e-04, -1.19600e-03]},
]

_franka_tf_tree = [
    Rx(_dhp['alpha']) @ Tx(_dhp['a']) @ Tz(_dhp['d'])  for _dhp in _franka_DH_params
]
_franka_inertia_tree = [
    (Tr(_ip['r']), np.diag(np.array([_ip['m']]*3 + _ip['I'][:3]))) for _ip in _franka_inertia_params
]