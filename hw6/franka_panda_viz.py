import meshcat
from meshcat import Visualizer
import meshcat.geometry as geom
import meshcat.transformations as tfm
import numpy as np
from jaxlie import SE3
import random
import glob 
from franka_config import _franka_tf_tree
from franka_kinematics import forward_kinematics

def get_rand_color():
    random_number = random.randint(0,16777215)
    hex_number = str(hex(random_number))
    hex_number ='0x'+ hex_number[2:]
    return hex_number

class FrankaVisualizer(Visualizer):
    def __init__(self) -> None:
        Visualizer.__init__(self)
        self._q_home = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])
        joint_tfs   = forward_kinematics(self._q_home)[0]
        jnt_list    = []
        base = self["base"]
        root = base
        for i in range(7):
            for j, _fname in enumerate(glob.glob('./franka_panda/assets/link{}*.obj'.format(i))):
                _geom = root['geom_{}'.format(j)]
                _geom.set_object(
                    geom.ObjMeshGeometry.from_file(_fname)
                )
            root = root['jnt{}'.format(i)]
            jnt_list.append(root)
            root.set_transform(np.array(joint_tfs[i].as_matrix()))

        for j, _fname in enumerate(glob.glob('./franka_panda/assets/link7*.obj'.format(i))):
            _geom = root['geom_7{}'.format(j)]
            _geom.set_object(
                geom.ObjMeshGeometry.from_file(_fname)
            )
        hand = root['hand']
        hand.set_transform(np.array(_franka_tf_tree[-1].as_matrix())@tfm.rotation_matrix(-np.pi/4, [0,0,1],[0,0,0]))
        for j, _fname in enumerate(glob.glob('./franka_panda/assets/hand*.obj'.format(i))):
            _geom = hand['geom_hand{}'.format(j)]
            _geom.set_object(
                geom.ObjMeshGeometry.from_file(_fname)
            )
        finger1 = hand["finger1"]
        finger1.set_transform(tfm.translation_matrix([0,0,0.0584]))
        for j, _fname in enumerate(glob.glob('./franka_panda/assets/finger*.obj'.format(i))):
            _geom = finger1['geom_hand{}'.format(j)]
            _geom.set_object(
                geom.ObjMeshGeometry.from_file(_fname)
            )

        finger2 = hand["finger2"]
        finger2.set_transform(tfm.translation_matrix([0,0,0.0584])@tfm.rotation_matrix(np.pi, [0.,0.,1.]))
        for j, _fname in enumerate(glob.glob('./franka_panda/assets/finger*.obj'.format(i))):
            _geom = finger2['geom_hand{}'.format(j)]
            _geom.set_object(
                geom.ObjMeshGeometry.from_file(_fname)
            )
        self._jnts = jnt_list

    def render(self, q):
        jnt_tfs = forward_kinematics(q)[0]
        for i, _tf in enumerate(jnt_tfs): 
            self._jnts[i].set_transform(np.array(_tf.as_matrix()))
            # self._jnts[i].set_transform(tfm.rotation_matrix(qi, [0,0,1]))