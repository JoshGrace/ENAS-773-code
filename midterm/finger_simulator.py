#! /usr/bin/python

import numpy as np
import math
from pyquaternion import Quaternion
import IPython
from scipy.optimize import fmin_l_bfgs_b, minimize
import copy


class modelOKinematics(object):

    '''static variables'''
    offsetT = None
    lenP = None
    lenD = None
    angleP = None
    restingDistalAngle = None

    staticInitialized = False
    motorPulleyRadius = 0.0105
    jointRadii = np.array([0.00635, 0.008])
    springConstant = np.array([0.178, 0.49])

    def __init__(self):

        self.jointPositions = [0, 0]
        self.contacts = None
        self.mFrame = None
        self.mFrameOld = None
        self.optMaxIter = 5
        self.toolTip = None
        self.jointBounds = [(0, math.pi), (0, math.pi / 2.)]

        if not modelOKinematics.staticInitialized:
            self.initHand()

    def getJointConfig(self):
        return self.jointPositions

    def getContacts(self):
        return self.contacts

    def getJointPositions(self):
        return self.jointPositions

    def updateTooltip(self):
        if self.mFrameOld == None:
            return
        if self.toolTip == None:
            return

        ttNew = self.mFrame * self.mFrameOld.Inverse() * self.toolTip
        self.toolTip = ttNew

    def initHand(self):
        # P to D
        transP = np.array([0, 0, 0.8])
        rotP = np.array([1, 0, 0, 0])
        # D to tip
        transD = np.array([0, 0, 0.001])
        rotD = np.array([1, 0, 0, 0])
        # base to P
        modelOKinematics.offsetT = np.array([ 0, 0, 0])
        rotP2 = np.array([1, 0, 0, 0])


        qP = Quaternion(x=rotP2[1], y=rotP2[2], z=rotP2[3], w=rotP2[0])
        modelOKinematics.angleP = qP.yaw_pitch_roll[2]
        qD = Quaternion(x=rotP[1], y=rotP[2], z=rotP[3], w=rotP[0])

        modelOKinematics.lenP = np.linalg.norm(transP)
        modelOKinematics.lenD = np.linalg.norm(transD)

        u = np.array(transP)
        v = np.array(transD)
        cosTheta = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))

        modelOKinematics.restingDistalAngle = math.acos(cosTheta) + qD.yaw_pitch_roll[0]

        modelOKinematics.staticInitialized = True

    def forwardKinematics(self, jointPositions):
        aa = 0
        fingerBaseOffset = modelOKinematics.offsetT

        R = np.zeros([3,3])
        R[0][0] = math.cos(aa)
        R[0][1] = -math.sin(aa)
        R[1][0] = math.sin(aa)
        R[1][1] = math.cos(aa)
        R[2][2] = 1

        a1 = jointPositions[0] + modelOKinematics.angleP
        a2 = jointPositions[1] + modelOKinematics.restingDistalAngle
        print(jointPositions[0])
        print(jointPositions[1])
        print(a1, a2)

        f = np.zeros(3)
        f[1] = modelOKinematics.lenP * math.cos(a1) + modelOKinematics.lenD * math.cos(a1 + a2)
        f[2] = modelOKinematics.lenP * math.sin(a1) + modelOKinematics.lenD * math.sin(a1 + a2)

        return f.dot(R) + fingerBaseOffset

    def totalDistErr(self, jointPositions, *contacts):
        '''The joint positions in jointPositions is ordered as ['base_to_prox_t', 'prox_to_distal_t']'''

        position_t = self.forwardKinematics([0, jointPositions[0], jointPositions[1]])

        e_t = np.linalg.norm(contacts[0] - position_t)

        return e_t

    def findJointConfig(self, p_t):

        ret = fmin_l_bfgs_b(self.totalDistErr, x0 = np.array([0, 0]), args=(p_t), bounds=self.jointBounds, approx_grad=True)

        if ret[2]['warnflag'] == 1:
            return None
        else:
            self.updateSystem(ret[0])
            return ret[0]

    def calculateContact(self, jointPositions):
        position_t = self.forwardKinematics([0, jointPositions[0], jointPositions[1]])

        contacts = np.row_stack((position_t))
        return contacts

    def updateSystem(self, jointPositions):
        self.jointPositions = jointPositions
        self.contacts = self.calculateContact(jointPositions)
        self.updateTooltip()

    def systemEnergy(self, jointPositions):
        '''The joint positions in jointPositions is ordered as ['base_to_prox_t', 'prox_to_distal_t']'''
        scp = modelOKinematics.springConstant[0]
        scd = modelOKinematics.springConstant[1]
        energy = scp * (jointPositions[0]**2) + scd * (jointPositions[1]**2)
        return energy / 2.


    def tendonConstraintFuncT(self, jointPositions, *deltaTheta):
        '''The joint positions in jointPositions is ordered as ['base_to_prox_t', 'prox_to_distal_t']'''

        tendon_t = (- self.jointPositions[0]) * modelOKinematics.jointRadii[0]\
                + (jointPositions[0] - self.jointPositions[1]) * modelOKinematics.jointRadii[1] - deltaTheta[0] * modelOKinematics.motorPulleyRadius
        return tendon_t

    def moveMotor(self, deltaTheta):
        '''deltaTheta contains motor displacements for t'''

        cons = ({'type': 'eq', 'fun': self.tendonConstraintFuncT, 'args': (deltaTheta)})

        jointBounds = self.jointBounds
        ret = minimize(self.systemEnergy, self.jointPositions, bounds=jointBounds, constraints=cons, method='SLSQP',  options={'disp': False, 'ftol': 1e-06, 'maxiter': self.optMaxIter})

        return ret

m_o = modelOKinematics()
ret = m_o.moveMotor([0, 0, 0])
m_o.updateSystem(ret.x)
print(m_o.getJointPositions())
print(m_o.getContacts())
