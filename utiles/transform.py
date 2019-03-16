import numpy as np
import math
from enum import Enum




def toExtMat(rotation,translation=None, PoseParameterModel='Eulerzyx',radians=True):
    '''

    :param rotation: 3 rotation parameters in Euler angle (elevataion(x), azimuth(y), in-plane(z))
    :param translation: 3 translation parameters (x,y,z)
    :param PoseParameterModel:
    :param radians:
    :return:
    '''

    if not radians:
        x = math.radians(rotation[0])
        y = math.radians(rotation[1])
        z = math.radians(rotation[2])
    else:
        x = rotation[0]
        y = rotation[1]
        z = rotation[2]

    sx = math.sin(x)
    cx = math.cos(x)
    sy = math.sin(y)
    cy = math.cos(y)
    sz = math.sin(z)
    cz = math.cos(z)

    Rx = np.array([[1,0,0,0],
                   [0,cx,-sx,0],
                   [0,sx,cx,0],
                   [0,0,0,1]])
    Ry = np.array([[cy,0,sy,0],
                   [0,1,0,0],
                   [-sy,0,cy,0],
                   [0,0,0,1]])
    Rz = np.array([[cz, -sz, 0,0],
                   [sz, cz, 0,0],
                   [0, 0, 1,0],
                   [0,0,0,1]])

    if PoseParameterModel == 'Eulerxyz':
        mat = np.dot(np.dot(Rz,Ry),Rx)
    elif  PoseParameterModel == 'Eulerxzy':
        mat = np.dot(np.dot(Ry, Rz), Rx)
    elif PoseParameterModel == 'Euleryxz':
        mat = np.dot(np.dot(Rz, Rx), Ry)
    elif PoseParameterModel == 'Euleryzx':
        mat = np.dot(np.dot(Rx, Rz), Ry)
    elif PoseParameterModel == 'Eulerzxy':
        mat = np.dot(np.dot(Ry, Rx), Rz)
    elif PoseParameterModel == 'Eulerzyx':
        mat = np.dot(np.dot(Rx, Ry), Rz)
    elif PoseParameterModel == 'AxisAngle':
        raise NotImplementedError

    if translation is not None:
        mat[0][3] = translation[0]
        mat[1][3] = translation[1]
        mat[2][3] = translation[2]

    return mat




def SO3toSE3(rot, trans):
    ext = np.zeros((4, 4), dtype=np.float32)
    ext[0:3, 0:3] = rot
    ext[0:3, 3] = trans
    ext[3, 3] = 1.0

    return ext


def toParameters(Ext, PoseParameterModel='Eulerzyx', toRadians=False):
    '''

    :param Ext: SE3 extrinsic matrix
    :param PoseParameterModel: factor as a specific Euler form
    :param toRadians:
    :return: rotation, translation parameters
    '''
    translation = Ext[0:3, 3]

    if PoseParameterModel == 'Eulerzyx':
        r02 = Ext[0][2]
        if r02 < 1:
            if r02 > -1:
                thetaY = math.asin(r02)
                thetaX = math.atan2(-Ext[1][2], Ext[2][2])
                thetaZ = math.atan2(-Ext[0][1], Ext[0][0])
            else:  # r02 = -1
                thetaY = -90
                thetaX = -math.atan2(Ext[1][0], Ext[1][1])
                thetaZ = 0
        else:  # r02 = 1
            thetaY = 90
            thetaX = math.atan2(Ext[1][0], Ext[1][1])
            thetaZ = 0

    if not toRadians:
        thetaX = math.degrees(thetaX)
        thetaY = math.degrees(thetaY)
        thetaZ = math.degrees(thetaZ)

    rotation = [thetaX, thetaY, thetaZ]

    return rotation, translation

def rotatePose(rotation,pose1,radians):

    Ext1 = toExtMat(pose1,None,PoseParameterModel='Eulerzyx',radians=radians)
    Ext2 = toExtMat(rotation,None,PoseParameterModel='Eulerzyx',radians=radians)

    return np.dot(Ext2,Ext1)


def setTranslation(Ext,translation):
    Ext[0:3,3] = translation
    return Ext
