import numpy as np
import math
from enum import Enum

#TODO: Construct a Pose class
def toExtMat(rotation, translation=None, PoseParameterModel='Eulerzyx', isRadian=True):
    '''

    :param rotation:   rotation parameters in Euler angle [elevataion(x), azimuth(y), in-plane(z)]
                    or  axis vector that is to be rotate around in Axis angle [wx, wy, wz] (L1 norm of this vector represents ceta)
    :param translation: translation parameters [tx, ty, tz]
    :param PoseParameterModel: Either "Euler angle", "Axis angle" or "Quaternion"
    :param isRadian: if the rotation vector is in radian
    :return: Extrinsic matrix (SE3)
    '''

    # Euler Angle
    if PoseParameterModel.find('Euler') >= 0:
        mat = __toRotationMatrix(rot=rotation, trans=translation, poseParm=PoseParameterModel, isRadian=isRadian)
    # Axis Angle
    elif PoseParameterModel == 'AxisAngle':
        mat = Rodrigues(w=rotation, u=translation)
    # Quaternion
    elif PoseParameterModel == 'Quaternion':
        raise NotImplementedError

    return mat


def __toRotationMatrix(rot, trans, poseParm, isRadian):
    """
    Given rotation vector rot and translation vector trans, generate a SE3 matrix
    :param rot: rotation parameters in Euler angle [elevataion(x), azimuth(y), in-plane(z)]
    :param trans:  translation parameters [tx, ty, tz]
    :param poseParm: Either poseParm "Eulerxyz", "Eulerxzy", "Euleryxz", "Euleryzx", "Eulerzxy", "Eulerzyx"
    :param isRadian: if the rotation vector is in radian
    :return: Extrinsic matrix (SE3)
    """
    if not isRadian:
        x = math.radians(rot[0])
        y = math.radians(rot[1])
        z = math.radians(rot[2])
    else:
        x = rot[0]
        y = rot[1]
        z = rot[2]

    sx = math.sin(x)
    cx = math.cos(x)
    sy = math.sin(y)
    cy = math.cos(y)
    sz = math.sin(z)
    cz = math.cos(z)

    Rx = np.array([[1, 0, 0, 0],
                   [0, cx, -sx, 0],
                   [0, sx, cx, 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[cy, 0, sy, 0],
                   [0, 1, 0, 0],
                   [-sy, 0, cy, 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[cz, -sz, 0, 0],
                   [sz, cz, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    # Choose one order of rotation (Euler_3rd_2nd_1st)
    if poseParm == 'Eulerxyz':
        mat = np.dot(np.dot(Rz, Ry), Rx)
    elif poseParm == 'Eulerxzy':
        mat = np.dot(np.dot(Ry, Rz), Rx)
    elif poseParm == 'Euleryxz':
        mat = np.dot(np.dot(Rz, Rx), Ry)
    elif poseParm == 'Euleryzx':
        mat = np.dot(np.dot(Rx, Rz), Ry)
    elif poseParm == 'Eulerzxy':
        mat = np.dot(np.dot(Ry, Rx), Rz)
    elif poseParm == 'Eulerzyx':
        mat = np.dot(np.dot(Rx, Ry), Rz)

    # set translation
    if trans is not None:
        mat[0][3] = trans[0]
        mat[1][3] = trans[1]
        mat[2][3] = trans[2]

    return mat


def Rodrigues(w, u):
    '''
    Given axis vector w and translation vector u, generate a SE3 matrix
    :param w:  generator coefficients of Lie algebra [wx, wy, wz]
    :param u: translation vector [tx, ty, tz]
    :return: SE3 matrix (Lie group)
    '''
    if not np.any(u):
        u = np.zeros((3, 1))
    Wx = np.array([[0, -w[2], w[1]],
                   [w[2], 0, - w[0]],
                   [-w[1], w[0], 0]])
    ceta = math.sqrt(np.dot(w, np.transpose(w)))
    ceta_sq = ceta * ceta
    A = math.sin(ceta) / ceta
    B = (1 - math.cos(ceta)) / (ceta_sq)
    C = (1 - A) / ceta_sq

    R = np.eye(3) + A * Wx + B * np.dot(Wx, Wx)
    V = np.eye(3) + B * Wx + C * np.dot(Wx, Wx)

    # to SE3 matrix (Lie group)
    mat = np.block([[R, np.dot(V, np.reshape(u,(3,1)))],
                    [np.zeros(3), 1]])

    return mat


def se3toSE3(se3):
    """
     Transform se3 (Lie algebra) to SE3 (Lie group) using exponential map (Rodrigues formula)
    :param se3: se3 matrix (Lie algebra)
    :return: SE3: SE3 matrix (Lie group)
    """
    w = np.array([se3[2][1], se3[0][2], se3[1][0]])
    u = np.array([se3[0][3], se3[1][3], se3[2][3]])

    # perform exponential map using Rodrigues formula
    return Rodrigues(w, u)


def SE3tose3(SE3):
    """
    Transform SE3 (Lie group) to se3 (Lie algebra) using inverse exponential map
    :param SE3:
    :return: se3 matrix (Lie algebra)
    """
    t = np.reshape(SE3[0:3,3],(3,1))  # to 3x1
    R = SE3[0:3,0:3]
    ceta = math.acos((np.trace(R) - 1) / 2)
    ceta_sq = ceta * ceta
    A = math.sin(ceta) / ceta
    B = (1 - math.cos(ceta)) / ceta_sq

    # inverse exponential map of R = exp(Wx):
    #   Wx = ln(R); R = exp(Wx)
    Wx = (ceta / (2 * math.sin(ceta))) * (R - np.transpose(R))

    V_inv = np.eye(3) - 0.5 * Wx + (1 / ceta_sq) * (1 - A / (2 * B)) * np.dot(Wx, Wx)
    u = np.dot(V_inv, t)  # 3x1

    # to se3 matrix (Lie algebra), recall that se3[3][3] is zero
    mat = np.block([[Wx, u],
                    [np.zeros(3), 0]])

    return mat

def SO3toSE3(rot, trans):
    """
    Given rotation vector rot and translation vector trans, generate a SO3 matrix
    :param rot: rotation parameters in Euler angle [elevataion(x), azimuth(y), in-plane(z)]
    :param trans:  translation parameters [tx, ty, tz]
    :return: SE3 matrix (Lie group)
    """
    ext = np.zeros((4, 4), dtype=np.float32)
    ext[0:3, 0:3] = rot
    ext[0:3, 3] = trans
    ext[3, 3] = 1.0

    return ext


def toParameters(Ext, PoseParameterModel='Eulerzyx', toRadian=False):
    '''
    Factor out pose parameters according to the specified pose model
    :param Ext: SE3 extrinsic matrix
    :param PoseParameterModel: factor as a specific Euler form, others are currently not implemented
    :param toRadian:
    :return: rotation (or axis vector in case of Axis angle pose parameter), translation parameters
    '''

    # Euler Angle
    if PoseParameterModel.find('Euler') >= 0:
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

        if not toRadian:
            thetaX = math.degrees(thetaX)
            thetaY = math.degrees(thetaY)
            thetaZ = math.degrees(thetaZ)

        rotation = [thetaX, thetaY, thetaZ]

    # Axis Angle
    elif PoseParameterModel == 'AxisAngle':
        se3 = SE3tose3(Ext)
        w, u = [se3[2][1], se3[0][2], se3[1][0]], se3[0:3,3]
        rotation, translation =  w, u
    elif PoseParameterModel == 'Quaternion':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return rotation, translation


#TODO: merge to Pose class
def rotatePose(rotation, pose1, radians):
    Ext1 = toExtMat(pose1, None, PoseParameterModel='Eulerzyx', radians=radians)
    Ext2 = toExtMat(rotation, None, PoseParameterModel='Eulerzyx', radians=radians)

    return np.dot(Ext2, Ext1)


#TODO: merge to Pose class
def setTranslation(Ext, translation):
    Ext[0:3, 3] = translation
    return Ext
