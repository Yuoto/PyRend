import numpy as np
import math
import cv2

class Pose():
    def __init__(self,rvec, tvec, PoseParamModel='axis', isRadian=True):
        self.PoseParamModel = PoseParamModel
        self.isRadian = isRadian
        self.rvec = rvec
        self.tvec = tvec
        self.convertYZMat = np.array([[1, 0, 0, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0, 0, 1]])
        self.SE3 = self.toSE3()
        self.se3 = self.SE3toParam()
        self.SE3_gl = self.convertYZMat.dot(self.SE3)



    def update(self,rvec=None, tvec=None, PoseParamModel=None):
        if PoseParamModel != None:
            self.PoseParamModel = PoseParamModel
        if np.any(rvec != None):
            self.rvec = rvec
        if np.any(tvec != None):
            self.tvec = tvec

        if  np.all(rvec == None) and  np.all(tvec == None):
            self.rvec = self.se3[:3]
            self.SE3 = self.__axixToSE3(hasUvec=True)
            self.SE3_gl = self.convertYZMat.dot(self.SE3)
        else:
            self.SE3 = self.toSE3()
            self.se3 = self.__SE3Tose3()
            self.SE3 = self.__axixToSE3(hasUvec=True)
            self.SE3_gl = self.convertYZMat.dot(self.SE3)


    def toSE3(self, hasUvec=False):
        # Euler Angle
        if self.PoseParamModel.find('euler') >= 0:
           return self.__eulerToSE3()
        # Axis Angle
        elif self.PoseParamModel == 'axis':
            return self.__axixToSE3(hasUvec)
        # Quaternion
        elif self.PoseParamModel == 'quaternion':
            raise NotImplementedError

    '''
    def toSO3(self):
        if self.PoseParamModel.find('euler') >= 0:
           return self.__eulerToSO3()
        # Axis Angle
        elif self.PoseParamModel == 'axis':
            return self.__axixToSO3()
        # Quaternion
        elif self.PoseParamModel == 'quaternion':
            raise NotImplementedError
    
    def __eulerToSO3(self):
        # Check if the angle is converted to radians
        rvec = self.rvec if self.isRadian else np.radians(self.rvec)
        sx, sy, sz = np.sin(rvec)
        cx, cy, cz = np.cos(rvec)

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

        if poseParm == 'eulerxyz':
            mat = np.dot(np.dot(Rz, Ry), Rx)
        elif poseParm == 'eulerxzy':
            mat = np.dot(np.dot(Ry, Rz), Rx)
        elif poseParm == 'euleryxz':
            mat = np.dot(np.dot(Rz, Rx), Ry)
        elif poseParm == 'euleryzx':
            mat = np.dot(np.dot(Rx, Rz), Ry)
        elif poseParm == 'eulerzxy':
            mat = np.dot(np.dot(Ry, Rx), Rz)
        elif poseParm == 'eulerzyx':
            mat = np.dot(np.dot(Rx, Ry), Rz)
        self.SO3 = mat
        return mat
    '''
    def __eulerToSE3(self):
        mat = np.eye(4)
        mat[:3,:3] = self.SO3
        # set translation
        mat[0:3,3] = self.tvec

        return mat

    '''
    def __axixToSO3(self):
        self.SO3 = self.SE3[:3,:3]
        return self.SO3
    '''
    def __axixToSE3(self, hasUvec=False):
        """
                Calculate SE(3) from a given axis angle vector se(3) = (rvec, uvec)^T
        :return:  SE(3) matrix
        """
        # Check if the angle is converted to radians
        rvec = self.rvec if self.isRadian else np.radians(self.rvec)

        #1. OpenCV method
        SO3 = cv2.Rodrigues(np.array(rvec))[0]

        #2.  R = cos(theta) * I + (1 - cos(theta)) * r * rT + sin(theta) * [r_x]
        theta = np.linalg.norm(rvec)
        itheta = 1. / theta if theta else 0.
        r = rvec * itheta
        r_x = self.__skew(r)
        c = np.cos(theta)
        s = np.sin(theta)
        rrt = np.array([[r[0] * r[0], r[1] * r[0], r[2] * r[0]],
                        [r[0] * r[1], r[1] * r[1], r[2] * r[1]],
                        [r[0] * r[2], r[1] * r[2], r[2] * r[2]]])
        self.SO3 = c * np.eye(3) + (1 - c) * rrt + s * r_x
        self.SO3_gl = self.convertYZMat[:3,:3].dot(self.SO3)

        # debug
        '''
        theta = np.arccos((np.trace(self.SO3) - 1) / 2)
        sincInv = theta/(2*np.sin(theta)) if theta else 0.5
        lnR = sincInv * (self.SO3 - self.SO3.T)
        self.so3 = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])

        itheta = 1. / theta if theta else 0.
        r = self.so3 * itheta
        r_x = self.__skew(r)
        rrt = np.array([[r[0] * r[0], r[1] * r[0], r[2] * r[0]],
                        [r[0] * r[1], r[1] * r[1], r[2] * r[1]],
                        [r[0] * r[2], r[1] * r[2], r[2] * r[2]]])
        # J_inv =(theta/2)cot(theta/2) I + (1-(theta/2)cot(theta/2))r*rT - (theta/2)[r_x]
        thetaHalf = theta / 2
        cotthetaH = thetaHalf / np.tan(thetaHalf) if theta else 1.
        J_inv = cotthetaH * np.eye(3) + (1 - cotthetaH) * rrt - thetaHalf * r_x
        '''

        if hasUvec:
            sinc = np.sin(theta)/theta if theta else 1.
            cosc = (1-np.cos(theta))/theta if theta else 0.
            #J = sin(theta)/theta I + (1-sin(theta)/theta)rrT + (1-cos(theta))/theta * [r_x]
            J = sinc* np.eye(3) + (1-sinc)*rrt + cosc*r_x
            #a = np.dot(J,J_inv)
            self.tvec = J.dot(self.se3[3:6])

        mat = np.eye(4)
        mat[:3, :3] = self.SO3
        mat[:3, 3] = self.tvec

        return mat
    '''
    def SO3toParam(self):
        if self.PoseParamModel.find('euler') >= 0:
            return self.__SO3ToEuler()
            # Axis Angle
        elif self.PoseParamModel == 'axis':
            return self.__SO3Toso3()
            # Quaternion
        elif self.PoseParamModel == 'quaternion':
            raise NotImplementedError
        '''
    def SE3toParam(self):
        if self.PoseParamModel.find('euler') >= 0:
            return self.__SE3ToEuler()
            # Axis Angle
        elif self.PoseParamModel == 'axis':
            return self.__SE3Tose3()
            # Quaternion
        elif self.PoseParamModel == 'quaternion':
            raise NotImplementedError
    '''
    def __SO3ToEuler(self):
        if self.PoseParamModel  == 'eulerzyx':
            r02 = self.SO3[0][2]
            if r02 < 1:
                if r02 > -1:
                    thetaY = math.asin(r02)
                    thetaX = math.atan2(-self.SO3[1][2], self.SO3[2][2])
                    thetaZ = math.atan2(-self.SO3[0][1], self.SO3[0][0])
                else:  # r02 = -1
                    thetaY = -90
                    thetaX = -math.atan2(self.SO3[1][0], self.SO3[1][1])
                    thetaZ = 0
            else:  # r02 = 1
                thetaY = 90
                thetaX = math.atan2(self.SO3[1][0], self.SO3[1][1])
                thetaZ = 0
        else:
            print('Decomposing euler angle to format other than zyx is currently not implemented!')
            raise NotImplementedError

        return np.array([thetaX, thetaY, thetaZ])
   
    def __SE3ToEuler(self):
        rvec = self.__SO3ToEuler()
        tvec = self.SE3[:,3]
        return np.hstack((rvec, tvec))
     '''
    '''
    def __SO3Toso3(self):
        ceta = arccos(np.trace(self.SO3) - 1) / 2
        lnR = 0.5*ceta/np.sin(ceta)*(self.SO3-self.SO3.T)
        self.so3 = np.array([lnR[1,2], lnR[2,0], lnR[0,1]])
        return self.so3
    '''
    def __skew(self, x):
        return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

    def __SE3Tose3(self):
        """
            Calculate se(3) = [ rvec, uvec] given SE(3)
        :return:
        """
        theta = np.arccos((np.trace(self.SO3) - 1) / 2)
        sincInv = theta/(2*np.sin(theta)) if theta else 0.5
        sinc = np.sin(theta)/theta if theta else 1.
        lnR = sincInv * (self.SO3 - self.SO3.T)
        self.so3 = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])

        itheta = 1. / theta if theta else 0.
        r = self.so3 * itheta
        r_x = self.__skew(r)
        rrt = np.array([[r[0] * r[0], r[1] * r[0], r[2] * r[0]],
                        [r[0] * r[1], r[1] * r[1], r[2] * r[1]],
                        [r[0] * r[2], r[1] * r[2], r[2] * r[2]]])

        # J_inv =(theta/2)cot(theta/2) I + (1-(theta/2)cot(theta/2))r*rT - (theta/2)[r_x]
        thetaHalf = theta / 2
        cotthetaH = thetaHalf / np.tan(thetaHalf) if theta else 1.
        J_inv = cotthetaH * np.eye(3) + (1 - cotthetaH) * rrt - thetaHalf * r_x

        self.se3 = np.hstack((self.so3, np.dot(J_inv, self.tvec)))

        return self.se3

    def convertYZ(self):
        self.SO3 = self.convertYZMat[:3,:3].dot(self.SO3)
        self.SE3 = self.convertYZMat.dot(self.SE3)


def toHomo(vectors):
    """

    :param vectors:  N x 3 vectors
    :return:  4 x N vectors
    """

    return np.concatenate((vectors, np.ones((vectors.shape[0], 1))), axis=1).T



'''
def Rodrigues(w, u):
    
    Given axis vector w and translation vector u, generate a SE3 matrix
    :param w:  generator coefficients of Lie algebra [wx, wy, wz]
    :param u: translation vector [tx, ty, tz]
    :return: SE3 matrix (Lie group)

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
'''




#TODO: merge to Pose class
def rotatePose(rotation, pose1, radians):
    Ext1 = toExtMat(pose1, None, PoseParameterModel='Eulerzyx', radians=radians)
    Ext2 = toExtMat(rotation, None, PoseParameterModel='Eulerzyx', radians=radians)

    return np.dot(Ext2, Ext1)


#TODO: merge to Pose class
def setTranslation(Ext, translation):
    Ext[0:3, 3] = translation
    return Ext
