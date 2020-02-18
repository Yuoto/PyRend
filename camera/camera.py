import numpy as np


class Camera():
    def __init__(self, windowSize, focal, center, near=0.01, far=100, depthScale=1):
        '''

        :param windowSize: [x,y] size of the window
        :param focal: [fx,fy] focal length
        :param center: [cx,cy] principle point
        :param near: near plane
        :param far: far plane
        :param depthScale:
        '''

        # OpenGL uses right hand system, but the camera is facing at the -Z direction
        self.windowSize = windowSize
        self.focal = focal
        self.center = center
        self.far = far
        self.near = near
        self.depthScale = depthScale
        self.OpenGLperspective = self.__setOpenGLPerspective()
        self.intrinsic = self.__setIntrinsic()

    def setIntrinsic(self, intrin):
        self.intrinsic = intrin
        self.focal = (intrin[0][0],intrin[1][1])
        self.center = (intrin[0][2],intrin[1][2])

        #update OpenGLpersective
        self.OpenGLperspective = self.__setOpenGLPerspective()

    def __setIntrinsic(self):

        a = self.focal[0] * self.depthScale
        b = self.focal[1] * self.depthScale
        cx, cy = self.center
        intrinsic = np.array([[a, 0, cx, 0],
                              [0, b, cy, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)
        return intrinsic

    def GetCameraViewMatrix(self, up, eye, at, inplane, isRadian=True):

        if not isRadian:
            inplane = np.radians(inplane)

        camDir = eye - at
        camDir = camDir / np.linalg.norm(camDir)

        # Rotate up vector by inplane degree
        up = np.dot(camDir,up) * camDir + np.sin(inplane)* np.cross(camDir,up) - np.cos(inplane)*np.cross(camDir,np.cross(camDir,up))

        camRight = np.cross(up, camDir)
        camRight = camRight / np.linalg.norm(camRight)

        camUp = np.cross(camDir, camRight)

        R_inv = np.block([[camRight, 0],
                          [camUp, 0],
                          [camDir, 0],
                          [0, 0, 0, 1]])
        T_inv = np.block([[1, 0, 0, -eye[0]],
                          [0, 1, 0, -eye[1]],
                          [0, 0, 1, -eye[2]],
                          [0, 0, 0, 1]])
        return np.dot(R_inv, T_inv)

    def __setOpenGLPerspective(self):

        fx, fy = self.focal
        cx, cy = self.center
        f = self.far
        n = self.near

        OpenGLperspective = np.array([[fx / cx, 0, 0, 0],
                                      [0, fy / cy, 0, 0],
                                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                                      [0, 0, -1, 0]], dtype=np.float32)


        return OpenGLperspective

    def project(self, points, isOpenCV=False):

        '''

        :param points: 4 x N (x,y,z,w) in homogeneous coordinate of the camera
        :param isOpenCV: if rendered with OpenCV intrinsic matrix (if not, then near & far plane is considered)
        :param invertX: if invert the x axis during projection
        :return: (x,y) in screen coordinate
        '''

        convert_xz = np.eye(4)
        convert_xz[0, 0] = -1
        convert_xz[1, 1] = 1
        convert_xz[2, 2] = -1

        # OpenCV Method
        if isOpenCV:
            pix_pts = np.dot(mcam1.intrinsic, np.dot(convert_xz,points))
            pix_pts = (pix_pts[:2, :] / pix_pts[2, :] + 1) * 0.5 * np.array(
                [[self.windowSize[0]], [self.windowSize[1]]])
            pix_pts = pix_pts.astype(np.int32).T

        # OpenGL Method
        else:
            pix_pts = np.dot(self.OpenGLperspective, np.dot(convert_xz,points))
            pix_pts = (pix_pts[:2, :] / pix_pts[2, :] + 1) * 0.5 * np.array(
                [[self.windowSize[0]], [self.windowSize[1]]])
            pix_pts = pix_pts.astype(np.int32).T
        return pix_pts
