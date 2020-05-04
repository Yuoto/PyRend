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
        self.distCoeff = None

    def setDistCoeff(self, distCoeff):
        self.distCoeff = distCoeff


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

        # ==== 1. Ideal symmetry camera
        # Reference: http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
        '''
        OpenGLperspective = np.array([[fx / cx, 0, 0, 0],
                                      [0, fy / cy, 0, 0],
                                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                                      [0, 0, -1, 0]], dtype=np.float32)
             '''

        # ==== 2. asymmetry camera (with OpenCV calibrated parameters)
        # https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
        OpenGLperspective = np.array([[2 * fx / self.windowSize[0], 0, 1 - 2 * cx / self.windowSize[0], 0],
                                      [0, 2 * fy / self.windowSize[1], 2 * cy / self.windowSize[1] - 1, 0],
                                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                                      [0, 0, -1, 0]], dtype=np.float32)

        return OpenGLperspective

    def project(self, points):
        """

        :param points: Nx3 or Nx4 homogeneous coordinate
        :return: pixels
        """
        cloud = points.copy()
        # Since we are using openGL coordinate system, the camera is looking at the -Z axis, and the right hand side is +X axis, the up is +Y axis
        #================================================================================#
        #      +Y        #              #      -Y        #              #      -Y        #
        #       |__ +x   # divided      #       |__ -x   # invert       #       |__ +x   #
        #      /         # by      -->  #      /         # x axis  -->  #      /         #
        #    +z          # negative     #    +z          #              #    +z          #
        #  eye(cam)      # depth        #  eye(cam)      #              #  eye(cam)      #
        #================================================================================#
        #
        # This makes our depth < 0, and thus inverting the x & y axis when dividing the depth value (when multiplying intrinsic matrix)
        # However, when storing image to array, the x index increases to the right and y index increases downward, thus a change in x axis is needed
        # Add negation to y,z or add negation to x (camera coordinate)

        cloud[:, 0] = -cloud[:, 0]
        #or
        #cloud[:,1] = -cloud[:,1]
        #cloud[:, 2] = -cloud[:, 2]

        computed_pixels = np.zeros((cloud.shape[0], 2))
        computed_pixels[:, 0] = cloud[:, 0] * self.focal[0] / cloud[:, 2] + self.center[0]
        computed_pixels[:, 1] = cloud[:, 1] * self.focal[1] / cloud[:, 2] + self.center[1]
        return np.round(computed_pixels).astype(np.int)

    def backproject(self, depth, mask):
        """

        :param depth:  m x n depth map
        :param mask:  m x n mask
        :return:
        """

        constant_x = 1.0 / self.focal[0]
        constant_y = 1.0 / self.focal[1]

        row, col = depth.shape
        coords = np.zeros((row, col, 2), dtype=np.uint)
        coords[..., 0] = np.arange(row)[:, None]
        coords[..., 1] = np.arange(col)
        coords = coords[mask]
        coords = coords.reshape((-1, 2))

        output = np.zeros((len(coords), 3))
        values = depth[coords[:, 0], coords[:, 1]]
        # Since this is the inverse of project function, also the X axis has to be negated
        output[:, 0] = -(coords[:, 1] - self.center[0]) * values * constant_x
        output[:, 1] = (coords[:, 0] - self.center[1]) * values * constant_y
        output[:, 2] = values

        return output

    def backproject_points(self, depth, coords):
        """

        :param depth: N x 1 depth value
        :param coords: Nx3 or Nx4 homogeneous coordinate
        :return: output: Nx3 camera coodinate
        """
        constant_x = 1.0 / self.focal[0]
        constant_y = 1.0 / self.focal[1]

        output = np.zeros((len(coords), 3))
        # Since this is the inverse of project function, also the X axis has to be negated
        output[:, 0] = -(coords[:, 0] - self.center[0]) * depth * constant_x
        output[:, 1] = (coords[:, 1] - self.center[1]) * depth * constant_y
        output[:, 2] = depth
        return output

    '''
    def project(self, points, isOpenCV=False):

        """

        :param points: 4 x N (x,y,z,w) in homogeneous coordinate of the camera
        :param isOpenCV: if rendered with OpenCV intrinsic matrix (if not, then near & far plane is considered)
        :param invertX: if invert the x axis during projection
        :return: (x,y) in screen coordinate
        """

        convert_xz = np.eye(4)
        convert_xz[0, 0] = -1


        # OpenCV Method
        if isOpenCV:
            pix_pts = np.dot(self.intrinsic, np.dot(convert_xz, points))
            pix_pts = pix_pts[:2, :] / pix_pts[2, :]
            pix_pts = pix_pts.astype(np.int32).T

        # OpenGL Method
        else:
            convert_xz[2, 2] = -1
            pix_pts = np.dot(self.OpenGLperspective, np.dot(convert_xz,points))
            pix_pts = (pix_pts[:2, :] / pix_pts[2, :] + 1) * 0.5 * np.array(
                [[self.windowSize[0]], [self.windowSize[1]]])
            pix_pts = pix_pts.astype(np.int32).T
        return pix_pts

    def toCamCord(self, mask, depth, isOpenCV=False):
        """

        :param mask:
        :param depth:
        :param isOpenCV:
        :return:
        """
        ymap = np.array([[j for i in range(self.windowSize[0])] for j in range(self.windowSize[1])])
        xmap = np.array([[i for i in range(self.windowSize[0])] for j in range(self.windowSize[1])])
        depth_masked = depth[mask].flatten()[:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[mask].flatten()[:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[mask].flatten()[:, np.newaxis].astype(np.float32)
        pt2 = depth_masked

        # OpenCV Method
        if isOpenCV:
            pt0 = (xmap_masked - mcam1.center[0]) * pt2 / mcam1.focal[0]
            pt1 = (ymap_masked - mcam1.center[1]) * pt2 / mcam1.focal[1]
            pt1 = np.flipud(pt1)
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud[:,0] = -cloud[:,0]


        # OpenGL Method
        else:
            pix = np.concatenate((xmap_masked, ymap_masked, pt2), axis=1)
            pt1 = np.flipud(pt1)
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud[:, 0] = -cloud[:, 0]


            convert_xz[2, 2] = -1
            pixels = np.dot(self.OpenGLperspective, np.dot(convert_xz, points))
            pixels = (pixels[:2, :] / pixels[2, :] + 1) * 0.5 * np.array(
                [[self.windowSize[0]], [self.windowSize[1]]])
            pixels = pixels.astype(np.int32).T
        return cloud
    '''