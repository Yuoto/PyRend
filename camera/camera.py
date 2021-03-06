import numpy as np
import logging

class Camera():
    def __init__(self, window_size, focal=(None), center=None, distCoeff = None, near=10, far=10000, depthScale=1, coord_system='opengl'):
        '''
                Camera class, given base project/ back project functions
        :param window_size: tuple or list (x,y),  size of the window
        :param focal: tuple or list (fx,fy), focal length
        :param center: tuple or list (cx,cy),  principle point
        :param near: near plane
        :param far: far plane
        :param depthScale:
        '''

        # OpenGL uses right hand system, but the camera is facing at the -Z direction
        self.window_size = window_size
        self.focal = focal
        self.center = center
        self.far = far
        self.near = near
        self.depthScale = depthScale
        self.distCoeff = distCoeff
        self.pos = np.array([0., 0., -1])
        self.cam_speed_const = 2.5
        self.coord_system = coord_system

        if focal == None:
            logging.warning('focal not set!')
        if center == None:
            logging.warning('center not set!')
        if distCoeff == None:
            logging.warning('distortion coefficient not set!')
        if focal != None and center != None:
            self.OpenGLperspective = self.__setOpenGLPerspective()
            self.intrinsic = self.__setIntrinsic()
            logging.warning('intrinsic values not set!')


    def setDistCoeff(self, distCoeff):
        self.distCoeff = distCoeff
        logging.info('distortion coefficient set!')

    def setIntrinsic(self, intrin):
        self.focal = (intrin[0][0],intrin[1][1])
        self.center = (intrin[0][2],intrin[1][2])

        #update OpenGL persective & OpenCV intrinsic values
        self.OpenGLperspective = self.__setOpenGLPerspective()
        self.intrinsic = self.__setIntrinsic()

    def __setIntrinsic(self):
        a = self.focal[0] * self.depthScale
        b = self.focal[1] * self.depthScale
        cx, cy = self.center
        intrinsic = np.array([[a, 0, cx, 0],
                              [0, b, cy, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)
        #logging.info('Intrinsic set!')
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

        if self.coord_system == 'opengl':
            # choose opengl camera coordinate  (-z forward, +y up)
            R_inv = np.block([[camRight, 0],
                              [camUp, 0],
                              [camDir, 0],
                              [0, 0, 0, 1]])
            T_inv = np.block([[1, 0, 0, -eye[0]],
                              [0, 1, 0, -eye[1]],
                              [0, 0, 1, -eye[2]],
                              [0, 0, 0, 1]])
        else:
            raise NotImplementedError
        return R_inv @ T_inv

    def __setOpenGLPerspective(self):
        fx, fy = self.focal
        cx, cy = self.center
        f = self.far
        n = self.near
        w = self.window_size[0]
        h = self.window_size[1]

        # ==== 1. Ideal symmetry camera
        # Reference: http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
        '''
        OpenGLperspective = np.array([[fx / cx, 0, 0, 0],
                                      [0, fy / cy, 0, 0],
                                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                                      [0, 0, -1, 0]], dtype=np.float32)

            '''
        # ==== 2. asymmetry camera (with OpenCV calibrated parameters)
        # # TODO: This uses window_coords='y down', i.e. flips Y first to get the same coordinate with normal image file
        # # https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
        OpenGLperspective = np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0],
                                      [0, 2 * fy / h, 2 * cy / h - 1, 0],
                                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                                      [0, 0, -1, 0]], dtype=np.float32)

        # # TODO: This assumes opengl world coordinate -> -Y up, z forward
        # # ==== 3. asymmetry camera, assuming opengl with the same coordinate as opencv
        # # https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ/22312303
        # OpenGLperspective = np.array([[2 * fx / w, 0, 2 * cx / w-1, 0],
        #                               [0, 2 * fy / h, 2 * cy / h-1, 0],
        #                               [0, 0, (f + n) / (f - n), 2 * f * n / (n - f)],
        #                               [0, 0, 1, 0]], dtype=np.float32)
        #
        # # TODO: This assumes opengl world coordinate -> Y up, -z forward
        # # TODO: This uses window_coords='y up', i.e. upside down with normal image file
        # # ==== 4. asymmetry camera
        # paper in ECCV2016 region based used
        # OpenGLperspective = np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0],
        #                               [0, -2 * fy / h, 1 - 2 * cy / h, 0],
        #                               [0, 0, -(f + n) / (f - n), 2 * f * n / (n - f)],
        #                               [0, 0, -1, 0]], dtype=np.float32)


        #logging.info('OpenGL Perspective set!')

        return OpenGLperspective

    def project(self, points, convertYZ =False):
        """

        :param np.array points: Nx3 or Nx4 homogeneous coordinate
        :return:  Nx2 pixels
        """
        cloud = points.copy()
        if len(cloud.shape) == 3:
            cloud = cloud.squeeze(axis=2)

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
        if convertYZ == True:
            cloud[:,1] = -cloud[:,1]
            cloud[:, 2] = -cloud[:, 2]
        # or
        #cloud[:, 0] = -cloud[:, 0]

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
        output[:, 0] = (coords[:, 1] - self.center[0]) * values * constant_x
        output[:, 1] = (coords[:, 0] - self.center[1]) * values * constant_y
        output[:, 2] = values

        return output

    def backproject_points(self, depth, coords):
        """

        :param depth: N x 1 depth value
        :param coords: N x 2 pixel coordinate
        :return: output: N x 3 camera coordinate
        """
        constant_x = 1.0 / self.focal[0]
        constant_y = 1.0 / self.focal[1]

        output = np.zeros((len(coords), 3))
        output[:, 0] = (coords[:, 0] - self.center[0]) * depth.squeeze() * constant_x
        output[:, 1] = (coords[:, 1] - self.center[1]) * depth.squeeze() * constant_y
        output[:, 2] = depth.squeeze()
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
                [[self.window_size[0]], [self.window_size[1]]])
            pix_pts = pix_pts.astype(np.int32).T
        return pix_pts

    def toCamCord(self, mask, depth, isOpenCV=False):
        """

        :param mask:
        :param depth:
        :param isOpenCV:
        :return:
        """
        ymap = np.array([[j for i in range(self.window_size[0])] for j in range(self.window_size[1])])
        xmap = np.array([[i for i in range(self.window_size[0])] for j in range(self.window_size[1])])
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
                [[self.window_size[0]], [self.window_size[1]]])
            pixels = pixels.astype(np.int32).T
        return cloud
    '''