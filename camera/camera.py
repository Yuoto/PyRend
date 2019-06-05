import numpy as np



class Camera():

    def __init__(self,windowSize,focal,center,near=0.1,far=2,depthScale=1):
        '''

        :param windowSize: [x,y] size of the window
        :param focal: [fx,fy] focal length
        :param center: [cx,cy] principle point
        :param near: near plane
        :param far: far plane
        :param depthScale:
        '''

        #OpenGL uses right hand system, but the camera is facing at the -Z direction
        self.windowSize = windowSize
        self.focal = focal
        self.center = center
        self.far = far
        self.near = near
        self.depthScale = depthScale
        self.OpenGLperspective = self.__setOpenGLPerspective()
        self.intrinsic = self.__setIntrinsic()



    def __setIntrinsic(self):

        a = self.focal[0] * self.depthScale
        b = self.focal[1] * self.depthScale
        cx, cy = self.center
        intrinsic = np.array([[a, 0, cx, 0],
                              [0, b, cy, 0],
                              [0, 0,  1, 0],
                              [0, 0,  0, 1]], dtype=np.float32)
        return intrinsic


    def GetCameraViewMatrix(self):
        assert (NotImplemented)

    def __setOpenGLPerspective(self):

        fx,fy = self.focal
        cx,cy = self.center
        f = self.far
        n = self.near

        OpenGLperspective = np.array([[fx/cx,0,0,0],
                         [0,fy/cy,0,0],
                         [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],
                         [0,0,-1,0]],dtype=np.float32)


        return OpenGLperspective



    def Project(self,point, isOpenCV=False):

        '''

        :param point: (x,y,z,z) in homogeneous coordinate of the camera
        :param isOpenCV: if rendered with OpenCV intrinsic matrix
        :return: (x,y) in screen coordinate
        '''

        # OpenCV Method
        if isOpenCV:
            pixels = np.dot(self.intrinsic,[point[0],point[1],point[2],point[3]])
            pixels = np.array([pixels[0] / pixels[2], pixels[1] / pixels[2]])

        # OpenGL Method
        else:
            pixels = np.dot(self.OpenGLperspective, [point[0], point[1], point[2], point[3]])
            pixels = np.array([(pixels[0]/point[2]+1)*0.5*self.windowSize[0],(pixels[1]/point[2]+1)*0.5*self.windowSize[1]])
        return pixels



