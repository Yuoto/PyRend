import glm
import math
import numpy as np
from enum import Enum

class direction(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4



class Camera():

    def __init__(self,windowSize,focal,center, position=glm.vec3(0, 0, 3), up=glm.vec3(0, 1, 0), cameraFront= glm.vec3(0, 0, -1), yaw=-90, pitch=0,near=0.1,far=100,depthScale=1,movementSpeed=1, mouseSensitivity=0.05, zoom=45):
        self.position = position

        #OpenGL uses right hand system, but the camera is facing at the -Z direction
        self.windowSize = windowSize
        self.cameraUp = up
        self.cameraFront = cameraFront
        self.yaw = yaw
        self.pitch = pitch
        self.movementSpeed = movementSpeed
        self.mouseSensitivity = mouseSensitivity
        self.focal = focal
        self.center = center
        self.far = far
        self.near = near
        self.depthScale = depthScale
        # zoom == fov
        self.zoom = zoom


        self.OpenGLperspective = self.__setOpenGLPerspective()
        self.intrinsic = self.__setIntrinsic()
        self.__updateCamera()

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
        return glm.lookAt(self.position,self.position + self.cameraFront,self.cameraUp)

    def __setOpenGLPerspective(self):
        a = self.focal[0] * self.depthScale
        b = self.focal[1] * self.depthScale
        cx,cy = self.center
        f = self.far
        n = self.near
        OpenGLperspective = np.array([[a/cx,0,0,0],
                         [0,b/cy,0,0],
                         [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],
                         [0,0,-1,0]],dtype=np.float32)
        return OpenGLperspective

    def __updateCamera(self):
        x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        y = math.sin(glm.radians(self.pitch))
        z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.cameraFront = glm.normalize(glm.vec3(x,y,z))
        self.cameraRight = glm.normalize(glm.cross(self.cameraFront,self.cameraUp))
        self.cameraUp = glm.normalize(glm.cross(self.cameraRight,self.cameraFront))

    def Project(self,point):

        #Note that the positive X axis of OpenGL is different from that of OpenCV
        pixels = np.dot(self.OpenGLperspective,[-point[0],point[1],point[2],point[3]])

        #d = np.array([pixels[0]/pixels[2],pixels[1]/pixels[2]])
        c = np.array([(pixels[0]/point[2]+1)*0.5*self.windowSize[0],(pixels[1]/point[2]+1)*0.5*self.windowSize[1]])
        return c

    def ProcessKeyBoard(self, direction, deltaTime):
        velocity = self.movementSpeed * deltaTime
        if direction is direction.FORWARD:
            self.position += self.cameraFront * velocity
        if direction is direction.BACKWARD:
            self.position -= self.cameraFront * velocity
        if direction is direction.LEFT:
            self.position -= self.cameraRight * velocity
        if direction is direction.RIGHT:
            self.position += self.cameraRight * velocity



    def ProcessMouseMovement(self, xoffset, yoffset, constrained = True):

        xoffset *= self.mouseSensitivity
        yoffset *= self.mouseSensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrained:
            if self.pitch > 89:
                self.pitch = 89
            if self.pitch < -89:
                self.pitch = -89

        self.__updateCamera()


