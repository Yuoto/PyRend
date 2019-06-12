import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glfw
from OpenGL.GL import *
from ctypes import  sizeof, c_void_p,c_float
import numpy as np
from shader import Shader
from _model import Model



# initialize glfw
def init_glfw():
    if not glfw.init():
        print('Failed to initialize GLFW')
        return

    # configuring glfw
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)




float_size = sizeof(c_float)





class Light:
    def __init__(self, lightPos=[0,0,3.0], lightColor=[1.0,1.0,1.0], lightStrength=0.5, lightConstant=1, lightLinear=0.0014, lightQuadratic=0.000007, Directional=True, Attenuation = True):
        '''

        :param lightPos: translation of the light
        :param lightColor: [r,g,b] color of the light, with a default value of [1.0, 1.0, 1.0] representing white light

        credit to: https://learnopengl.com/Lighting/Materials
        :param lightStrength: Strength of the light (independent of the model's material), ranged from 0~1
        :param lightConstant:
        :param lightLinear:
        :param lightQuadratic:
        :param Directional: Enable Directional light
        :param Attenuation: Enable Attenuation
        '''
        self.lightPos = lightPos
        self.lightColor = lightColor
        self.lightConstant = lightConstant
        self.lightLinear = lightLinear
        self.lightQuadratic = lightQuadratic
        self.enableDirectional = Directional
        self.enableAttenuation = Attenuation
        self.lightStrength = lightStrength

    def setPos(self,Pos):
        self.lightPos = Pos

    def setConstant(self,lightConstant):
        self.lightConstant = lightConstant

    def setLinear(self,lightLinear):
        self.lightLinear = lightLinear

    def setQuadratic(self,lightQuadratic):
        self.lightQuadratic = lightQuadratic

    def setColor(self,color):
        self.lightColor = color

    def setStrength(self,strength):
        self.lightStrength = strength

    def setDirectional(self, Directional):
        self.enableDirectional = Directional

    def setAttenuation(self, Attenuation):
        self.enableAttenuation = Attenuation


class Window:
    def __init__(self, windowSize, windowName):
        self.windowSize = windowSize
        self.window = self.__setUpWindow(self.windowSize, windowName)
    def __setUpWindow(self,windowSize,name):
        # -------- setting window

        init_glfw()
        window = glfw.create_window(windowSize[0], windowSize[1], name, None, None)
        if not window:
            glfw.terminate()
            return
        glfw.make_context_current(window)

        return window


    def processInput(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) is glfw.PRESS:
            glfw.set_window_should_close(self.window, True)


    def clearWindow(self, color, alpha=1):
        glClearColor(color[0], color[1], color[2], alpha)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    def updateWindow(self):
        # swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfw.swap_buffers(self.window)
        glfw.poll_events()



class Renderer:
    def __init__(self,light,camera,window, modelPath, vShaderPath,fShaderPath,vShaderLampPath,fShaderLampPath,vShaderTightBoxPath,fShaderTightBoxPath):

        self.camera = camera
        self.window = window
        self.light = light
        self.__modelShader = Shader(vShaderPath,fShaderPath)
        self.__lampShader = Shader(vShaderLampPath, fShaderLampPath)
        self.__tightBoxShader = Shader(vShaderTightBoxPath, fShaderTightBoxPath)
        self.__3dModel = Model(modelPath)
        self.__lampVertices = np.array([-0.5, -0.5, -0.5, 0, 0, -1, 0, 0,
                      0.5, -0.5, -0.5, 0, 0, -1, 1, 0,
                      0.5, 0.5, -0.5, 0, 0, -1, 1, 1,
                      0.5, 0.5, -0.5, 0, 0, -1, 1, 1,
                      -0.5, 0.5, -0.5, 0, 0, -1, 0, 1,
                      -0.5, -0.5, -0.5, 0, 0, -1, 0, 0,

                      -0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
                      0.5, -0.5, 0.5, 0, 0, 1, 1, 0,
                      0.5, 0.5, 0.5, 0, 0, 1, 1, 1,
                      0.5, 0.5, 0.5, 0, 0, 1, 1, 1,
                      -0.5, 0.5, 0.5, 0, 0, 1, 0, 1,
                      -0.5, -0.5, 0.5, 0, 0, 1, 0, 0,

                      -0.5, 0.5, 0.5, -1, 0, 0, 1, 0,
                      -0.5, 0.5, -0.5, -1, 0, 0, 1, 1,
                      -0.5, -0.5, -0.5, -1, 0, 0, 0, 1,
                      -0.5, -0.5, -0.5, -1, 0, 0, 0, 1,
                      -0.5, -0.5, 0.5, -1, 0, 0, 0, 0,
                      -0.5, 0.5, 0.5, -1, 0, 0, 1, 0,

                      0.5, 0.5, 0.5, 1, 0, 0, 1, 0,
                      0.5, 0.5, -0.5, 1, 0, 0, 1, 1,
                      0.5, -0.5, -0.5, 1, 0, 0, 0, 1,
                      0.5, -0.5, -0.5, 1, 0, 0, 0, 1,
                      0.5, -0.5, 0.5, 1, 0, 0, 0, 0,
                      0.5, 0.5, 0.5, 1, 0, 0, 1, 0,

                      -0.5, -0.5, -0.5, 0, -1, 0, 0, 1,
                      0.5, -0.5, -0.5, 0, -1, 0, 1, 1,
                      0.5, -0.5, 0.5, 0, -1, 0, 1, 0,
                      0.5, -0.5, 0.5, 0, -1, 0, 1, 0,
                      -0.5, -0.5, 0.5, 0, -1, 0, 0, 0,
                      -0.5, -0.5, -0.5, 0, -1, 0, 0, 1,

                      -0.5, 0.5, -0.5, 0, 1, 0, 0, 1,
                      0.5, 0.5, -0.5, 0, 1, 0, 1, 1,
                      0.5, 0.5, 0.5, 0, 1, 0, 1, 0,
                      0.5, 0.5, 0.5, 0, 1, 0, 1, 0,
                      -0.5, 0.5, 0.5, 0, 1, 0, 0, 0,
                      -0.5, 0.5, -0.5, 0, 1, 0, 0, 1
                      ], dtype=np.float32)
        self.__3DTightBoxVertices = self.get3DTightBox()
        self.__vboTightBox, self.__vaoTightBox = self.__setUp3DTightBox()
        self.__vboLamp, self.__vaoLamp = self.__setUpLamp(self.__lampVertices)
        self.__setUpBlending()
        self.updateLight()

    def __setUpBlending(self):
        # Enable depth test and blend
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def __setUp3DTightBox(self):
        vbo_TightBox = glGenBuffers(1)
        vao_TightBox = glGenVertexArrays(1)

        vertices = np.array([self.__3DTightBoxVertices[0][0],self.__3DTightBoxVertices[0][1],self.__3DTightBoxVertices[0][2]
            , self.__3DTightBoxVertices[1][0], self.__3DTightBoxVertices[1][1], self.__3DTightBoxVertices[1][2]
            , self.__3DTightBoxVertices[1][0], self.__3DTightBoxVertices[1][1], self.__3DTightBoxVertices[1][2]
            , self.__3DTightBoxVertices[3][0], self.__3DTightBoxVertices[3][1], self.__3DTightBoxVertices[3][2]
            , self.__3DTightBoxVertices[3][0], self.__3DTightBoxVertices[3][1], self.__3DTightBoxVertices[3][2]
            , self.__3DTightBoxVertices[2][0], self.__3DTightBoxVertices[2][1], self.__3DTightBoxVertices[2][2]
            , self.__3DTightBoxVertices[2][0], self.__3DTightBoxVertices[2][1], self.__3DTightBoxVertices[2][2]
            , self.__3DTightBoxVertices[0][0], self.__3DTightBoxVertices[0][1], self.__3DTightBoxVertices[0][2]
            , self.__3DTightBoxVertices[4][0], self.__3DTightBoxVertices[4][1], self.__3DTightBoxVertices[4][2]
            , self.__3DTightBoxVertices[5][0], self.__3DTightBoxVertices[5][1], self.__3DTightBoxVertices[5][2]
            , self.__3DTightBoxVertices[5][0], self.__3DTightBoxVertices[5][1], self.__3DTightBoxVertices[5][2]
            , self.__3DTightBoxVertices[7][0], self.__3DTightBoxVertices[7][1], self.__3DTightBoxVertices[7][2]
            , self.__3DTightBoxVertices[7][0], self.__3DTightBoxVertices[7][1], self.__3DTightBoxVertices[7][2]
            , self.__3DTightBoxVertices[6][0], self.__3DTightBoxVertices[6][1], self.__3DTightBoxVertices[6][2]
            , self.__3DTightBoxVertices[6][0], self.__3DTightBoxVertices[6][1], self.__3DTightBoxVertices[6][2]
            , self.__3DTightBoxVertices[4][0], self.__3DTightBoxVertices[4][1], self.__3DTightBoxVertices[4][2]
            , self.__3DTightBoxVertices[5][0], self.__3DTightBoxVertices[5][1], self.__3DTightBoxVertices[5][2]
            , self.__3DTightBoxVertices[1][0], self.__3DTightBoxVertices[1][1], self.__3DTightBoxVertices[1][2]
            , self.__3DTightBoxVertices[7][0], self.__3DTightBoxVertices[7][1], self.__3DTightBoxVertices[7][2]
            , self.__3DTightBoxVertices[3][0], self.__3DTightBoxVertices[3][1], self.__3DTightBoxVertices[3][2]
            , self.__3DTightBoxVertices[4][0], self.__3DTightBoxVertices[4][1], self.__3DTightBoxVertices[4][2]
            , self.__3DTightBoxVertices[0][0], self.__3DTightBoxVertices[0][1], self.__3DTightBoxVertices[0][2]
            , self.__3DTightBoxVertices[6][0], self.__3DTightBoxVertices[6][1], self.__3DTightBoxVertices[6][2]
            , self.__3DTightBoxVertices[2][0], self.__3DTightBoxVertices[2][1], self.__3DTightBoxVertices[2][2]],dtype=np.float32)

        glBindVertexArray(vao_TightBox)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_TightBox)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * float_size, c_void_p(0))
        glEnableVertexAttribArray(0)

        return vbo_TightBox, vao_TightBox

    def __setUpLamp(self,lamp_vertices):
        # Buffer setting for Lamp
        vbo_lamp = glGenBuffers(1)
        vao_lamp = glGenVertexArrays(1)
        glBindVertexArray(vao_lamp)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_lamp)
        glBufferData(GL_ARRAY_BUFFER, lamp_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 8 * float_size, c_void_p(0))
        glEnableVertexAttribArray(0)

        return vbo_lamp, vao_lamp

    def get3DTightBox(self):
        _3DBox = np.zeros([8, 3], dtype=np.float32)
        # convert to OpenGL left hand coordinate system
        _3DBox[0] = [self.__3dModel.Xmin, self.__3dModel.Ymin, self.__3dModel.Zmin]
        _3DBox[1] = [self.__3dModel.Xmin, self.__3dModel.Ymin, self.__3dModel.Zmax]
        _3DBox[2] = [self.__3dModel.Xmin, self.__3dModel.Ymax, self.__3dModel.Zmin]
        _3DBox[3] = [self.__3dModel.Xmin, self.__3dModel.Ymax, self.__3dModel.Zmax]
        _3DBox[4] = [self.__3dModel.Xmax, self.__3dModel.Ymin, self.__3dModel.Zmin]
        _3DBox[5] = [self.__3dModel.Xmax, self.__3dModel.Ymin, self.__3dModel.Zmax]
        _3DBox[6] = [self.__3dModel.Xmax, self.__3dModel.Ymax, self.__3dModel.Zmin]
        _3DBox[7] = [self.__3dModel.Xmax, self.__3dModel.Ymax, self.__3dModel.Zmax]

        return _3DBox

    def updateLight(self):
        # set the light shader
        self.__modelShader.use()
        self.__modelShader.setBool('light.enableDirectional', self.light.enableDirectional)
        self.__modelShader.setBool('light.enableAttenuation', self.light.enableAttenuation)

        self.__modelShader.setVec3('light.position', self.light.lightPos)
        self.__modelShader.setFloat('light.constant', self.light.lightConstant)
        self.__modelShader.setFloat('light.linear', self.light.lightLinear)
        self.__modelShader.setFloat('light.quadratic', self.light.lightQuadratic)
        self.__modelShader.setFloat('light.strength', self.light.lightStrength)
        self.__modelShader.setVec3('light.color', self.light.lightColor)





    def __setModelPose(self, modelMat, extrinsicMat):
        self.__modelShader.use()
        self.__modelShader.setMat4('intrinsic', self.camera.OpenGLperspective)
        self.__modelShader.setMat4('extrinsic', extrinsicMat)
        self.__modelShader.setMat4('model', modelMat)

    def __drawLamp(self, extrinsicMat):
        self.__lampShader.use()
        self.__lampShader.setVec3('color',self.light.lightColor)
        self.__lampShader.setMat4('intrinsic', self.camera.OpenGLperspective)
        self.__lampShader.setMat4('extrinsic', extrinsicMat)
        model = np.eye(4)
        self.__lampShader.setMat4('model', model)

        glBindVertexArray(self.__vaoLamp)
        glDrawArrays(GL_TRIANGLES, 0, 36)

    def __nonLinearDepth2Linear(self,depth):
        f = self.camera.far
        n = self.camera.near
        ndc_depth = 2*depth-1

        return ((2.0 * n * f) / (f + n - ndc_depth * (f - n))* 1000).astype(np.uint16)



    def draw(self, model, modelExtrinsic,lightExtrinsic,drawLamp=True, drawBox=False, color=[255,255,255], linearDepth=False):
        """

        :param model: model matrix (model space transformation matrix)
        :param modelExtrinsic: Extrinsic matrix of the model
        :param lightExtrinsic: Extrinsic matrix of the light source
        :param drawLamp: enable lamp drawing
        :param drawBox: enable box drawing
        :param color: color of the box
        :param linearDepth: enable linear depth (which is a must for correct groundtruth)
        :return: rgb, depth
        """

        self.__setModelPose(model,modelExtrinsic)
        self.__3dModel.draw(self.__modelShader)
        if drawLamp:
            self.__drawLamp(lightExtrinsic)

        if drawBox:
            self.__drawBox(model, modelExtrinsic, color)

        self.window.updateWindow()


        depth = glReadPixels(0, 0, self.window.windowSize[0], self.window.windowSize[1], GL_DEPTH_COMPONENT, GL_FLOAT)
        depth = np.flipud(depth.reshape(self.window.windowSize[::-1]))


        imageBuf = glReadPixels(0, 0, self.window.windowSize[0], self.window.windowSize[1], GL_RGB, GL_UNSIGNED_BYTE)
        im = np.fromstring(imageBuf, np.uint8)
        rgb = np.flipud(np.reshape(im, (self.window.windowSize[1], self.window.windowSize[0], 3)))
        mask = (rgb[:, :, 0] != 0) | (rgb[:, :, 1] != 0) | (rgb[:, :, 2] != 0)
        if linearDepth:
            depth = mask*self.__nonLinearDepth2Linear(depth)
        else:
            depth = mask*depth

        return rgb,depth



    def __drawBox(self,model,modelExtrinsic,color):
        self.__tightBoxShader.use()
        self.__tightBoxShader.setMat4('intrinsic', self.camera.OpenGLperspective)
        self.__tightBoxShader.setMat4('extrinsic', modelExtrinsic)
        self.__tightBoxShader.setMat4('model', model)
        self.__tightBoxShader.setVec3('color', color)

        glBindVertexArray(self.__vaoTightBox)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDrawArrays(GL_LINES, 0, 24)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


    def __del__(self):
        glDeleteVertexArrays(1, [self.__vaoLamp])




 # callback for window resize
def framebuffer_size_callback(window,width, height):
    # create viewport
    glViewport(0,0,width,height)








