import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glfw
import OpenGL.GL as gl
from ctypes import  sizeof, c_void_p,c_float
import numpy as np
from shader import Shader
from model import Model
import cv2
if __debug__:
    import copy


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


class Renderer:
    def __init__(self,light,camera,window, modelPath, vShaderPath,fShaderPath,vShaderLampPath,fShaderLampPath,vShaderTightBoxPath,fShaderTightBoxPath):

        self.camera = camera
        self.window = window
        self.light = light
        self.__modelShader = Shader(vShaderPath,fShaderPath)
        self.__lampShader = Shader(vShaderLampPath, fShaderLampPath)
        self.__tightBoxShader = Shader(vShaderTightBoxPath, fShaderTightBoxPath)
        self.__3dModel = Model(modelPath,window)
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
        self.__vboTightBox, self.__vaoTightBox = self.__setup_3D_tightbox()
        self.__vboLamp, self.__vaoLamp = self.__setup_lamp(self.__lampVertices)
        self.setup_blending()
        self.update_light()

    def setup_blending(self, FaceCull=False,Blend=True,DepthFunc='GL_LEQUAL',DepthTest=True,MultiSample=True):
        # Enable depth test and blend
        if Blend:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


        # Due to some depth problem for models in ShapeNet
        if DepthTest:
            gl.glEnable(gl.GL_DEPTH_TEST)
            if DepthFunc == 'GL_LEQUAL':
                gl.glDepthFunc(gl.GL_LEQUAL)
            elif DepthFunc == 'GL_LESS':
                gl.glDepthFunc(gl.GL_LESS)



        # Enable MSAA
        if MultiSample:
            gl.glEnable(gl.GL_MULTISAMPLE)

        # Enable Face Culling
        if FaceCull:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)

    def __setup_3D_tightbox(self):
        vbo_TightBox = gl.glGenBuffers(1)
        vao_TightBox = gl.glGenVertexArrays(1)

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

        gl.glBindVertexArray(vao_TightBox)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_TightBox)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 3 * float_size, c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        return vbo_TightBox, vao_TightBox

    def __setup_lamp(self,lamp_vertices):
        # Buffer setting for Lamp
        vbo_lamp = gl.glGenBuffers(1)
        vao_lamp = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_lamp)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_lamp)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, lamp_vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 8 * float_size, c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        return vbo_lamp, vao_lamp

    def get3DTightBox(self):
        _3DBox = np.zeros([8, 3], dtype=np.float32)
        _3DBox[0] = [self.__3dModel.Xmin, self.__3dModel.Ymin, self.__3dModel.Zmin]
        _3DBox[1] = [self.__3dModel.Xmin, self.__3dModel.Ymin, self.__3dModel.Zmax]
        _3DBox[2] = [self.__3dModel.Xmin, self.__3dModel.Ymax, self.__3dModel.Zmin]
        _3DBox[3] = [self.__3dModel.Xmin, self.__3dModel.Ymax, self.__3dModel.Zmax]
        _3DBox[4] = [self.__3dModel.Xmax, self.__3dModel.Ymin, self.__3dModel.Zmin]
        _3DBox[5] = [self.__3dModel.Xmax, self.__3dModel.Ymin, self.__3dModel.Zmax]
        _3DBox[6] = [self.__3dModel.Xmax, self.__3dModel.Ymax, self.__3dModel.Zmin]
        _3DBox[7] = [self.__3dModel.Xmax, self.__3dModel.Ymax, self.__3dModel.Zmax]

        return _3DBox

    def update_light(self):
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

    @staticmethod
    def draw_pixels(img):
        """
            Render images to the currently binded window

        Parameters
        ----------
        img : array_like, shape(h,w,3)
            image to be rendered with type np.uint8
        Returns
        -------
            None
        """

        #TODO: add viewport to function
        data = np.flipud(img)
        gl.glDrawPixels(data.shape[1], data.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    def __drawLamp(self, extrinsicMat):
        self.__lampShader.use()
        self.__lampShader.setVec3('color',self.light.lightColor)
        self.__lampShader.setMat4('intrinsic', self.camera.OpenGLperspective)
        self.__lampShader.setMat4('extrinsic', extrinsicMat)
        model = np.eye(4)
        self.__lampShader.setMat4('model', model)

        gl.glBindVertexArray(self.__vaoLamp)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)

    def __non_linear_depth_2_linear(self,depth):
        f = self.camera.far
        n = self.camera.near
        ndc_depth = 2*depth-1

        return ((2.0 * n * f) / (f + n - ndc_depth * (f - n)))


    def set_vertex_buffer(self, pos=None, normal=None, color=None, tex=None, indices=np.array([]), mesh_id = None):
        """
        :param vertices: the format for N vertices stored in buffer is in a non-interleaving form:
         [v1.pos,.... vN.pos, v1.normal, ... ,vN.normal, v1.color, ...vN.color, v1.tex, ... vN.tex]
         Note that both position & Normal have 3 channels (x,y,z),(Nx,Ny,Nz), color has 4 channels (r,g,b,a), texture has 2 (u,v).
        :return: None
        """
        self.__3dModel.set_buffer(pos, normal, color, tex, indices, mesh_id)

    def get_vertex_buffer(self, attribute='position', mesh_id = None):
        """

        :param attribute: choose any attribute from position/normal/color/textureCoord/indices
        :param mesh_id: what mesh id to be edit
        :return:
        """
        return self.__3dModel.get_buffer_data(attribute,mesh_id)

    def draw(self, model, modelExtrinsic,lightExtrinsic,drawLamp=True, drawBox=False, meshBymesh=False, color=[255,255,255], linearDepth=False):
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
        self.__3dModel.draw(self.__modelShader, meshBymesh)
        if drawLamp:
            self.__drawLamp(lightExtrinsic)

        if drawBox:
            self.__draw_box(model, modelExtrinsic, color)

        #self.window.updateWindow()


        depth = gl.glReadPixels(0, 0, self.window.window_size[0], self.window.window_size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)

        depth = depth.reshape(self.window.window_size[::-1])
        depth = np.flipud(depth)


        imageBuf = gl.glReadPixels(0, 0, self.window.window_size[0], self.window.window_size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        im = np.fromstring(imageBuf, np.uint8)

        #This is because of the y axis of the image coordinate system and that of the opencv image layout is inverted
        rgb = np.reshape(im, (self.window.window_size[1], self.window.window_size[0], 3))
        rgb = np.flipud(rgb)

        # Since the value from depth buffer contains non-linear depth ~[0,1], need to linearize
        # Apply mask s.t background has depth 0
        # mask = depth < 1
        # mask = np.logical_or(np.logical_or(rgb[...,0]!=0, rgb[...,1]!=0),rgb[...,2]!=0)
        # TODO: check this
        mask = depth < 1.
        if __debug__:
            print('before linearize: depth max', depth.max(), 'depth min', depth.min())

        if linearDepth:

            if self.camera.coord_system == 'opengl':
                vis_depth = copy.deepcopy(depth)
                cv2.imshow('renderer depth',
                           cv2.normalize(-vis_depth, -vis_depth, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                #TODO: add mask back
                #TODO: check this ->  # negative sign is added for the sake of -z forward opengl camera coordinate
                depth =  -self.__non_linear_depth_2_linear(depth)
                # depth = -1 * self.__non_linear_depth_2_linear(depth)
                if __debug__:
                    print('before mask: depth max', depth.max(), 'depth min', depth.min())
                    depth_masked = depth[mask]
                    print('filtered depth max', depth_masked.max(), 'depth min', depth_masked.min())
                    # cv2.imshow('renderer depth',  copy.deepcopy(depth))
            else:
                depth = mask * self.__non_linear_depth_2_linear(depth)
                # print(depth.max())
        else:
            depth = mask*depth

        return rgb, depth, mask



    def __draw_box(self,model,modelExtrinsic,color):
        self.__tightBoxShader.use()
        self.__tightBoxShader.setMat4('intrinsic', self.camera.OpenGLperspective)
        self.__tightBoxShader.setMat4('extrinsic', modelExtrinsic)
        self.__tightBoxShader.setMat4('model', model)
        self.__tightBoxShader.setVec3('color', color)

        gl.glBindVertexArray(self.__vaoTightBox)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_LINES, 0, 24)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)






 # callback for window resize
def framebuffer_size_callback(window,width, height):
    # create viewport
    gl.glViewport(0,0,width,height)




def load_texture(imPath, size=None):
    """
    User function to load a given texture into GPU and get a corresponding texture ID
    :param imPath: image path
    :param size: the desired output size  (perform cv2.resize if resize is needed(
    :return:
        textureID : the ID of the texture
        texH: texture height
        texW: texture width
    """
    textureID = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, textureID)

    # Set the texture wrapping parameters
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    # Set texture filtering parameters
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    # load image
    tex = cv2.imread(imPath)
    #cv2.imshow(imPath.rsplit('.')[0],tex)
    if size != None:
        tex = cv2.resize(tex,size)
    texH, texW, _ = tex.shape

    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, texW, texH, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, tex)
    gl.glEnable(gl.GL_TEXTURE_2D)

    return textureID, texH, texW

