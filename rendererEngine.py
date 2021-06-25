import sys,os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import OpenGL.GL as gl
from ctypes import  sizeof, c_void_p,c_float
import numpy as np
import cv2
from transform import translationMatrix, scaleMatrix

float_size = sizeof(c_float)



class RendererEngine:
    def __init__(self, window, models, lights, camera, modelShader, lightCubeShader, tightBoxShader):

        self.window = window

        self._models = models
        self._lights = lights

        self.camera = camera

        self._modelShader = modelShader
        self._lightCubeShader = lightCubeShader
        self._tightBoxShader = tightBoxShader
        # self._lightCubeShader = Shader(vShaderLampPath, fShaderLampPath)
        # self._modelShader = Shader(vShaderPath, fShaderPath)


        self._rendererWidth = None
        self._rendererHeight = None
        self.setWidthHeight(self.window.width, self.window.height)


        self.__pointLightVertices = np.array(\
            [-0.5, -0.5, -0.5,
              0.5, -0.5, -0.5,
              0.5, 0.5, -0.5,
              0.5, 0.5, -0.5,
              -0.5, 0.5, -0.5,
              -0.5, -0.5, -0.5,

              -0.5, -0.5, 0.5,
              0.5, -0.5, 0.5,
              0.5, 0.5, 0.5,
              0.5, 0.5, 0.5,
              -0.5, 0.5, 0.5,
              -0.5, -0.5, 0.5,

              -0.5, 0.5, 0.5,
              -0.5, 0.5, -0.5,
              -0.5, -0.5, -0.5,
              -0.5, -0.5, -0.5,
              -0.5, -0.5, 0.5,
              -0.5, 0.5, 0.5,

              0.5, 0.5, 0.5,
              0.5, 0.5, -0.5,
              0.5, -0.5, -0.5,
              0.5, -0.5, -0.5,
              0.5, -0.5, 0.5,
              0.5, 0.5, 0.5,

              -0.5, -0.5, -0.5,
              0.5, -0.5, -0.5,
              0.5, -0.5, 0.5,
              0.5, -0.5, 0.5,
              -0.5, -0.5, 0.5,
              -0.5, -0.5, -0.5,

              -0.5, 0.5, -0.5,
              0.5, 0.5, -0.5,
              0.5, 0.5, 0.5,
              0.5, 0.5, 0.5,
              -0.5, 0.5, 0.5,
              -0.5, 0.5, -0.5], dtype=np.float32)
        self.__tightBoxVertices = np.array( \
            [-0.5, -0.5, -0.5,
             -0.5, -0.5, 0.5,
             -0.5, 0.5, -0.5,
             -0.5, 0.5, 0.5,
             0.5, -0.5, -0.5,
             0.5, -0.5, 0.5,
             0.5, 0.5, -0.5,
             0.5, 0.5, 0.5,

             -0.5, -0.5, 0.5,
             -0.5, 0.5, 0.5,
             0.5, -0.5, 0.5,
             0.5, 0.5, 0.5,
             -0.5, -0.5, -0.5,
             -0.5, 0.5, -0.5,
             0.5, -0.5, -0.5,
             0.5, 0.5, -0.5,

             -0.5, -0.5, 0.5,
             0.5, -0.5, 0.5,
             -0.5, 0.5, 0.5,
             0.5, 0.5, 0.5,
             -0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,
             -0.5, 0.5, -0.5,
             0.5, 0.5, -0.5], dtype=np.float32)

        self.__vboLamp, self.__vaoLamp = self.__setupLight()
        self.__vboTightBox, self.__vaoTightBox = self.__setupTightBox()

        self.setup_blending()

        #TODO: deal with the problem if the attributes of one of the lights are changed
        # self.update_light()

    @staticmethod
    def clearWindow(color, alpha=1, depth=1.):
        """
        Clear window with specific values
        :param color: RGB tuple/list/array
        :param alpha: [0,1] alpha value
        :return: None
        """
        gl.glClearColor(color[0], color[1], color[2], alpha)
        gl.glClearDepth(depth)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def downloadFrame(self, type, flipOn=True):

        if type == "MASK":
            frame = gl.glReadPixels(0, 0, self._rendererWidth, self._rendererHeight, gl.GL_RED, gl.GL_UNSIGNED_BYTE)
            frame = np.fromstring(frame, np.uint8)
            #TODO: merge this
            frame = frame.reshape(self._rendererHeight, self._rendererWidth)

        elif type == "BGR":
            frame = gl.glReadPixels(0, 0, self._rendererWidth, self._rendererHeight, gl.GL_BGR, gl.GL_UNSIGNED_BYTE)
            frame = np.frombuffer(frame, np.uint8)
            frame = frame.reshape(self._rendererHeight, self._rendererWidth, 3)

        elif type == "DEPTH":
            frame = gl.glReadPixels(0, 0, self._rendererWidth, self._rendererHeight, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
            frame = frame.reshape((self._rendererHeight, self._rendererWidth))

        else:
            frame = np.zeros((self._rendererHeight, self._rendererWidth), dtype=np.uint8)

        if flipOn:
            frame = np.flipud(frame)

        return frame

    def linearizeDepth(self, depth):
        f = self.camera.far
        n = self.camera.near

        return (2. * n * f) / (f + n - (2 * depth - 1.0) * (f - n))


    def renderShaded(self):
        gl.glViewport(0, 0, self._rendererWidth, self._rendererHeight)

        # setup the same shader
        self._modelShader.use()
        self._modelShader.setVec3("viewPos", self.camera.position)
        self._modelShader.setFloat("material.shininess", 32.0)

        # # direction light
        # self._modelShader.setVec3("dirLight.direction", np.array([-0.2, -1.0, -0.3]))
        # self._modelShader.setVec3("dirLight.ambient", np.array([0.05, 0.05, 0.05]))
        # self._modelShader.setVec3("dirLight.diffuse", np.array([0.4, 0.4, 0.4]))
        # self._modelShader.setVec3("dirLight.specular", np.array([0.5, 0.5, 0.5]))

        # setup the lightings of the model shader
        for id, light in enumerate(self._lights):
            # point lights
            self._modelShader.setVec3("pointLights[{:d}].position".format(id), light.position)
            self._modelShader.setVec3("pointLights[{:d}].ambient".format(id), light.ambient)
            self._modelShader.setVec3("pointLights[{:d}].diffuse".format(id), light.diffuse)
            self._modelShader.setVec3("pointLights[{:d}].specular".format(id), light.specular)
            self._modelShader.setFloat("pointLights[{:d}].constant".format(id), light.constant)
            self._modelShader.setFloat("pointLights[{:d}].linear".format(id), light.linear)
            self._modelShader.setFloat("pointLights[{:d}].quadratic".format(id), light.quadratic)

        # for each model, setup the transformations
        for model in self._models:
            # setup shader of the model
            self._modelShader.setMat4("projection", self.camera.perspective)
            self._modelShader.setMat4("view", model.T_cw)

            # world transformation
            self._modelShader.setMat4("model", model.T_wm @ model.T_n)

            model.draw(self._modelShader)


    def renderSilhouette(self):
        pass

    def renderLight(self, scale):
        self._lightCubeShader.use()
        for light in self._lights:
            gl.glBindVertexArray(self.__vaoLamp)
            self._lightCubeShader.setVec3("color", light.diffuse)
            self._lightCubeShader.setMat4("model", translationMatrix(light.position) @ scaleMatrix(scale))
            self._lightCubeShader.setMat4("projection", self.camera.perspective)
            self._lightCubeShader.setMat4("view", self.camera.view)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)

    def renderTightBox(self, pos, scale, shader=None):
        if shader is not None:
            _shader = shader
        else:
            _shader = self._tightBoxShader

        _shader.use()
        _shader.setMat4('projection', self.camera.perspective)
        _shader.setMat4('view', self.camera.view)
        _shader.setMat4('model', translationMatrix(pos) @ scaleMatrix(scale))

        gl.glBindVertexArray(self.__vaoTightBox)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_LINES, 0, 24)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def renderTSDFOutter(self, pos, scale, shader=None):
        self._lightCubeShader.use()
        gl.glBindVertexArray(self.__vaoLamp)
        self._lightCubeShader.setVec3("color", np.ones(3))
        self._lightCubeShader.setMat4("model", translationMatrix(pos) @ scaleMatrix(scale))
        self._lightCubeShader.setMat4("projection", self.camera.perspective)
        self._lightCubeShader.setMat4("view", self.camera.view)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)


    def setWidthHeight(self, width, height):
        self._rendererWidth = width
        self._rendererHeight = height


    def setup_blending(self, FaceCull=False,Blend=False,DepthFunc='GL_LEQUAL',DepthTest=True,MultiSample=True):
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


    # def setup3DTightbox(self, xmin, xmax, ymin, ymax, zmin, zmax):
    #     vbo_TightBox = gl.glGenBuffers(1)
    #     vao_TightBox = gl.glGenVertexArrays(1)
    #
    #     _vertices = np.array([
    #           xmin, ymin, zmin
    #         , self.__3DTightBoxVertices[1][0], self.__3DTightBoxVertices[1][1], self.__3DTightBoxVertices[1][2]
    #         , self.__3DTightBoxVertices[1][0], self.__3DTightBoxVertices[1][1], self.__3DTightBoxVertices[1][2]
    #         , self.__3DTightBoxVertices[3][0], self.__3DTightBoxVertices[3][1], self.__3DTightBoxVertices[3][2]
    #         , self.__3DTightBoxVertices[3][0], self.__3DTightBoxVertices[3][1], self.__3DTightBoxVertices[3][2]
    #         , self.__3DTightBoxVertices[2][0], self.__3DTightBoxVertices[2][1], self.__3DTightBoxVertices[2][2]
    #         , self.__3DTightBoxVertices[2][0], self.__3DTightBoxVertices[2][1], self.__3DTightBoxVertices[2][2]
    #         , xmin, ymin, zmin
    #         , self.__3DTightBoxVertices[4][0], self.__3DTightBoxVertices[4][1], self.__3DTightBoxVertices[4][2]
    #         , self.__3DTightBoxVertices[5][0], self.__3DTightBoxVertices[5][1], self.__3DTightBoxVertices[5][2]
    #         , self.__3DTightBoxVertices[5][0], self.__3DTightBoxVertices[5][1], self.__3DTightBoxVertices[5][2]
    #         , self.__3DTightBoxVertices[7][0], self.__3DTightBoxVertices[7][1], self.__3DTightBoxVertices[7][2]
    #         , self.__3DTightBoxVertices[7][0], self.__3DTightBoxVertices[7][1], self.__3DTightBoxVertices[7][2]
    #         , self.__3DTightBoxVertices[6][0], self.__3DTightBoxVertices[6][1], self.__3DTightBoxVertices[6][2]
    #         , self.__3DTightBoxVertices[6][0], self.__3DTightBoxVertices[6][1], self.__3DTightBoxVertices[6][2]
    #         , self.__3DTightBoxVertices[4][0], self.__3DTightBoxVertices[4][1], self.__3DTightBoxVertices[4][2]
    #         , self.__3DTightBoxVertices[5][0], self.__3DTightBoxVertices[5][1], self.__3DTightBoxVertices[5][2]
    #         , self.__3DTightBoxVertices[1][0], self.__3DTightBoxVertices[1][1], self.__3DTightBoxVertices[1][2]
    #         , self.__3DTightBoxVertices[7][0], self.__3DTightBoxVertices[7][1], self.__3DTightBoxVertices[7][2]
    #         , self.__3DTightBoxVertices[3][0], self.__3DTightBoxVertices[3][1], self.__3DTightBoxVertices[3][2]
    #         , self.__3DTightBoxVertices[4][0], self.__3DTightBoxVertices[4][1], self.__3DTightBoxVertices[4][2]
    #         , xmin, ymin, zmin
    #         , self.__3DTightBoxVertices[6][0], self.__3DTightBoxVertices[6][1], self.__3DTightBoxVertices[6][2]
    #         , self.__3DTightBoxVertices[2][0], self.__3DTightBoxVertices[2][1], self.__3DTightBoxVertices[2][2]],dtype=np.float32)
    #
    #     gl.glBindVertexArray(vao_TightBox)
    #
    #     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_TightBox)
    #     gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)
    #     gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 3 * float_size, c_void_p(0))
    #     gl.glEnableVertexAttribArray(0)
    #
    #     return vbo_TightBox, vao_TightBox


    def __setupLight(self):
        # Buffer setting for Lamp
        vbo_lamp = gl.glGenBuffers(1)
        vao_lamp = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_lamp)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_lamp)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.__pointLightVertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 3 * float_size, c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        return vbo_lamp, vao_lamp

    def __setupTightBox(self):
        vbo_box = gl.glGenBuffers(1)
        vao_box = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_box)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_box)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.__tightBoxVertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 3 * float_size, c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        return vbo_box, vao_box


    # def update_light(self, light):
    #     # set the light shader
    #     self._modelShader.use()
    #     self._modelShader.setBool('light.enableDirectional', light.enableDirectional)
    #     self._modelShader.setBool('light.enableAttenuation', light.enableAttenuation)
    #     self._modelShader.setVec3('light.position', light.position)
    #     self._modelShader.setFloat('light.constant', light.constant)
    #     self._modelShader.setFloat('light.linear', light.linear)
    #     self._modelShader.setFloat('light.quadratic', light.quadratic)
    #     self._modelShader.setVec3('light.ambient', light.ambient)
    #     self._modelShader.setVec3('light.diffuse', light.diffuse)
    #     self._modelShader.setVec3('light.specular', light.specular)

    # def __setModelPose(self, modelMat, extrinsicMat):
    #     self.__modelShader.use()
    #     self.__modelShader.setMat4('intrinsic', self.camera.OpenGLperspective)
    #     self.__modelShader.setMat4('extrinsic', extrinsicMat)
    #     self.__modelShader.setMat4('model', modelMat)

    # @staticmethod
    # def draw_pixels(img):
    #     """
    #         Render images to the currently binded window
    #
    #     Parameters
    #     ----------
    #     img : array_like, shape(h,w,3)
    #         image to be rendered with type np.uint8
    #     Returns
    #     -------
    #         None
    #     """
    #
    #     #TODO: add viewport to function
    #     data = np.flipud(img)
    #     gl.glDrawPixels(data.shape[1], data.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

    # def __drawLamp(self, extrinsicMat):
    #     self.__lampShader.use()
    #     self.__lampShader.setVec3('color',self.light.lightColor)
    #     self.__lampShader.setMat4('intrinsic', self.camera.OpenGLperspective)
    #     self.__lampShader.setMat4('extrinsic', extrinsicMat)
    #     model = np.eye(4)
    #     self.__lampShader.setMat4('model', model)
    #
    #     gl.glBindVertexArray(self.__vaoLamp)
    #     gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)

    # def __non_linear_depth_2_linear(self,depth):
    #     f = self.camera.far
    #     n = self.camera.near
    #     ndc_depth = 2*depth-1
    #
    #     return (2.0 * n * f) / (f + n - ndc_depth * (f - n))


    # def set_vertex_buffer(self, pos=None, normal=None, color=None, tex=None, indices=np.array([]), mesh_id = None):
    #     """
    #     :param vertices: the format for N vertices stored in buffer is in a non-interleaving form:
    #      [v1.pos,.... vN.pos, v1.normal, ... ,vN.normal, v1.color, ...vN.color, v1.tex, ... vN.tex]
    #      Note that both position & Normal have 3 channels (x,y,z),(Nx,Ny,Nz), color has 4 channels (r,g,b,a), texture has 2 (u,v).
    #     :return: None
    #     """
    #     self.__3dModel.set_buffer(pos, normal, color, tex, indices, mesh_id)

    # def get_vertex_buffer(self, attribute='position', mesh_id = None):
    #     """
    #
    #     :param attribute: choose any attribute from position/normal/color/textureCoord/indices
    #     :param mesh_id: what mesh id to be edit
    #     :return:
    #     """
    #     return self.__3dModel.get_buffer_data(attribute,mesh_id)



    # def draw(self, model, modelExtrinsic,lightExtrinsic,drawLamp=True, drawBox=False, meshBymesh=False, color=[255,255,255], linearDepth=False, draw_inverse=True):
    #     """
    #
    #     :param model: model matrix (model space transformation matrix)
    #     :param modelExtrinsic: Extrinsic matrix of the model
    #     :param lightExtrinsic: Extrinsic matrix of the light source
    #     :param drawLamp: enable lamp drawing
    #     :param drawBox: enable box drawing
    #     :param color: color of the box
    #     :param linearDepth: enable linear depth (which is a must for correct groundtruth)
    #     :return: rgb, depth
    #     """
    #
    #     # gl.glClearDepth(1.)
    #     # gl.glDepthFunc(gl.GL_LESS)
    #     # gl.glClearDepth(0.)
    #     # gl.glDepthFunc(gl.GL_GREATER)
    #     # gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
    #     self.__setModelPose(model,modelExtrinsic)
    #     self.__3dModel.draw(self.__modelShader, meshBymesh)
    #     if drawLamp:
    #         self.__drawLamp(lightExtrinsic)
    #
    #     if drawBox:
    #         self.__draw_box(model, modelExtrinsic, color)
    #
    #     #self.window.updateWindow()
    #     depth = gl.glReadPixels(0, 0, self.window.window_size[0], self.window.window_size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
    #
    #     depth = depth.reshape(self.window.window_size[::-1])
    #     depth = np.flipud(depth)
    #     # depth = 1-depth
    #
    #
    #     imageBuf = gl.glReadPixels(0, 0, self.window.window_size[0], self.window.window_size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    #     im = np.fromstring(imageBuf, np.uint8)
    #
    #     #This is because of the y axis of the image coordinate system and that of the opencv image layout is inverted
    #     rgb = np.reshape(im, (self.window.window_size[1], self.window.window_size[0], 3))
    #     rgb = np.flipud(rgb)
    #
    #     # Since the value from depth buffer contains non-linear depth ~[0,1], need to linearize
    #     # Apply mask s.t background has depth 0
    #     # mask = depth < 1
    #     # mask = np.logical_or(np.logical_or(rgb[...,0]!=0, rgb[...,1]!=0),rgb[...,2]!=0)
    #     # TODO: check this
    #     # mask = depth < 1.
    #     # mask = depth > 0.
    #     # assert np.any(mask)
    #
    #     # if draw_inverse:
    #     #     gl.glClearDepth(0.)
    #     #     gl.glDepthFunc(gl.GL_GREATER)
    #     #     # gl.glClearDepth(1.)
    #     #     # gl.glDepthFunc(gl.GL_LESS)
    #     #     gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
    #     #     self.__3dModel.draw(self.__modelShader, meshBymesh)
    #     #     inv_depth = gl.glReadPixels(0, 0, self.window.window_size[0], self.window.window_size[1], gl.GL_DEPTH_COMPONENT,
    #     #                             gl.GL_FLOAT)
    #     #
    #     #     inv_depth = inv_depth.reshape(self.window.window_size[::-1])
    #     #     inv_depth = np.flipud(inv_depth)
    #     #     # inv_depth = 1- inv_depth
    #
    #
    #     # if __debug__:
    #     #     print('before linearize: depth max', depth.max(), 'depth min', depth.min())
    #     #     print('before linearize: depth inv max', inv_depth.max(), 'depth inv min', inv_depth.min())
    #
    #     # Since the value from depth buffer contains non-linear depth ~[0,1], background depth will be cast to 1.
    #     mask = depth < 1
    #     if linearDepth:
    #         depth = -1 * mask * self.__non_linear_depth_2_linear(depth)
    #     else:
    #         depth = mask * depth
    #
    #     return rgb, depth
    #
    #     # if linearDepth:
    #     #
    #     #     if self.camera.coord_system == 'opengl':
    #     #         if __debug__:
    #     #             vis_depth = copy.deepcopy(depth)
    #     #             vis_inv_depth = copy.deepcopy(inv_depth)
    #     #             cv2.imshow('renderer depth',
    #     #                        cv2.normalize(-vis_depth, -vis_depth, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    #     #             cv2.imshow('renderer inverse depth',
    #     #                        cv2.normalize(-vis_inv_depth, -vis_inv_depth, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    #     #         #TODO: add mask back
    #     #         #TODO: check this ->  # negative sign is added for the sake of -z forward opengl camera coordinate
    #     #         depth =  -self.__non_linear_depth_2_linear(depth)
    #     #         inv_depth = -self.__non_linear_depth_2_linear(inv_depth)
    #     #         # depth = -1 * self.__non_linear_depth_2_linear(depth)
    #     #         if __debug__:
    #     #             print('before mask: depth max', depth.max(), 'depth min', depth.min())
    #     #             print('before mask: inv depth max', inv_depth.max(), 'inv depth min', inv_depth.min())
    #     #             depth_masked = depth[mask]
    #     #             inv_depth_masked = inv_depth[mask]
    #     #             print('filtered depth max', depth_masked.max(), 'depth min', depth_masked.min())
    #     #             print('filtered inv depth max', inv_depth_masked.max(), 'depth inv min', inv_depth_masked.min())
    #     #             # cv2.imshow('renderer depth',  copy.deepcopy(depth))
    #     #     else:
    #     #         depth = mask * self.__non_linear_depth_2_linear(depth)
    #     #         # print(depth.max())
    #     # else:
    #     #     depth = mask*depth
    #
    #     # return rgb, depth, mask, inv_depth

    # def drawBox(self, pos, scale, shader=None):
    #     if shader is not None:
    #         _shader = shader
    #     else:
    #         _shader = self.__tightBoxShader
    #
    #     _shader.use()
    #     _shader.setMat4('projection', self.camera.perspective)
    #     _shader.setMat4('view', self.camera.view)
    #     _shader.setMat4('model', translationMatrix(pos) @ scaleMatrix(scale))
    #     # self._shader.setVec3('color', color)
    #
    #     gl.glBindVertexArray(self.__vaoTightBox)
    #     gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    #     gl.glDrawArrays(gl.GL_LINES, 0, 36)
    #     gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)



#  # callback for window resize
# def framebuffer_size_callback(window,width, height):
#     # create viewport
#     gl.glViewport(0,0,width,height)




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

