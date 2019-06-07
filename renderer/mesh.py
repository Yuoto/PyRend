from OpenGL.GL import *
from ctypes import sizeof, c_void_p, c_float
from shader import Shader
import numpy as np

class PhongParam:
    def __init__(self, Ka = 3*[1.], Kd = 3*[1.], Ks = 3*[1.], Ns = 1.0):
        """

        :param Ka: ambient coefficient, with default value [1.0, 1.0, 1.0]
        :param Kd: diffuse coefficient, with default value [1.0, 1.0, 1.0]
        :param Ks: specular coefficient, with default value [1.0, 1.0, 1.0]
        :param Ns: shininess, with default value 1.0
        See more for .mtl format at http://paulbourke.net/dataformats/mtl/
        """
        self.Ka = Ka
        self.Kd = Kd
        self.Ks = Ks
        self.Ns = Ns

class Texture:
    def __init__(self,id=None,type='',path=''):

        """

        :param id: The texture ID returned from glGenTextures()
        :param type: texture map type
        :param path: file name of the texture
        """
        self.id = id
        self.type = type
        self.path = path


class Mesh():
    def __init__(self, vertices, indices, textures, phongParam):

        self.vertices = np.array(vertices,dtype=np.float32)
        self.indices = np.array(indices,dtype=np.uint32)
        self.textures = textures
        self.phongParam = phongParam
        self.stride = 12*sizeof(c_float)

        self.__setupMesh()


    def __setupMesh(self):

        # ----------------- generate & bind __VAO, __VBO
        self.__VAO = glGenVertexArrays(1)
        self.__VBO = glGenBuffers(1)
        self.__EBO = glGenBuffers(1)

        glBindVertexArray(self.__VAO)

        glBindBuffer(GL_ARRAY_BUFFER,self.__VBO)
        glBufferData(GL_ARRAY_BUFFER,self.vertices,GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,self.__EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,self.indices,GL_STATIC_DRAW)


        # ----------------- setup & enable attributes
        # vertex positions
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, self.stride,c_void_p(0))

        # vertex normals
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,  self.stride, c_void_p(3 * sizeof(c_float)))

        # vertex color
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, self.stride, c_void_p(6 * sizeof(c_float)))

        #vertex texture coords
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE,  self.stride, c_void_p(10 * sizeof(c_float)))



        glBindVertexArray(0)



    def draw(self,shader):

        # Set the phong shading model to the corresponding mesh
        shader.setVec3('material.Ka', self.phongParam.Ka)
        shader.setVec3('material.Kd', self.phongParam.Kd)
        shader.setVec3('material.Ks', self.phongParam.Ks)
        shader.setFloat('material.Ns', self.phongParam.Ns)

        if any(self.textures):
            # ------------------ Active Textures
            shader.setBool('hasTexture', 1)
            for i,tex in enumerate(self.textures):
                if tex.type == 'diffusion':
                    shader.setInt('material.map_Kd', i)
                    glActiveTexture(GL_TEXTURE0 + i)
                    glBindTexture(GL_TEXTURE_2D, tex.id)

                elif tex.type == 'specular':
                    shader.setInt('material.map_Ks', i)
                    glActiveTexture(GL_TEXTURE0 + i)
                    glBindTexture(GL_TEXTURE_2D, tex.id)

                else:
                    shader.setInt('material.map_Kd', i)
                    glActiveTexture(GL_TEXTURE0 + i)
                    glBindTexture(GL_TEXTURE_2D, tex.id)

                    shader.setInt('material.map_Ka', i)
                    glActiveTexture(GL_TEXTURE0 + i)
                    glBindTexture(GL_TEXTURE_2D, tex.id)

        else:
            shader.setBool('hasTexture',0)



        # -------------------- Draw
        glBindVertexArray(self.__VAO)
        glDrawElements(GL_TRIANGLES,self.indices.size,GL_UNSIGNED_INT,c_void_p(0))
        glBindVertexArray(0)

        if any(self.textures):
            glActiveTexture(GL_TEXTURE0)


