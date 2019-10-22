from OpenGL.GL import *
from ctypes import sizeof, c_void_p, c_float
from shader import Shader
import numpy as np


class PhongParam:
    def __init__(self, Ka=3 * [1.], Kd=3 * [1.], Ks=3 * [1.], Ns=1.0):
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
    def __init__(self, id=None, type='', path=''):
        """

        :param id: The texture ID returned from glGenTextures()
        :param type: texture map type
        :param path: file name of the texture
        """
        self.id = id
        self.type = type
        self.path = path


class Mesh():
    def __init__(self, position, normal, color, texcoord, indices, textures, phongParam, attribute_mask):

        # self.vertices = np.array(vertices, dtype=np.float32)
        # self.indices = np.array(indices,dtype=np.uint32)
        self.textures = textures
        self.phongParam = phongParam
        self.position = position
        self.normal = normal
        self.color = color
        self.texcoord = texcoord
        self.indices = indices

        self.position_size = 3 * sizeof(c_float)
        self.normal_size = 3 * sizeof(c_float)
        self.color_size = 4 * sizeof(c_float)
        self.texture_size = 2 * sizeof(c_float)
        self.attribute_mask = attribute_mask

        self.__setupMesh()


    def set_mesh_buffer(self, pos, normal, color, tex):

        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)

        if self.position.any():
            pos = np.reshape(pos,-1)
            if len(pos) <= self.total_pos_nbyte:
                try:
                    glBufferSubData(GL_ARRAY_BUFFER, self.position_offset, pos)
                except:
                    print('Abort! Input size out of position buffer range!')
            else:
                print('input size out of position buffer range or no position arttribute!')


        if self.normal.any():
            normal = np.reshape(normal,-1)
            if len(normal) <= self.total_normal_nbyte:
                try:
                    glBufferSubData(GL_ARRAY_BUFFER, self.normal_offset, normal)
                except:
                    print('Abort! Input size out of normal buffer range or no normal arttribute!')
            else:
                print('input size out of normal buffer range!')

        if self.color.any():
            color = np.reshape(color,-1)
            if len(color) <= self.total_color_nbyte:
                try:
                    glBufferSubData(GL_ARRAY_BUFFER, self.color_offset, color)
                except:
                    print('Abort! Input size out of color buffer range!')
            else:
                print('input size out of color buffer range or no color arttribute!')


        if self.texcoord.any():
            tex = np.reshape(tex,-1)
            if len(tex) <= self.total_texcoord_nbyte:
                try:
                    glBufferSubData(GL_ARRAY_BUFFER, self.texcoord_offset, tex)
                except:
                    print('Abort! Input size out of texture coordinate buffer range!')
            else:
                print('input size out of texture coordinate buffer range or no coordinate arttribute!')

        glBindVertexArray(0)

    def get_mesh_buffer(self, attribute):

        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)
        if attribute == 'position' and self.attribute_mask[0]:
            data =  glGetBufferSubData(GL_ARRAY_BUFFER, self.position_offset, self.total_pos_nbyte)
            data.dtype = np.float32
            data = np.reshape(data,(-1,3))

        elif attribute == 'normal' and self.attribute_mask[1]:
            data =  glGetBufferSubData(GL_ARRAY_BUFFER, self.normal_offset, self.total_normal_nbyte)
            data.dtype = np.float32
            data = np.reshape(data,(-1,3))

        elif attribute == 'color' and self.attribute_mask[2]:
            data =  glGetBufferSubData(GL_ARRAY_BUFFER, self.color_offset, self.total_color_nbyte)
            data.dtype = np.float32
            data = np.reshape(data,(-1,4))

        elif attribute == 'textureCoord' and self.attribute_mask[3]:
            data =  glGetBufferSubData(GL_ARRAY_BUFFER, self.texcoord_offset, self.total_texcoord_nbyte)
            data.dtype = np.float32
            data = np.reshape(data,(-1,2))

        glBindVertexArray(0)
        return data

    def set_ebo_buffer(self, indices):
        if len(indices) == len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__EBO)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)
        else:
            print('indices size does not match mesh vertices size')

    def get_ebo_buffer(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__EBO)
        data = glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, self.indices.nbytes)
        data.dtype = np.int32

        glBindVertexArray(0)
        return data

    def __setupMesh(self):

        # ----------------- generate & bind __VAO, __VBO
        self.__VAO = glGenVertexArrays(1)
        self.__VBO = glGenBuffers(1)
        self.__EBO = glGenBuffers(1)

        self.position_offset = 0
        self.normal_offset = 0
        self.color_offset = 0
        self.texcoord_offset = 0
        self.total_pos_nbyte = self.position.nbytes
        self.total_normal_nbyte = self.normal.nbytes
        self.total_color_nbyte = self.color.nbytes
        self.total_texcoord_nbyte = self.texcoord.nbytes
        self.total_buffer_nbyte = self.total_pos_nbyte  + self.total_normal_nbyte + self.total_color_nbyte + self.total_texcoord_nbyte

        # Bind to the array buffer
        glBindVertexArray(self.__VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.__VBO)

        # =============== The below VBO buffer is a non-interleaving 1D array that stores first all the pos, then all the normals, ... etc.
        # Fill the position data & set attributes


        glBufferData(GL_ARRAY_BUFFER,self.total_buffer_nbyte, None , GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0,self.total_pos_nbyte,self.position)
        current_size = self.total_pos_nbyte

        # Fill the normal data
        if self.normal.any():
            self.normal_offset = current_size
            glBufferSubData(GL_ARRAY_BUFFER, current_size,self.normal)
            current_size += self.total_normal_nbyte

        # Fill the color data
        if self.color.any():
            self.color_offset = current_size
            glBufferSubData(GL_ARRAY_BUFFER, current_size,self.color)
            current_size += self.total_color_nbyte

        # Fill the texcoord data
        if self.texcoord.any():
            self.texcoord_offset = current_size
            glBufferSubData(GL_ARRAY_BUFFER, current_size, self.texcoord)
            current_size += self.total_texcoord_nbyte

        # Bind to the element array buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.__EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)

        # ----------------- setup & enable attributes
        # vertex positions
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.position_size, c_void_p(0))

        # vertex normals
        current_size = self.total_pos_nbyte
        if self.normal.any():
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.normal_size, c_void_p(current_size))
            current_size += self.total_normal_nbyte

        # vertex color
        if self.color.any():
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, self.color_size, c_void_p(current_size))
            current_size += self.total_color_nbyte

        # vertex texture coords
        if self.texcoord.any():
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, self.texture_size, c_void_p(current_size))
            current_size += self.total_texcoord_nbyte

        glBindVertexArray(0)



    def draw(self, shader):

        # Set the phong shading model to the corresponding mesh
        shader.use()
        shader.setVec3('material.Ka', self.phongParam.Ka)
        shader.setVec3('material.Kd', self.phongParam.Kd)
        shader.setVec3('material.Ks', self.phongParam.Ks)
        shader.setFloat('material.Ns', self.phongParam.Ns)

        # Set to default color if no color is presented
        if self.color.any():
            shader.setBool('hasColor', 1)
        else:
            shader.setBool('hasColor', 0)
        if self.texcoord.any():
            shader.setBool('hasTexture',1)
        else:
            shader.setBool('hasTexture', 0)
        if self.normal.any():
            shader.setBool('hasNormal', 1)
        else:
            shader.setBool('hasNormal', 0)


        if self.texcoord.any():
            # ------------------ Active Textures
            for i, tex in enumerate(self.textures):
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
                    shader.setInt('material.map_Ka', i)
                    # shader.setInt('material.map_Ks', i)
                    glActiveTexture(GL_TEXTURE0 + i)
                    glBindTexture(GL_TEXTURE_2D, tex.id)


        # -------------------- Draw
        glBindVertexArray(self.__VAO)
        glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_INT, c_void_p(0))
        glBindVertexArray(0)

        if self.texcoord.any():
            glActiveTexture(GL_TEXTURE0)


