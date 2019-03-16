
import os
from OpenGL.GL import *
from shader import Shader
from mesh import Mesh,Texture
import pyassimp
import glm
import cv2
import numpy as np
from ctypes import  c_void_p
from scipy.misc import imread
from libc.stdint cimport uint32_t
cimport numpy as np

class Model:

    def __init__(self, directory):
        self.meshes = []
        self.textures_loaded = []
        self.Xmax = 0
        self.Xmin = 0
        self.Ymax = 0
        self.Ymin = 0
        self.Zmax = 0
        self.Zmin = 0
        self.directory = directory.rsplit('/', 1)[0]
        self.__loadModel(directory)

    def __get_directory(self):
        return self.directory


    def __loadModel(self,  path):
        scene = pyassimp.load(path,processing= pyassimp.postprocess.aiProcess_FlipUVs | pyassimp.postprocess.aiProcess_Triangulate)

        if not scene:
            print('ERROR::ASSIMP:: cannot load model')
        self.processNode(scene.rootnode, scene)

    def draw(self,shader):
        for mesh in self.meshes:
            mesh.draw(shader)

    def processNode(self,node,scene):
        for mesh in node.meshes:
            self.meshes.append(self.processMesh(mesh,scene))
        for child in node.children:
            self.processNode(child,scene)



    def processMesh(self, mesh, scene):

        cdef float[:,:] meshVertices = mesh.vertices

        if not mesh.normals.any():
            mn= np.zeros((1,1),dtype = np.float32)
        else:
            mn= mesh.normals
        cdef float[:,:] meshNormals = mn

        if not mesh.colors.any():
            mc= np.zeros((1,1,1),dtype = np.float32)
        else:
            mc= mesh.colors
        cdef float[:,:,:] meshColors = mc

        if not mesh.texturecoords.any():
            mt= np.zeros((1,1,1),dtype = np.float32)
        else:
            mt= mesh.texturecoords
        cdef float[:,:,:] meshTexturecoords = mt
        cdef float[:] a = np.array([0., 0., 0.],dtype = np.float32)
        cdef float[:] b = np.array([255., 255., 255., 255.],dtype = np.float32)
        cdef float[:] c = np.array([0., 0.],dtype = np.float32)

        cdef Py_ssize_t totalVerticesNum = len(meshVertices)
        vertices = np.zeros(totalVerticesNum*12,dtype=np.float32)
        cdef float[:] cvertices = vertices

        cdef Py_ssize_t indicesLen
        if mesh.faces.any():
            indicesLen = 3*len(mesh.faces)
        indices = np.zeros(indicesLen,dtype=np.uint32)
        cdef uint32_t[:] cindices = indices



        textures = []

        # for the current mesh, process all the attribut
        cdef Py_ssize_t idx
        cdef Py_ssize_t base
        HasNormal = mesh.normals.any()
        HasColor = mesh.colors.any()
        HasText = mesh.texturecoords.any()
        HasFace = mesh.faces.any()
        for idx in range(totalVerticesNum):

            base = 12*idx
            cvertices[base:base+3] = meshVertices[idx]

            #Find Tight 3D Bounding Box
            if meshVertices[idx][0] > self.Xmax:
                self.Xmax = meshVertices[idx][0]
            elif meshVertices[idx][0] < self.Xmin:
                self.Xmin = meshVertices[idx][0]
            if meshVertices[idx][1] > self.Ymax:
                self.Ymax = meshVertices[idx][1]
            elif meshVertices[idx][1] < self.Ymin:
                self.Ymin = meshVertices[idx][1]
            if meshVertices[idx][2] > self.Zmax:
                self.Zmax = meshVertices[idx][2]
            elif meshVertices[idx][2] < self.Zmin:
                self.Zmin = meshVertices[idx][2]


            if HasNormal:
                cvertices[base + 3:base + 6] = meshNormals[idx]
            else:
                cvertices[base + 3:base + 6] = a
            if HasColor:
                cvertices[base+6:base+10] = meshColors[0][idx]
            else:
                cvertices[base + 6:base+10] = b
            if HasText:
                cvertices[base + 10:base + 12] = meshTexturecoords[0][idx][0:2]
            else:
                cvertices[base + 10:base + 12] = c


        # for the current mesh, process all the faces stored in it
        # retrieve all indices of the face and store into indices vector
        cdef Py_ssize_t faceId
        cdef Py_ssize_t Id
        cdef Py_ssize_t index
        cdef int[:] face
        if HasFace:
            for faceId,face in enumerate(mesh.faces):
                cindices[faceId*3]   = face[0]
                cindices[faceId*3+1] = face[1]
                cindices[faceId*3+2] = face[2]

        # for the current mesh, process materials
        if mesh.materialindex >= 0:
            _material = scene.materials[mesh.materialindex]

            # diffuse maps
            Maps = self.loadMaterialTextures(_material)
            textures.extend(Maps)


        return Mesh(cvertices, cindices, textures)


    def loadMaterialTextures(self,mat):
        textures = []
        normal_path = ''
        diff_path = ''
        spec_path = ''
        bmp_path = ''
        png_path = ''
        for key, value in mat.properties.items():
            if key == 'file':
                if value.find('_ddn') > 0:
                    normal_path = value
                elif value.find('_dif') > 0:
                    diff_path = value
                elif value.find('_spec') > 0:
                    spec_path = value
                elif value.find('.bmp') > 0:
                    bmp_path = value
                elif value.find('.png') > 0:
                    png_path = value

        paths = {'diffusion':diff_path,'normal':normal_path,'specular':spec_path, 'bmp_path':bmp_path, 'png_path':png_path}

        for key,mapPath in paths.items():
            if mapPath is not '':
                texture = Texture()
                texture.id = self.TextureFromFile(mapPath)
                texture.type = key
                texture.path =os.path.join(self.directory,mapPath)
                textures.append(texture)

        return textures


    def TextureFromFile(self, path):
        filename = os.path.join(self.__get_directory(),path)

        textureID = glGenTextures(1)
        im = imread(filename)
        if np.shape(im)[0]>0:
            glBindTexture(GL_TEXTURE_2D, textureID)
            texHeight, texWidth, depth = np.shape(im)
            if path.split('.', 1)[1] == 'png' and depth == 4:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, im)
            else:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, im)
            glGenerateMipmap(GL_TEXTURE_2D)
            # warpings
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            # filterings
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            # load texture
        else:
            print('Fail to load textures at path %s',self.__get_directory())

        return textureID





if __name__ == '__main__':



    print('d')

