
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

        length = len(mesh.vertices)
        vertices = np.zeros(length*12,dtype=np.float32)
        indices = np.array([],dtype=np.uint32)
        textures = []
        HasNormal = mesh.normals.any()
        HasColor = mesh.colors.any()
        HasText = mesh.texturecoords.any()
        HasFace = mesh.faces.any()
        # for the current mesh, process all the attribut

        for idx in range(length):

            base = 12*idx
            vertices[base:base+3] = mesh.vertices[idx]

            #Find Tight 3D Bounding Box
            if mesh.vertices[idx][0] > self.Xmax:
                self.Xmax = mesh.vertices[idx][0]
            elif mesh.vertices[idx][0] < self.Xmin:
                self.Xmin = mesh.vertices[idx][0]
            if mesh.vertices[idx][1] > self.Ymax:
                self.Ymax = mesh.vertices[idx][1]
            elif mesh.vertices[idx][1] < self.Ymin:
                self.Ymin = mesh.vertices[idx][1]
            if mesh.vertices[idx][2] > self.Zmax:
                self.Zmax = mesh.vertices[idx][2]
            elif mesh.vertices[idx][2] < self.Zmin:
                self.Zmin = mesh.vertices[idx][2]


            if HasNormal:
                vertices[base+3:base+6] = mesh.normals[idx]
            else:
                vertices[base + 3:base + 6] = 3*[0.]
            if HasColor:
                vertices[base+6:base+10] = mesh.colors[0][idx]
            else:
                vertices[base + 6:base+10] = 4*[255.]
            if HasText:
                vertices[base + 10:base + 12] = mesh.texturecoords[0][idx].tolist()[0:2]
            else:
                vertices[base + 10:base + 12] =  2*[0.]


        # for the current mesh, process all the faces stored in it
        # retrieve all indices of the face and store into indices vector
        if HasFace:
            for face in mesh.faces:
                for index in face:
                    indices = np.append(indices,np.uint32(index))

        # for the current mesh, process materials
        if mesh.materialindex >= 0:
            _material = scene.materials[mesh.materialindex]

            # diffuse maps
            Maps = self.loadMaterialTextures(_material)
            textures.extend(Maps)


        return Mesh(vertices, indices, textures)


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

