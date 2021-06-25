import os
from OpenGL.GL import *
from shader import Shader
from mesh import Mesh,Texture, PhongParam
from utiles.transform import translationMatrix, scaleMatrix
import pyassimp
import numpy as np
from ctypes import  c_void_p
from imageio import imread


class Model:

    def __init__(self, name, path, scale, T_wm=np.eye(4)):
        self.textures_loaded = []
        self.meshes = []
        if path.find('\\'): self.path = path.rsplit('\\', 1)[0]
        self.directory = self.path.rsplit('/', 1)[0]
        self.name = name
        self.scale = scale

        # pose info
        self.T_cw = translationMatrix(np.array([0,0,-1]))    # viewMat
        self.T_wm = T_wm                                    # modelMat
        self.T_n = scaleMatrix(np.array([self.scale]*3))     # normalizeMat

        self.Xmax = float('-inf')
        self.Xmin = float('inf')
        self.Ymax = float('-inf')
        self.Ymin = float('inf')
        self.Zmax = float('-inf')
        self.Zmin = float('inf')

        # loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
        self.__loadModel(path)
        self.sortMeshes()

    def __del__(self):
        pyassimp.release(self.scene)

    def __get_directory(self):
        return self.directory


    def __loadModel(self,  path):
        # scene = pyassimp.load(path,processing= pyassimp.postprocess.aiProcess_FlipUVs | pyassimp.postprocess.aiProcess_Triangulate)
        self.scene = pyassimp.load(path, processing=pyassimp.postprocess.aiProcess_Triangulate | \
                                                    pyassimp.postprocess.aiProcess_GenNormals  | \
                                                    pyassimp.postprocess.aiProcess_CalcTangentSpace)

        if not self.scene:
            print('ERROR::ASSIMP:: cannot load model')
        self.processNode(self.scene.rootnode, self.scene)


    def setPoseTcw(self, Tcw):
        """
        Set the pose of T world 2 camera space
        :param Tcw: shape (4,4) numpy array, SE3 transformation
        """
        self.T_cw = Tcw

    def setPoseTwm(self, Twm):
        """
        Set the pose of T model 2 world space
        :param Twm: shape (4,4) numpy array, SE3 transformation
        """
        self.T_wm = Twm


    def getTightBox(self):
        return np.array([[self.Xmin, self.Ymin, self.Zmin],
                         [self.Xmin, self.Ymin, self.Zmax],
                         [self.Xmin, self.Ymax, self.Zmin],
                         [self.Xmin, self.Ymax, self.Zmax],
                         [self.Xmax, self.Ymin, self.Zmin],
                         [self.Xmax, self.Ymin, self.Zmax],
                         [self.Xmax, self.Ymax, self.Zmin],
                         [self.Xmax, self.Ymax, self.Zmax]],dtype=np.float)


    def set_buffer(self,pos, normal, color, tex, indices, mesh_id):
        if mesh_id == None:
            if indices.any():
                self.meshes[0].set_ebo_buffer(indices)
            else:
                self.meshes[0].set_mesh_buffer(pos, normal, color, tex)
        else:
            if indices.any():
                self.meshes[mesh_id].set_ebo_buffer(indices)
            else:
                self.meshes[mesh_id].set_mesh_buffer(pos, normal, color, tex)

    def get_buffer_data(self, attribute,mesh_id):
        if mesh_id == None:
            if attribute == 'indices':
                data = self.meshes[0].get_ebo_buffer()
            else:
                data = self.meshes[0].get_mesh_buffer(attribute)
        else:
            if attribute == 'indices':
                data = self.meshes[mesh_id].get_ebo_buffer()
            else:
                data = self.meshes[mesh_id].get_mesh_buffer(attribute)
        return  data

    def sortMeshes(self):
        """
        Due to the need of order dependent transparency
        1. Draw all opaque objects first.
        2. Sort all the transparent objects by their depth (far -> near).
        3. Draw all the transparent objects in sorted order.
        """
        opacMeshes = [m for m in self.meshes if m.phongParam.alpha == 1.0 ]
        transparMeshes = [m for m in self.meshes if m.phongParam.alpha != 1.0]

        sorted(transparMeshes,key = lambda m: m.center[2], reverse=True)
        self.meshes = opacMeshes + transparMeshes


    def draw(self,shader):
        for mesh in self.meshes:
            mesh.draw(shader)


    def processNode(self,node,scene):
        for mesh in node.meshes:
            self.meshes.append(self.processMesh(mesh,scene))
        for child in node.children:
            self.processNode(child,scene)


    def processMesh(self, mesh, scene):

        textures = []
        HasPos = mesh.vertices.any()
        HasNormal = mesh.normals.any()
        HasColor = mesh.colors.any()
        HasText = mesh.texturecoords.any()
        HasFace = mesh.faces != []
        attribute_mask = [HasPos, HasNormal, HasColor, HasText]

        # calculate bounding box
        if self.Xmax < mesh.vertices[:, 0].max(): self.Xmax = mesh.vertices[:, 0].max()
        if self.Xmin > mesh.vertices[:, 0].min(): self.Xmin = mesh.vertices[:, 0].min()
        if self.Ymax < mesh.vertices[:, 1].max(): self.Ymax = mesh.vertices[:, 1].max()
        if self.Ymin > mesh.vertices[:, 1].min(): self.Ymin = mesh.vertices[:, 1].min()
        if self.Zmax < mesh.vertices[:, 2].max(): self.Zmax = mesh.vertices[:, 2].max()
        if self.Zmin > mesh.vertices[:, 2].min(): self.Zmin = mesh.vertices[:, 2].min()

        # for the current mesh, process all the attribute

        center = np.mean(mesh.vertices,axis=0)
        position = np.reshape(mesh.vertices, -1)
        if HasNormal:
            normal = np.reshape(mesh.normals, -1)
        else:
            normal = np.array([])
        if HasColor:
            color = np.reshape(mesh.colors[0], -1)
        else:
            color = np.array([])
        if HasText:
            texcoord = np.reshape(mesh.texturecoords[0][:,0:2], -1)
        else:
            texcoord = np.array([])
        if HasFace:
            indices = np.array([item for sublist in mesh.faces for item in sublist]).astype(np.uint32)
            #indices = np.reshape(mesh.faces,-1).astype(np.uint32)
        else:
            indices = np.array([])

        # for the current mesh, process materials
        if mesh.materialindex >= 0:
            _material = scene.materials[mesh.materialindex]

            # diffuse maps
            Maps, phongParam = self.loadMaterialTextures(_material)
            textures.extend(Maps)


        return Mesh(position, center, normal, color, texcoord, indices, textures, phongParam, attribute_mask)


    def loadMaterialTextures(self,mat):
        textures = []
        phongParam = PhongParam()

        normal_path = ''
        diff_path = ''
        spec_path = ''
        bmp_path = ''
        png_path = ''
        jpg_path = ''


        for key, value in mat.properties.items():
            if type(value) == bytes:
                value=value.decode("utf-8")
            if key == 'ambient':
                if type(value) == int:
                    phongParam.Ka = [value/255.]*3
                else:
                    phongParam.Ka = value
            elif key == 'diffuse':
                if type(value) == int:
                    phongParam.Kd = [value/255.]*3
                else:
                    phongParam.Kd = value
            elif key == 'specular':
                if type(value) == int:
                    phongParam.Ks = [value/255.]*3
                else:
                    phongParam.Ks = value
            elif key == 'shininess':
                phongParam.Ns = value
            elif key == 'emissive':
                if type(value) == int:
                    phongParam.Ke = [value/255.]*3
                else:
                    phongParam.Ke = value
            elif key == 'opacity':
                if value == 0:
                    phongParam.alpha = 1
                else:
                    phongParam.alpha = value
            # for maps path
            elif key == 'file':
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
                elif value.find('.jpg') > 0:
                    jpg_path = value

        paths = {'diffusion':diff_path,'normal':normal_path,'specular':spec_path, 'bmp_path':bmp_path, 'png_path':png_path, 'jpg_path':jpg_path}

        for key,mapPath in paths.items():
            if mapPath is not '':
                #check if texture is already loaded
                if mapPath not in self.textures_loaded:
                    texture = Texture()
                    texture.id = self.TextureFromFile(mapPath)
                    texture.type = key
                    texture.path =os.path.join(self.directory,mapPath)
                    if texture.id is not -1:
                        textures.append(texture)
                        self.textures_loaded.append(mapPath)

        return textures, phongParam


    def TextureFromFile(self, path):
        filename = os.path.join(self.__get_directory(),path)

        textureID = glGenTextures(1)
        im = imread(filename)
        if np.shape(im)[0]>0 :
            try:
                texHeight, texWidth, depth = np.shape(im)
            except:
                print('texture'+filename+' has unexpected dimension')
                return -1
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

