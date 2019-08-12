import os
from OpenGL.GL import *

class Shader:


    def __init__(self, vertexPath, fragmentPath):

        # read shader from path
        try:
            with open(vertexPath,'r') as f:
                self.vShaderCode = f.read()
        except IOError:
            print('Could not read file:'+ vertexPath)

        try:
            with open(fragmentPath,'r') as f:
                self.fShaderCode = f.read()
        except IOError:
            print('Could not read file:'+ fragmentPath)


        # compile shaders
        self.vShader = self.create_shader(GL_VERTEX_SHADER,self.vShaderCode)
        self.fShader = self.create_shader(GL_FRAGMENT_SHADER, self.fShaderCode)
        self.create_program()

    def create_shader(self, shader_type, source_code):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source_code)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader))
        return shader

    def create_program(self):

        self.programID = glCreateProgram()
        glAttachShader(self.programID,self.vShader)
        glAttachShader(self.programID, self.fShader)
        glLinkProgram(self.programID)

        if glGetProgramiv(self.programID, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.programID))
        glDeleteShader(self.vShader)
        glDeleteShader(self.fShader)

    def use(self):
        glUseProgram(self.programID)

    def setBool(self, UniformName, value):
        glUniform1i(glGetUniformLocation(self.programID, UniformName), (value))

    def setInt(self,UniformName,value):
        glUniform1i(glGetUniformLocation(self.programID,UniformName),value)

    def setFloat(self, UniformName, value):
        glUniform1f(glGetUniformLocation(self.programID, UniformName),value)

    def setMat4(self, UniformName, mat):
        #Transpose is set to GL_TRUE since OpenGL is column major wheras numpy is row major
        glUniformMatrix4fv(glGetUniformLocation(self.programID, UniformName),1,GL_TRUE,mat)

    def setVec3(self, UniformName, vec3):
        glUniform3f(glGetUniformLocation(self.programID, UniformName), vec3[0],vec3[1],vec3[2])

