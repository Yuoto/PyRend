import numpy as np

class Light:
    def __init__(self, position=np.array([1., 0., -1.]),
                 ambient=np.array([1., 1., 1.]),
                 diffuse=np.array([1., 1., 1.]),
                 specular=np.array([1., 1., 1.]),
                 constant=1., linear=14.e-4, quadratic=7.e-6, directional=True, attenuation=True):
        '''
        credit to: https://learnopengl.com/Lighting/Materials

        :param position: position of the light
        :param ambient:
        :param diffuse:
        :param specular:
        :param constant:
        :param linear:
        :param quadratic:
        :param directional: Enable Directional light
        :param attenuation: Enable Attenuation
        '''
        self.position = position
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.enableDirectional = directional
        self.enableAttenuation = attenuation

    def setPosition(self, position):
        self.position = position

    def setConstant(self,constant):
        self.constant = constant

    def setLinear(self,linear):
        self.linear = linear

    def setQuadratic(self,quadratic):
        self.quadratic = quadratic

    def setAmbient(self,ambient):
        self.ambient = ambient

    def setDiffuse(self,diffuse):
        self.diffuse = diffuse

    def setSpecular(self,specular):
        self.specular = specular

    def setDirectional(self, directional):
        self.enableDirectional = directional

    def setAttenuation(self, attenuation):
        self.enableAttenuation = attenuation