import time
import ctypes
import sys,os
sys.path.append(os.path.dirname(__file__))
from renderer.renderer import Light,Window,Renderer
from camera.camera import Camera
from utiles.transform import toExtMat, SO3toSE3
import glfw
import math,random
import numpy as np
from scipy.misc import imread,imsave




random.seed(0)

def getRandomPose(elevationRange,azimuthRange,inplaneRange):

    azimuth = random.uniform(azimuthRange[0],azimuthRange[1])
    elevation = random.uniform(elevationRange[0],elevationRange[1])
    inplane = random.uniform(inplaneRange[0],inplaneRange[1])

    return [elevation,azimuth,inplane]






def main():

    # Shader info
    vShaderPath = '/home/yuoto/AR/Renderer/renderer/shader/rendererShader.vs'
    fShaderPath = '/home/yuoto/AR/Renderer/renderer/shader/rendererShader.fs'
    vShaderLampPath = '/home/yuoto/AR/Renderer/renderer/shader/lamp.vs'
    fShaderLampPath = '/home/yuoto/AR/Renderer/renderer/shader/lamp.fs'
    vShaderTightBoxPath ='/home/yuoto/AR/Renderer/renderer/shader/TightBox.vs'
    fShaderTightBoxPath = '/home/yuoto/AR/Renderer/renderer/shader/TightBox.fs'

    # Model info
    #modelPath = '/home/yuoto/AR/estimation/models/obj_02.ply'
    modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack_dataset/data/models/dragon/geometry.ply'
    #modelPath = '/home/yuoto/practice/OpenGL_Practice/suit/nanosuit.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/OPT/Model3D/bike/bike.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack+/dragon/Drogon.obj'

    #=== If used ShapeNet model, put .mtl and texture file (.jpg) in the same directory that contains .obj file
    #modelPath = '/home/yuoto/AR/Renderer/3dmodel/1a6a67905880e4911a4d5e0a785b0e03.obj'



    # Window
    SCR_WIDTH = 960
    SCR_HEIGHT = 540
    mwindow = Window((SCR_WIDTH, SCR_HEIGHT), 'Renderer Test')

    # Light info
    mlight1 = Light()


    # Camera info
    focal = (540.685, 540.685)
    center = (479.75, 269.75)
    mcam1 = Camera([SCR_WIDTH,SCR_HEIGHT],focal, center)

    depth = -1
    mrenderer = Renderer(mlight1, mcam1,mwindow, modelPath, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,vShaderTightBoxPath,fShaderTightBoxPath)


    while not glfw.window_should_close(mwindow.window):
    #for i in range(5):
        # inputs
        mwindow.processInput()

        mwindow.clearWindow((0., 0., 0.))
        curT = time.time()
        # set light properties

        ceta = math.radians(curT * 50)

        lightPos = np.array(
            [1.5 * math.cos(math.radians(curT * 100)), 0, depth + 1.5 * math.sin(math.radians(curT * 100))])

        mlight1.setAttenuation(True)
        mlight1.setDirectional(True)
        mlight1.setPos(lightPos)

        # set model pose & draw
        lightRot = np.array([0, 0, 0])
        lightTrans = lightPos
        modelRot = np.array([0, ceta, math.radians(90)])
        modelTrans = np.array([0, 0, depth])

        # Dataset 3D model scale (m)
        modelScale = 1
        modelMat = np.diag(3 * [modelScale] + [1.])
        LightExt = toExtMat(lightRot, lightTrans, PoseParameterModel='Eulerzyx', radians=True)

        ModelExt = toExtMat(modelRot, modelTrans, PoseParameterModel='Eulerzyx', radians=True)

        mrenderer.setModelMaterial(mlight1, ambient=0.5, diffuse=0.5, specular=0.5, shininess=1.0)
        rgb,im_depth = mrenderer.draw(modelMat, ModelExt, LightExt, drawLamp=True, drawBox=True, linearDepth=False)


        #imsave(str(i) + '.png', im_depth)

    glfw.terminate()
    
    return


if __name__ == '__main__':

    main()



