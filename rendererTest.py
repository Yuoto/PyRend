import time
import ctypes
import sys, os

sys.path.append(os.path.dirname(__file__))
from renderer.renderer import Light, Window, Renderer
from camera.camera import Camera
from utiles.transform import toExtMat, SO3toSE3
import glfw
import math, random
import numpy as np
import cv2
from scipy.misc import imread, imsave

random.seed(0)


def draw_box_without_OpenGL(renderer, img, modelMat, ModelExt):
    '''

    :param renderer: renderer object
    :param img: image to draw
    :param modelMat: model matrix
    :return: ModelExt: model intrinsic matrix
    '''
    tightBox = renderer.get3DTightBox()
    pixels = np.zeros((8, 2), dtype=int)
    for j, xc in enumerate(tightBox):
        pixels[j] = renderer.camera.project(np.dot(ModelExt, np.dot(modelMat, np.append(xc, [1.0], axis=0))),
                                            isOpenCV=True,
                                            invertX=True)
    cv2.line(img, tuple(pixels[0]), tuple(pixels[1]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[2]), tuple(pixels[3]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[4]), tuple(pixels[5]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[6]), tuple(pixels[7]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[0]), tuple(pixels[4]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[1]), tuple(pixels[5]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[3]), tuple(pixels[7]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[2]), tuple(pixels[6]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[1]), tuple(pixels[3]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[5]), tuple(pixels[7]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[0]), tuple(pixels[2]), color=(255, 255, 255), thickness=1)
    cv2.line(img, tuple(pixels[4]), tuple(pixels[6]), color=(255, 255, 255), thickness=1)

    return img


def getRandomPose(elevationRange, azimuthRange, inplaneRange):
    azimuth = random.uniform(azimuthRange[0], azimuthRange[1])
    elevation = random.uniform(elevationRange[0], elevationRange[1])
    inplane = random.uniform(inplaneRange[0], inplaneRange[1])

    return [elevation, azimuth, inplane]


def main():
    # Shader info (Use absolute path)
    vShaderPath = '/home/yuoto/AR/Renderer/renderer/shader/rendererShader.vs'
    fShaderPath = '/home/yuoto/AR/Renderer/renderer/shader/rendererShader.fs'
    vShaderLampPath = '/home/yuoto/AR/Renderer/renderer/shader/lamp.vs'
    fShaderLampPath = '/home/yuoto/AR/Renderer/renderer/shader/lamp.fs'
    vShaderTightBoxPath = '/home/yuoto/AR/Renderer/renderer/shader/TightBox.vs'
    fShaderTightBoxPath = '/home/yuoto/AR/Renderer/renderer/shader/TightBox.fs'

    # Model info
    # modelPath = '/home/yuoto/AR/estimation/models/obj_02.ply'
    # modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack_dataset/data/models/dragon/geometry.ply'
    modelPath = '/home/yuoto/practice/OpenGL_Practice/suit/nanosuit.obj'
    # modelPath = '/home/yuoto/AR/tracking/datasets/OPT/Model3D/bike/bike.obj'
    # modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack+/dragon/Drogon.obj'

    # === If used ShapeNet model, put .mtl and texture file (.jpg) in the same directory that contains .obj file
    # modelPath = '/home/yuoto/AR/Renderer/3dmodel/1a4216ac5ffbf1e89c7ce4b816b39bd0.obj'



    # Window
    SCR_WIDTH = 960
    SCR_HEIGHT = 540
    mwindow = Window((SCR_WIDTH, SCR_HEIGHT), 'Renderer Test')

    # Light info
    mlight1 = Light()

    # Camera info
    focal = (540.685, 540.685)
    center = (479.75, 269.75)
    mcam1 = Camera([SCR_WIDTH, SCR_HEIGHT], focal, center)

    mrenderer = Renderer(mlight1, mcam1, mwindow, modelPath, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,
                         vShaderTightBoxPath, fShaderTightBoxPath)

    while not glfw.window_should_close(mwindow.window):
        # for i in range(50):
        # inputs
        mwindow.processInput()

        mwindow.clearWindow((0., 0., 0.))
        curT = time.time()

        ceta = math.radians(curT * 50)

        lightPos = np.array(
            [0,0,1])
        radius = 1


        azimuth = np.pi*(np.sin(ceta)+1)
        elevation = np.pi*(np.sin(ceta/2)+1)/2

        # ================================================================================================================
        # Usually, when using outside-in tracking (i.e. concerning about object pose), the camera is always located at the center
        # However if we were to do view point sampling, it is better to treat the object at the center and all we concern about is the camera pose
        # At that time, camera position has to be defined (better using spherical coordinate
        camPos = np.array([radius*np.cos(elevation)*np.cos(azimuth), radius*np.sin(elevation), radius*np.cos(elevation)*np.sin(azimuth)])

        # set light properties (remember to call updateLight())
        mlight1.setStrength(0.5)
        mlight1.setColor(3 * [1.])
        mlight1.setAttenuation(True)
        mlight1.setDirectional(True)
        mlight1.setPos(lightPos)
        mrenderer.updateLight()

        # set model pose & draw
        lightRot = np.array([0, 0, 0])
        lightTrans = lightPos

        # Dataset 3D model scale (m)
        modelScale = 0.03
        modelMat = np.diag(3 * [modelScale] + [1.])

        # ==============================================================================================
        # When sampling viewpoints, model is always at the origin, and hence object pose is not needed
        '''
        modelRot = np.array([math.radians(-90), 0, ceta])
        modelTrans = np.array([0, 0, depth])
        ModelExt = toExtMat(modelRot, modelTrans, PoseParameterModel='Eulerzyx', isRadian=True)
        '''

        ModelExt = mcam1.GetCameraViewMatrix(up=[0, 1, 0], eye=camPos, at=[0, 0, 0])

        LightExt = toExtMat(lightRot, lightTrans, PoseParameterModel='Eulerzyx', isRadian=True)



        rgb, im_depth = mrenderer.draw(modelMat, ModelExt, LightExt, drawLamp=True, drawBox=True, linearDepth=False)

    # imsave('box.png',rgb)
    # img = imread('box.png')

    # img = draw_box_without_OpenGL(mrenderer,img,modelMat,ModelExt)
    # imsave('box.png', img)


    glfw.terminate()

    return


if __name__ == '__main__':
    main()
