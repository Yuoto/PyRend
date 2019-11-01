import time
import ctypes
import sys, os
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
sys.path.append(os.path.dirname(__file__))
from window.window import Window
from renderer.renderer import Light, Renderer
from camera.camera import Camera
from utiles.transform import toExtMat, SO3toSE3, setTranslation
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
                                            isOpenCV=False
                                            )
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
    vShaderPath = "D:\MultimediaIClab\AR\Rendering\pyrend/renderer\shader/rendererShader.vs"
    fShaderPath = 'D:\MultimediaIClab\AR\Rendering\pyrend/renderer\shader/rendererShader.fs'
    vShaderLampPath = 'D:\MultimediaIClab\AR\Rendering\pyrend/renderer\shader\lamp.vs'
    fShaderLampPath = 'D:\MultimediaIClab\AR\Rendering\pyrend/renderer\shader\lamp.fs'
    vShaderTightBoxPath = 'D:\MultimediaIClab\AR\Rendering\pyrend/renderer\shader\TightBox.vs'
    fShaderTightBoxPath = 'D:\MultimediaIClab\AR\Rendering\pyrend/renderer\shader\TightBox.fs'

    # Model info
    #modelPath = '/home/yuoto/AR/estimation/models/obj_02.ply'
    modelPath = 'C:/Users\Win10-PC\Downloads/3dmodels/3dmodels\dragon\geometry.ply'
    #modelPath = '/home/yuoto/practice/OpenGL_Practice/suit/nanosuit.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/OPT/Model3D/bike/bike.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack+/dragon/Drogon.obj'

    # === If used ShapeNet model, put .mtl and texture file (.jpg) in the same directory that contains .obj file
    #modelPath = '/home/yuoto/AR/Renderer/3dmodel/cam/7bff4fd4dc53de7496dece3f86cb5dd5.obj'
    #modelPath = '/home/yuoto/Downloads/mesh_0.obj'
    #modelPath = 'D:\MultimediaIClab\AR\Rendering\sss.obj'

    # Setup Imgui context
    GUI = False
    if GUI:
        imgui.create_context()

    # Window
    SCR_WIDTH = 1280
    SCR_HEIGHT = 720
    mwindow = Window((SCR_WIDTH, SCR_HEIGHT), 'Renderer Test')
    if GUI:
        app_window = GlfwRenderer(mwindow.window)

    # Light info
    mlight1 = Light()

    # Camera info
    focal = (540.685, 540.685)
    center = (479.75, 269.75)
    mcam1 = Camera([SCR_WIDTH, SCR_HEIGHT], focal, center,far=100)

    mrenderer = Renderer(mlight1, mcam1, mwindow, modelPath, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,
                         vShaderTightBoxPath, fShaderTightBoxPath)

    #for i in range(200):
    while not glfw.window_should_close(mwindow.window):
        # gui set up
        if GUI:
            imgui.new_frame()

        # inputs
        glfw.poll_events()
        if GUI:
            app_window.process_inputs()
        mwindow.clearWindow((0.1, 0.1,0.1))


        curT = time.time()

        ceta = math.radians(curT * 1000)

        lightPos = np.array(
            [0,1,0])
        radius = 1

        azimuth = np.radians(ceta)
        elevation = np.radians(0)

        #azimuth = np.pi*(np.sin(ceta))
        #elevation = np.pi*(np.sin(ceta/2))

        light_radius = 2
        light_ceta =  math.radians(50*curT)
        lightPos = np.array([light_radius * np.cos(light_ceta) * np.cos(light_ceta), light_radius * np.sin(light_ceta),
                             light_radius * np.cos(light_ceta) * np.sin(light_ceta)])
        radius = 1

        # ================================================================================================================
        # Usually, when using outside-in tracking (i.e. concerning about object pose), the camera is always located at the center
        # However if we were to do view point sampling, it is better to treat the object at the center and all we concern about is the camera pose
        # At that time, camera position has to be defined (better using spherical coordinate
        camPos = np.array([radius*np.cos(elevation)*np.cos(azimuth), radius*np.sin(elevation), radius*np.cos(elevation)*np.sin(azimuth)])


        # set light properties (remember to call updateLight())
        mlight1.setStrength(0.4)
        mlight1.setColor(3 * [1.])
        mlight1.setAttenuation(True)
        mlight1.setDirectional(True)
        mlight1.setPos(lightPos)
        mrenderer.updateLight()

        # set model pose & draw
        lightRot = np.array([0, 0, 0])
        lightTrans = lightPos + np.array([0, -1.5, -light_radius])

        # Dataset 3D model scale (m)
        modelScale = 1
        modelMat = np.diag(3 * [modelScale] + [1.])

        # ==============================================================================================
        # When sampling viewpoints, model is always at the origin, and hence object pose is not needed

        #v = mrenderer.get_vertex_buffer(attribute='position', mesh_id=None)
        #v = v + v*(np.sin(ceta))/10
        #mrenderer.set_vertex_buffer(pos = v)
        #ind = mrenderer.get_vertex_buffer(attribute='indices', mesh_id=None)
        #mrenderer.set_vertex_buffer(indices=ind[::-1])


        #modelRot = np.array([math.radians(0), 0, 0])
        #modelTrans = np.array([0, 0, -1])
        #ModelExt = toExtMat(modelRot, modelTrans, PoseParameterModel='Eulerzyx', isRadian=True)


        ModelExt = mcam1.GetCameraViewMatrix(up=[0, 1, 0], eye=camPos, at=[0, 0, 0], inplane=np.radians(0),isRadian=True)
        #ModelExt = setTranslation(ModelExt,np.array([0, 0, -radius]))

        LightExt = toExtMat(lightRot, lightTrans, PoseParameterModel='Eulerzyx', isRadian=True)


        rgb, im_depth = mrenderer.draw(modelMat, ModelExt, LightExt, drawLamp=False, drawBox=True, linearDepth=True)

        # ====================================== GUI
        if GUI:
            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File", True):

                    clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", 'Cmd+Q', False, True
                    )

                    if clicked_quit:
                        exit(1)

                    imgui.end_menu()
                imgui.end_main_menu_bar()

            imgui.show_test_window()

            # print(app_window.io.mouse_down[0])
            imgui.begin("Custom window", True)
            imgui.text("Bar")
            imgui.text_colored("Eggs", 0.2, 1., 0.)
            imgui.end()

            imgui.show_test_window()
            imgui.render()
            app_window.render(imgui.get_draw_data())

        mwindow.updateWindow()
        #outputframe = gl.glReadPixels(0, 0, mwindow.windowSize[0], mwindow.windowSize[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        #outputframe = np.fromstring(outputframe, np.uint8)
        #outputframe = np.flipud(np.reshape(outputframe, (mwindow.windowSize[1], mwindow.windowSize[0], 3)))

        # ===================================================

    if GUI:
        app_window.shutdown()
    glfw.terminate()


    return


if __name__ == '__main__':
    main()
