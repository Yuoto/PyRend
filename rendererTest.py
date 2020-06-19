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
from utiles.transform import toExtMat, SO3toSE3, setTranslation, toHomo
import glfw
import math, random
import numpy as np
import cv2
from ctypes import  sizeof, c_void_p,c_float
random.seed(0)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

def getRandomPose(elevationRange, azimuthRange, inplaneRange):
    azimuth = random.uniform(azimuthRange[0], azimuthRange[1])
    elevation = random.uniform(elevationRange[0], elevationRange[1])
    inplane = random.uniform(inplaneRange[0], inplaneRange[1])

    return [elevation, azimuth, inplane]

'''
def draw_custom(shader, numpy_vertices, indices, vao, cam, extrinsic, modelmat, width):
    shader.use()
    shader.setMat4('intrinsic', cam.OpenGLperspective)
    shader.setMat4('extrinsic', extrinsic)
    shader.setMat4('model', modelmat)

    gl.glBindVertexArray(vao)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    gl.glLineWidth(width)
    gl.glDrawElements(gl.GL_LINES, indices.size, gl.GL_UNSIGNED_INT, c_void_p(0))
    gl.glBindVertexArray(0)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    convert_xz = np.eye(4)
    convert_xz[0, 0] = -1
    pix_pts = np.dot(cam.intrinsic, np.dot(convert_xz, numpy_vertices))
    pix_pts = pix_pts[:2, :] / pix_pts[2, :]
    pix_pts = pix_pts.astype(np.int32).T

    return pix_pts
'''


def drawAxis(objExtrinsic,modelMat,camera,length=0.1):
    """

    :param objExtrinsic: view matrix (Extrinsic matrix)
    :param modelMat: model matrix (object space -> world space)
    :param camera: camera object
    :return: 4 x 3 axis camera coordinate, 4 x 2 axis camera screen coordinate
    """

    axis = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]
                     ], dtype=np.float32)*length

    axisHomo = toHomo(axis)
    axisCamCoord = np.dot(objExtrinsic, np.dot(modelMat, axisHomo))
    axisPix = camera.project(axisCamCoord.T)

    return axisCamCoord.T, axisPix.T

def showPointCloud(pcd,axisCamCoord):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], marker='.', s=1, c='black')

    ax.plot(axisCamCoord[(0,1), 0], axisCamCoord[(0,1), 1], axisCamCoord[(0,1), 2], marker='.', c='red')
    ax.plot(axisCamCoord[(0,2), 0], axisCamCoord[(0,2), 1], axisCamCoord[(0,2), 2], marker='.', c='green')
    ax.plot(axisCamCoord[(0,3), 0], axisCamCoord[(0,3), 1], axisCamCoord[(0,3), 2], marker='.', c='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Z Label')
    plt.show()


def main():
    # Shader info (Use absolute path)
    vShaderPath = os.path.join(os.path.dirname(__file__),"renderer/shader/rendererShader.vs")
    fShaderPath = os.path.join(os.path.dirname(__file__),"renderer/shader/rendererShader.fs")
    vShaderLampPath = os.path.join(os.path.dirname(__file__),"renderer/shader/lamp.vs")
    fShaderLampPath = os.path.join(os.path.dirname(__file__),"renderer/shader/lamp.fs")
    vShaderTightBoxPath = os.path.join(os.path.dirname(__file__),"renderer/shader/TightBox.vs")
    fShaderTightBoxPath = os.path.join(os.path.dirname(__file__),"renderer/shader/TightBox.fs")
    vShaderGyoPath = os.path.join(os.path.dirname(__file__), "renderer/shader/gyo.vs")
    fShaderGyoPath = os.path.join(os.path.dirname(__file__), "renderer/shader/gyo.fs")

    # Model info
    #modelPath = '/home/yuoto/AR/estimation/models/obj_02.ply'
    modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack_dataset/data/models/dragon/geometry.ply'
    #modelPath = '/home/yuoto/practice/OpenGL_Practice/suit/nanosuit.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/OPT/Model3D/bike/bike.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack+/dragon/Drogon.obj'

    # === If used ShapeNet model, put .mtl and texture file (.jpg) in the same directory that contains .obj file
    #modelPath = '/home/yuoto/AR/Renderer/3dmodel/cam/7bff4fd4dc53de7496dece3f86cb5dd5.obj'
    #modelPath1 = r'/home/yuoto/AR/SurgeryProject/Brain_blender/CAD/brain_clean/demo_head_clean.obj'
    #modelPath2= r'/home/yuoto/AR/SurgeryProject/Brain_blender/CAD/brain_clean/demo_head_transparent.obj'
    #modelPath = r'/home/yuoto/AR/SurgeryProject/Brain_blender/CAD/model/print/demo_multi_surgery_printed_milli_edited.obj'
    #modelPath = r'/home/yuoto/AR/Renderings/Renderer/mesh_20.obj'
    #modelPath = 'D:\MultimediaIClab\AR\Rendering\sss.obj'

    # Setup Imgui context
    GUI = False
    if GUI:
        imgui.create_context()

    # Window
    SCR_WIDTH = 960
    SCR_HEIGHT = 540

    mwindow = Window(windowSize=(SCR_WIDTH, SCR_HEIGHT), windowName='Renderer Test', visible=True)
    if GUI:
        app_window = GlfwRenderer(mwindow.window)

    # Light info
    mlight1 = Light()

    # Camera info
    focal = (540.685, 540.685)
    center = (479.75, 269.75)


    mcam1 = Camera([SCR_WIDTH, SCR_HEIGHT], focal, center, near = 0.01,far = 100)

    mrenderer = Renderer(mlight1, mcam1, mwindow, modelPath, vShaderPath, fShaderPath, vShaderLampPath,
                          fShaderLampPath,
                          vShaderTightBoxPath, fShaderTightBoxPath)
    '''
    mrenderer1 = Renderer(mlight1, mcam1, mwindow, modelPath1, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,
                         vShaderTightBoxPath, fShaderTightBoxPath)
    mrenderer2 = Renderer(mlight1, mcam1, mwindow, modelPath2, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,
                         vShaderTightBoxPath, fShaderTightBoxPath)
    '''
    #v = mrenderer.get_vertex_buffer(attribute='position').reshape((-1, 3))


    #for i in range(200):
    while not glfw.window_should_close(mwindow.window):
        # gui set up
        if GUI:
            imgui.new_frame()

        # inputs
        glfw.poll_events()
        if GUI:
            app_window.process_inputs()
        mwindow.clearWindow((0., 0., 0.))


        curT = time.time()

        ceta = math.radians(curT * 1000)


        azimuth = np.radians(ceta)
        elevation = np.radians(0)

        #azimuth = np.pi*(np.sin(ceta))
        #elevation = np.pi*(np.sin(ceta/2))

        light_radius = 0.5
        light_ceta =  math.radians(50*curT)
        lightPos = np.array([light_radius * np.cos(light_ceta) * np.cos(light_ceta), light_radius * np.sin(light_ceta) + 0.15,
                             light_radius * np.cos(light_ceta) * np.sin(light_ceta)])
        radius = 0.3

        # ================================================================================================================
        # Usually, when using outside-in tracking (i.e. concerning about object pose), the camera is always located at the center
        # However if we were to do view point sampling, it is better to treat the object at the center and all we concern about is the camera pose
        # At that time, camera position has to be defined (better using spherical coordinate
        camPos = np.array([radius*np.cos(elevation)*np.sin(azimuth), radius*np.sin(elevation), radius*np.cos(elevation)*np.cos(azimuth)])


        # set model pose & draw


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


        #modelRot = np.array([math.radians(curT * 200), 0, 0])
        #modelTrans = np.array([0, 0, 0])
        #objExtrinsic = toExtMat(modelRot, modelTrans, PoseParameterModel='Eulerzyx', isRadian=True)


        objExtrinsic = mcam1.GetCameraViewMatrix(up=[0, 1, 0], eye=camPos, at=[0, 0, 0], inplane=np.radians(0),isRadian=True)
        modelMat = setTranslation(modelMat,np.array([0, 0, 0]))


        # set light properties (remember to call updateLight())
        mlight1.setStrength(0.35)
        mlight1.setColor(3 * [1.])
        mlight1.setAttenuation(True)
        mlight1.setDirectional(True)
        mlight1.setPos(objExtrinsic.dot(np.concatenate((lightPos,np.ones(1)))))
        mrenderer.updateLight()
        #mrenderer1.updateLight()
        #mrenderer2.updateLight()


        '''
        rgb1, im_depth = mrenderer1.draw(modelMat, objExtrinsic, objExtrinsic, drawLamp=False, drawBox=False, linearDepth=True)
        mwindow.clearWindow((0.3, 0.3, 0.3))
        rgb2, im_depth = mrenderer2.draw(modelMat, objExtrinsic, objExtrinsic, drawLamp=False, drawBox=False,
                                     linearDepth=True)
        '''
        # ===========================
        #   Caution: need to clarify what camera coordinate is the extrinsic matrix transformed to
        #   If OpenCV pose estimation is used, than the camera coordinate is  z forward/ -Y up, then no convertYZ is needed
        #   If OpenGL is used to render (and ext mat transformed to coordinate to opencv cam), then it requires the camera coordinate to be -z forward/ Y up, hence a convertYZ is needed
        #  Also, Y up, -z forward  3D object file format is needed (or there is a need for axis conversion)
        # ===========================
        rgb, im_depth = mrenderer.draw(modelMat, objExtrinsic, objExtrinsic, drawLamp=False, drawBox=False,
                                     linearDepth=True)

        '''
         #===========================
        #   Caution: need to clarify what camera coordinate is the extrinsic matrix transformed to
        #   If OpenCV pose estimation is used, than the camera coordinate is  z forward/ -Y up, then no convertYZ is needed
        #   If OpenGL is used to render (and ext mat transformed to coordinate to opencv cam), then it requires the camera coordinate to be -z forward/ Y up, hence a convertYZ is needed
        #  Also, Y up, -z forward  3D object file format is needed (or there is a need for axis conversion)
        #=========================== 
        
        #===== 4 x N model space CAD pcd
        v_homo = toHomo(v)
        cam_coord = np.dot(objExtrinsic, np.dot(modelMat, v_homo))

        pix_pts = mcam1.project(cam_coord.T)


        # ===== Back project N x 2 camera screen --> N x 3 camera space CAD & jts pixels
        mask = im_depth != 0
        cloud = mcam1.backproject(im_depth,mask)
        pix_cloud = mcam1.project(cloud)


        axisCamCoord, axisPix = drawAxis(objExtrinsic,modelMat,mcam1)
        #showPointCloud(cloud,axisCamCoord)



        rgb[pix_cloud[:, 1], pix_cloud[:, 0]] = np.array([0, 255,0])

        cv2.imshow('jts', rgb)
        if cv2.waitKey(1) == 27:
            break
        '''

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


        '''
        imageio.imsave('output/shot/rendered{:d}.png'.format(i),rgb1)

        imageio.imsave('output/transparent/rendered{:d}.png'.format(i), rgb2)
        '''
        mwindow.updateWindow()
        # ===================================================

    if GUI:
        app_window.shutdown()
    glfw.terminate()


    return


if __name__ == '__main__':
    main()
