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
from ctypes import  sizeof, c_void_p,c_float
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

def ToCamCoordinates(camera, depth):
    constant_x = 1.0 / camera.focal[0]
    constant_y = 1.0 / camera.focal[1]
    row, col = depth.shape
    coords = np.indices((row, col)).swapaxes(0, 2).swapaxes(0, 1)

    output = np.zeros((row,col, 3),dtype=np.float32)
    output[:, :, 0] = (coords[:, :, 0] - camera.center[1]) * depth * constant_y
    output[:, :, 1] = (coords[:, :, 1] - camera.center[0]) * depth * constant_x
    output[:, :, 2] = depth

    return output

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
    #modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack_dataset/data/models/dragon/geometry.ply'
    #modelPath = '/home/yuoto/practice/OpenGL_Practice/suit/nanosuit.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/OPT/Model3D/bike/bike.obj'
    #modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack+/dragon/Drogon.obj'

    # === If used ShapeNet model, put .mtl and texture file (.jpg) in the same directory that contains .obj file
    #modelPath = '/home/yuoto/AR/Renderer/3dmodel/cam/7bff4fd4dc53de7496dece3f86cb5dd5.obj'
    modelPath = r'/home/yuoto/AR/Renderings/Renderer/mesh_0.obj'
    #modelPath = 'D:\MultimediaIClab\AR\Rendering\sss.obj'
    jts = np.load(r'/home/yuoto/AR/Renderings/Renderer/mesh_joint0.npy')
    jts_model = np.zeros((24, 4))
    jts_model[:, 3] = np.array(24 * [1])
    jts_model[:, :3] = jts
    jts_model = jts_model.T

    # Setup Imgui context
    GUI = False
    if GUI:
        imgui.create_context()

    # Window
    SCR_WIDTH = 960
    SCR_HEIGHT = 540
    #SCR_WIDTH = 424
    #SCR_HEIGHT = 512
    mwindow = Window(windowSize=(SCR_WIDTH, SCR_HEIGHT), windowName='Renderer Test', visible=True)
    if GUI:
        app_window = GlfwRenderer(mwindow.window)

    # Light info
    mlight1 = Light()

    # Camera info
    focal = (540.685, 540.685)
    center = (479.75, 269.75)
    #center = (269.75, 479.75)
    #focal = (366.458,366.736)
    #center = (207.470-1,254.026-1)
    mcam1 = Camera([SCR_WIDTH, SCR_HEIGHT], focal, center, near = 0.01,far = 10000)

    mrenderer = Renderer(mlight1, mcam1, mwindow, modelPath, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,
                         vShaderTightBoxPath, fShaderTightBoxPath)

    v = mrenderer.get_vertex_buffer(attribute='position').reshape((-1, 3))


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

        azimuth = np.radians(0)
        elevation = np.radians(30)

        #azimuth = np.pi*(np.sin(ceta))
        #elevation = np.pi*(np.sin(ceta/2))

        light_radius = 2
        light_ceta =  math.radians(50*curT)
        lightPos = np.array([light_radius * np.cos(light_ceta) * np.cos(light_ceta), light_radius * np.sin(light_ceta),
                             light_radius * np.cos(light_ceta) * np.sin(light_ceta)])
        radius = 4

        # ================================================================================================================
        # Usually, when using outside-in tracking (i.e. concerning about object pose), the camera is always located at the center
        # However if we were to do view point sampling, it is better to treat the object at the center and all we concern about is the camera pose
        # At that time, camera position has to be defined (better using spherical coordinate
        camPos = np.array([radius*np.cos(elevation)*np.sin(azimuth), radius*np.sin(elevation), radius*np.cos(elevation)*np.cos(azimuth)])


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


        #modelRot = np.array([math.radians(curT * 200), 0, 0])
        #modelTrans = np.array([0, 0, 0])
        #ModelExt = toExtMat(modelRot, modelTrans, PoseParameterModel='Eulerzyx', isRadian=True)


        ModelExt = mcam1.GetCameraViewMatrix(up=[0, 1, 0], eye=camPos, at=[0, 0, 0], inplane=np.radians(0),isRadian=True)
        #obj_move_Ext = setTranslation(ModelExt,np.array([0.5*np.cos(light_ceta), 0, -radius]))

        LightExt = toExtMat(lightRot, lightTrans, PoseParameterModel='Eulerzyx', isRadian=True)


        rgb, im_depth = mrenderer.draw(modelMat, ModelExt, LightExt, drawLamp=False, drawBox=False, linearDepth=True)


        #===== 4 x N model space CAD pcd
        v_model = np.zeros((len(v[:,0]), 4))
        v_model[:, 3] = np.array(len(v[:,0]) * [1])
        v_model[:, :3] = v
        v_model = v_model.T

        # ===== 4 x N model space axis(for navigation) pcd
        gyo_x = np.array([[0,0,0],
                        [1,0,0]],dtype=np.float32)
        gyo_y = np.array([[0, 0, 0],
                          [0, 1, 0]], dtype=np.float32)
        gyo_z = np.array([[0, 0, 0],
                          [0, 0, 1]], dtype=np.float32)
        gyo_x_homo = np.concatenate((gyo_x,np.ones((gyo_x.shape[0],1),dtype=np.float32)),axis=1).T
        gyo_y_homo = np.concatenate((gyo_y,np.ones((gyo_y.shape[0],1),dtype=np.float32)),axis=1).T
        gyo_z_homo = np.concatenate((gyo_z,np.ones((gyo_z.shape[0],1),dtype=np.float32)),axis=1).T


        # ===== 4 x N camera space CAD & jts pcd
        gyo_cam_coord_x = np.dot(ModelExt, np.dot(modelMat, gyo_x_homo))
        gyo_cam_coord_y = np.dot(ModelExt, np.dot(modelMat, gyo_y_homo))
        gyo_cam_coord_z = np.dot(ModelExt, np.dot(modelMat, gyo_z_homo))
        cam_coord = np.dot(ModelExt, np.dot(modelMat, v_model))
        cam_coord_jts = np.dot(ModelExt, np.dot(modelMat, jts_model))


        # ===== N x 2 camera screen space CAD & jts pixels
        pix_gyo_x = mcam1.project(gyo_cam_coord_x.T)
        pix_gyo_y = mcam1.project(gyo_cam_coord_y.T)
        pix_gyo_z = mcam1.project(gyo_cam_coord_z.T)
        pix_pts = mcam1.project(cam_coord.T)
        pix_jts = mcam1.project(cam_coord_jts.T)


        # ===== Back project N x 2 camera screen --> N x 3 camera space CAD & jts pixels
        mask = im_depth != 0
        cloud = mcam1.backproject(im_depth,mask)
        cloud_jts = mcam1.backproject_points(cam_coord_jts.T[:,2], pix_jts)

        # ===== project to pixel again the back-projected N x 3 camera space
        pix_cloud = mcam1.project(cloud)
        pix_cloud_jts = mcam1.project(cloud_jts)


        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        cam_coord = cam_coord.T
        cam_coord_jts = cam_coord_jts.T
        gyo_cam_coord_x = gyo_cam_coord_x.T
        print(gyo_cam_coord_x)
        gyo_cam_coord_y = gyo_cam_coord_y.T
        gyo_cam_coord_z = gyo_cam_coord_z.T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cloud[:, 0],cloud[:, 1],cloud[:, 2], marker='.', s=1, c='blue')
        ax.scatter(cloud_jts[:, 0], cloud_jts[:, 1], cloud_jts[:, 2], marker='.', s=50, c='green')
        ax.scatter(cam_coord_jts[:, 0], cam_coord_jts[:, 1], cam_coord_jts[:, 2], marker='.', s=25, c='red')
        ax.scatter(cam_coord[:, 0], cam_coord[:, 1], cam_coord[:, 2], marker='.', s=1, c='black')
        ax.plot(gyo_cam_coord_x[:, 0], gyo_cam_coord_x[:, 1], gyo_cam_coord_x[:, 2], marker='.',  c='red')
        ax.plot(gyo_cam_coord_y[:, 0], gyo_cam_coord_y[:, 1], gyo_cam_coord_y[:, 2], marker='.', c='green')
        ax.plot(gyo_cam_coord_z[:, 0], gyo_cam_coord_z[:, 1], gyo_cam_coord_z[:, 2], marker='.', c='blue')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Z Label')
        plt.show()


        rgb[pix_cloud[:, 1], pix_cloud[:, 0]] = np.array([0, 255,0])
        rgb[pix_cloud_jts[:, 1], pix_cloud_jts[:, 0]] = np.array([255, 0, 0])
        #rgb[pix_pts[:, 1], pix_pts[:, 0]] = np.array([255, 255, 255])
        #rgb[pix_jts[:, 1], pix_jts[:, 0]] = np.array([0, 0, 255])
        rgb=cv2.line(np.float32(rgb), tuple(pix_gyo_x[0]), tuple(pix_gyo_x[1]), color=(0, 0, 255),
                 thickness=1).astype(np.uint8)
        rgb = cv2.line(np.float32(rgb), tuple(pix_gyo_y[0]), tuple(pix_gyo_y[1]), color=(0, 255, 0),
                       thickness=1).astype(np.uint8)
        rgb = cv2.line(np.float32(rgb), tuple(pix_gyo_z[0]), tuple(pix_gyo_z[1]), color=(255, 0, 0),
                       thickness=1).astype(np.uint8)

        cv2.imshow('jts', rgb)
        if cv2.waitKey(1) == 27:
            break


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
