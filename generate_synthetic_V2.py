import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.dirname(__file__))
from renderer.renderer import Light,Window,Renderer
from camera.camera import Camera
from utiles.transform import toExtMat, SO3toSE3
from utiles.inout import  saveJson, loadJson
from utiles.dataSetUitles import computeBoundingBox, crop
import glfw
import math,random
import numpy as np
from scipy.misc import imread,imsave
from tqdm import tqdm
import cv2


random.seed(0)

def getRandomPose(elevationRange,azimuthRange,inplaneRange):

    azimuth = random.uniform(azimuthRange[0],azimuthRange[1])
    elevation = random.uniform(elevationRange[0],elevationRange[1])
    inplane = random.uniform(inplaneRange[0],inplaneRange[1])

    return [elevation,azimuth,inplane]






def main():

    # Shader info
    vShaderPath = '/home/yuoto/AR/tracking/algorithms/ECCV2019Tracking/renderer/shader/rendererShader.vs'
    fShaderPath = '/home/yuoto/AR/tracking/algorithms/ECCV2019Tracking/renderer/shader/rendererShader.fs'
    vShaderLampPath = '/home/yuoto/AR/tracking/algorithms/ECCV2019Tracking/renderer/shader/lamp.vs'
    fShaderLampPath = '/home/yuoto/AR/tracking/algorithms/ECCV2019Tracking/renderer/shader/lamp.fs'
    vShaderTightBoxPath ='/home/yuoto/AR/tracking/algorithms/ECCV2019Tracking/renderer/shader/TightBox.vs'
    fShaderTightBoxPath = '/home/yuoto/AR/tracking/algorithms/ECCV2019Tracking/renderer/shader/TightBox.fs'

    # Model info
    #modelPath = '/home/yuoto/AR/estimation/models/obj_02.ply'
    modelPath = '/home/yuoto/AR/tracking/datasets/deeptrack_dataset/data/models/dragon/geometry.ply'
    #modelPath = '/home/yuoto/AR/tracking/datasets/OPT/Model3D/jet/jet.obj'
    #modelPath = '/home/yuoto/practice/OpenGL_Practice/suit/nanosuit.obj'

    #training data info
    dataInfoPath = '/home/yuoto/AR/Renderer/genSyntheticConfig.json'

    dataInfo = loadJson(dataInfoPath)
    ELEVATIONRANGE = dataInfo['ELEVATIONRANGE']
    AZIMUTHRANGE = dataInfo['AZIMUTHRANGE']
    INPLANERANGE = dataInfo['INPLANERANGE']
    RADIUS = dataInfo['RADIUS']
    PERTURB_ELEV = dataInfo['PERTURB_ELEV']
    PERTURB_AZIM = dataInfo['PERTURB_AZIM']
    PERTURB_IN = dataInfo['PERTURB_IN']
    SAMPLE_QUANTITY = dataInfo['SAMPLE_QUANTITY']
    IMAGESIZE = dataInfo['IMAGESIZE']
    PRELOAD = dataInfo['PRELOAD']
    OUTPUTPATH = dataInfo['OUTPUTPATH']
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
    gtPath = os.path.join(OUTPUTPATH, 'gt.json')
    infoPath = os.path.join(OUTPUTPATH, 'info.json')
    objs = ['dragon']
    gt = {}
    info = {}





    

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


    mrenderer = Renderer(mlight1, mcam1, modelPath, vShaderPath, fShaderPath, vShaderLampPath, fShaderLampPath,vShaderTightBoxPath,fShaderTightBoxPath)

    '''
    with open(gt_path) as data_file:
        gt = json.load(data_file)

    trans = np.zeros(3,dtype=np.float32)
    rot = np.zeros(3,dtype=np.float32)

    for i in range(3):
        trans[i] = np.float32(gt['0']['vector'][str(i)])
        rot[i] = np.float32(gt['0']['vector'][str(3+i)])
    viewPointExt = toExtMat(rot, trans,PoseParameterModel='Eulerzyx',radians=True)
    '''


    if PRELOAD:
        gt = loadJson(gtPath)
        preload_count = len(gt)
        print("Continue generating... This Dataset already contains {} samples".format(preload_count))
    else:
        preload_count = 0

    for j,obj in enumerate(objs):
        for i in tqdm(range(SAMPLE_QUANTITY - preload_count)):

            idx = i + preload_count
            # inputs
            #mwindow.processInput(mcam1)
            mwindow.clearWindow((0.,0.,0.))

            #set light properties
            # Random light Position, isDirectional
            ceta = random.uniform(0, 2 * math.pi)
            lightPos = np.array([RADIUS*math.cos(ceta),random.uniform(-1,1),RADIUS*math.sin(ceta)])

            mlight1.setAttenuation(True)
            mlight1.setDirectional(random.randint(0,1))
            mlight1.setPos(lightPos)

            # set model pose & draw
            lightRot = np.array([0,0,0])
            lightTrans = lightPos




            # Dataset 3D model scale (m)
            modelScale = 1
            modelMat =  np.diag(3*[modelScale]+[1.])
            LightExt = toExtMat(lightRot,lightTrans,PoseParameterModel='Eulerzyx',radians=True)


            # viewpoint pose
            viewPointRot = getRandomPose(ELEVATIONRANGE,AZIMUTHRANGE,INPLANERANGE)
            viewPointTrans = [0, 0, -RADIUS]
            viewPointExt = toExtMat(viewPointRot, viewPointTrans, PoseParameterModel='Eulerzyx', radians=False)

            # Pertrubation pose
            relateRot = getRandomPose(PERTURB_ELEV, PERTURB_AZIM,PERTURB_IN)
            relateRotExt = toExtMat(relateRot,PoseParameterModel='Eulerzyx', radians=False)
            pertrubExt = SO3toSE3(np.dot(relateRotExt[0:3,0:3],viewPointExt[0:3,0:3]),viewPointTrans)


            mrenderer.setModelMaterial(mlight1,ambient=0.5,diffuse=0.5,specular= 0.5,shininess=1.0)
            mrenderer.draw(modelMat,viewPointExt,LightExt,drawLamp=False)
            mwindow.updateWindow()
            viewPointImg = mwindow.screenShot()

            mwindow.clearWindow((0., 0., 0.))
            mrenderer.draw(modelMat, pertrubExt, LightExt, drawLamp=False)
            mwindow.updateWindow()
            pertrubImg = mwindow.screenShot()
            

            tightBox = mrenderer.get3DTightBox()
            viewPointBB, _ = computeBoundingBox(mcam1, modelMat, viewPointExt,tightBox)
            pertrubBB, _ = computeBoundingBox(mcam1, modelMat, pertrubExt, tightBox)

            viewPointImg = crop(viewPointImg, viewPointBB)
            pertrubImg = crop(pertrubImg, pertrubBB)

            viewPointImg = cv2.resize(viewPointImg,tuple(IMAGESIZE),interpolation=cv2.INTER_CUBIC)
            pertrubImg = cv2.resize(pertrubImg, tuple(IMAGESIZE),interpolation=cv2.INTER_CUBIC)

            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(pertrubImg)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(viewPointImg)
            '''

            plt.show()

            np.save(os.path.join(OUTPUTPATH, str(idx)+'.npy'),viewPointImg)
            np.save(os.path.join(OUTPUTPATH, str(idx)+'p.npy'),pertrubImg)
            #imsave(os.path.join(OUTPUTPATH, str(idx) + '.png'), viewPointImg)
            #imsave(os.path.join(OUTPUTPATH, str(idx) + 'p.png'), pertrubImg)

            gt[idx]={'R':viewPointRot,'T':viewPointTrans, 'R_rel':relateRot, 'obj':obj, '2DBB':viewPointBB.tolist(), '2DBBp':pertrubBB.tolist()}

            #Dump ground truths for every 1000 images, if the total number of training images are less then 1000, then this line should be altered.
            if (idx+1) % 1000 == 0:
                saveJson(gtPath, gt)


        info[j] = {'obj': obj,'3DTightBox': tightBox.tolist(),'focal':[540.685, 540.685], 'center': [479.75, 269.75], 'SCR_WIDTH':SCR_WIDTH, 'SCR_HEIGHT':SCR_HEIGHT,
                   'ELEVATIONRANGE':ELEVATIONRANGE,'AZIMUTHRANGE':AZIMUTHRANGE,'INPLANERANGE':INPLANERANGE,'PERTURB_ELEV':PERTURB_ELEV,'PERTURB_AZIM':PERTURB_AZIM, 'PERTURB_IN':PERTURB_IN, 'IMAGESIZE':IMAGESIZE
                   }
        saveJson(infoPath, info)

    '''
    im = mwindow.screenShot()
    orig = imread('0.png')
    imsave('render.png',im)
    blendMask = im == 0
    final = (~blendMask)*im + blendMask*orig
    imsave('blend.png', final)
    '''

    glfw.terminate()

    return


if __name__ == '__main__':

    main()



