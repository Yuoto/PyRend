import numpy as np




# Tight bound,
def computeBoundingBox(cam, modelMat, pose, tightBox):
    xMin = cam.windowSize[0]
    xMax = 0
    yMin = cam.windowSize[1]
    yMax = 0
    camCoor = np.array([np.dot(pose, np.dot(modelMat, np.append(tightBox[i], [1.0]))) for i in range(8)])


    pixel = np.zeros((8,2),dtype=np.uint32)
    for i in range(8):
        pixel[i] = np.round(cam.Project(camCoor[i]))

        if pixel[i][0] <= xMin:
            xMin = pixel[i][0]
        if pixel[i][0] >= xMax:
            xMax = pixel[i][0]
        if pixel[i][1] <= yMin:
            yMin = pixel[i][1]
        if pixel[i][1] >= yMax:
            yMax = pixel[i][1]


    xCenter = (xMin + xMax) / 2
    yCenter = (yMin + yMax) / 2
    halfWidth = max((xMax - xCenter), (yMax - yCenter))
    xh = xCenter + halfWidth
    xl = xCenter - halfWidth
    yh = yCenter + halfWidth
    yl = yCenter - halfWidth

    viewPointBB = np.array([xl, xh, yl, yh], dtype=np.int32)

    return viewPointBB, pixel



def crop(im,bb):
    return  im[bb[2]:bb[3],bb[0]:bb[1],:]

def cropTensor(tensor,bb):
    return  tensor[:,bb[2]:bb[3],bb[0]:bb[1]]