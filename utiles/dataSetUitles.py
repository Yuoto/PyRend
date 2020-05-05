import numpy as np
from utiles.transform import toHomo


def checkBoundary(cam, xL, yL, BBwidth, BBheight):
    xL = xL if xL > 0 else 0
    yL = yL if yL > 0 else 0
    BBwidth = xL + BBwidth if xL + BBwidth < cam.windowSize[0] else cam.windowSize[0] - xL - 1
    BBheight = yL + BBheight if yL + BBheight < cam.windowSize[1] else cam.windowSize[1] - yL - 1

    return np.array([xL, yL, BBwidth, BBheight], dtype=np.int32)


# Tight bound
def computeBoundingBox(cam, modelMat, extMat, tightBox, scale=(2, 2)):
    """

    :param cam:
    :param modelMat:
    :param extMat:
    :param tightBox:
    :param scale:  (sx, sy)  tuple, meansing the scale of the bounding box that is to be extended from center
    :return:  BB: bounding box [xL, yL, BBwidth, BBheight]
    """
    camCoor = np.dot(extMat, np.dot(modelMat, toHomo(tightBox)))
    pixels = cam.project(camCoor.T)
    pixels_sorted = np.sort(pixels)
    xMin, yMin = pixels_sorted[0]
    xMax, yMax = pixels_sorted[-1]

    xCenter = (xMin + xMax) / 2
    yCenter = (yMin + yMax) / 2
    halfWidthX = (xMax - xCenter) * scale[0]
    halfWidthY = (yMax - yCenter) * scale[1]

    return checkBoundary(cam, xCenter - halfWidthX, yCenter - halfWidthY, 2 * halfWidthX, 2 * halfWidthY)


def crop(im, bb):
    return im[bb[2]:bb[3], bb[0]:bb[1], :]


def cropTensor(tensor, bb):
    return tensor[:, bb[2]:bb[3], bb[0]:bb[1]]
