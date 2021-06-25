import json
import imageio
import numpy as np

def saveJson(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def loadJson(path):
    with open(path, 'r') as f:
        return json.load(f)

def loadDepth(path):
    # scale down 1000 to convert to original depth
    return imageio.imread(path).astype(np.float32)/1000.

def saveDepth(path, im):
    # scale up 1000 to convert to uint16
    imageio.imwrite(path, (im*1000).astype(np.uint16))