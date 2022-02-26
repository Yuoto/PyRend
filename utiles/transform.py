import numpy as np
import math
import cv2
import logging

def getExMatFromP(p):
    """

    :param p: p [6, 1] axis angle
    :return: T: [4, 4] SE3 group
    """
    T = np.eye(4)
    T[:3, :] = getRtFromP(p)
    return T

def getPFromExMat(T):
    """

    :param T: [4, 4] SE3 group
    :return: p [6, 1] axis angle
    """
    r = getAxisAngleFromRotMat(T[:3, :3]) # [3, 1]
    p = np.concatenate((r, T[:3, 3:4]), axis=0)

    return p


def angleBetweenTwoVectors(v1, v2):

    cosA = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cosA >= 1.0:
        angle = 0
    elif cosA <= -1.0:
        angle = 180
    else:
        angle = np.arccos(cosA)/np.pi*180
    return angle


def getRtFromP(p):
    """

    :param p: [6, 1] axis angle
    :return: Rt: [3, 4] transformation matrix
    """
    Rt = np.zeros((3, 4))
    Rt[:3, :3] = getRotMatFromAxisAngle(p[:3])
    Rt[:3, 3:4] = p[3:6]

    return Rt

def getPosesFromP(p):
    """

    :param p: [N, 1] flattened axis angles
    :return: poses [N/6, 12]
    """
    assert p.shape[0] % 6 ==0
    num = int(p.shape[0] / 6)
    poses = np.zeros((num, 12))
    for i in range(num):
        poses[i,:] = getRtFromP(p[6*i: 6*i + 6]).reshape(-1, order='F')
    return poses

def getPFromPoses(poses):
    """
        param: poses [N, 12]
        return: lie algebra [6N, 1]

    layout: r11, r21, r31, r12, r22, r32, r13, r23, r33, t1, t2, t3
    """

    num = poses.shape[0]
    p = np.zeros((6*num, 1))

    for i in range(num):
        T = poses[i, :].reshape((3,4), order='F') 
        p[6*i: 6*i +3] = getAxisAngleFromRotMat(T[:3, :3])
        p[6*i + 3: 6*i + 6] = T[:3, 3:4]

    return p



def getAxisAngleFromRotMat(R):
    """
        param: R [3, 3] rotation matrix
        return: lie algebra [3, 1]

    Explanation:
    a = acos((trace(R)-1)/2)
    r = (a/2sin(a)) * [R(3, 2)-R(2, 3), R(1, 3)-R(3, 1), R(2, 1)-R(1, 2)]
    """
    r = cv2.Rodrigues(R)[0]
    # a = np.arccos((np.trace(R)-1)/2)
    #
    # if a < 1e-5:
    #     r = 0.5 * np.array([R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1]])
    # elif a > (np.pi - 1e-5):
    #     S = 0.5*(R-np.eye(3))
    #     b = np.sqrt(S[0][0]+1)
    #     c = np.sqrt(S[1][1]+1)
    #     d = np.sqrt(S[2][2]+1)
    #     if b > 1e-5:
    #         c = S[1][0] / b
    #         d = S[2][0] / b
    #     elif c > 1e-5:
    #         b = S[0][1] / c
    #         d = S[2][1] / c
    #     else:
    #         b = S[0][2] / d
    #         c = S[1][2] / d
    #     r = np.array([b, c, d])
    # else:
    #     r = (a/2/np.sin(a)) * np.array([R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1]])
    # assert np.allclose(r_cv2, r)
    return r

def getRotMatFromAxisAngle(r):

    R2 = cv2.Rodrigues(r)[0]

    I = np.eye(3)
    a = np.linalg.norm(r)
    a2 = a * a
    W = getCrossProductMatrix(r)
    W2 = W @ W
    if a < 1e-5:
        R = I + W + 0.5 * W2
    else:
        R = I + W * np.sin(a) / a + W2 * (1 - np.cos(a)) / a2
    assert np.allclose(R, R2)
    return R2

def getCrossProductMatrix(r):
    """
    :param r: [3, 1] vector
    :return: cross product matrix
    """
    hat = np.array([[0., -r[2, 0], r[1, 0]],
                     [r[2, 0], 0., -r[0, 0]],
                     [-r[1, 0], r[0, 0], 0.]])

    return hat


def scaleMatrix(scale):
    return np.array([[scale[0], 0, 0, 0],
                     [0, scale[1], 0, 0],
                     [0, 0, scale[2], 0],
                     [0, 0, 0, 1]
                     ], dtype=np.float)

def translationMatrix(tvec):
    return np.array([[1, 0, 0, tvec[0]],
                     [0, 1, 0, tvec[1]],
                     [0, 0, 1, tvec[2]],
                     [0, 0, 0, 1]
                     ], dtype=np.float)

def rotationMatrix(angle, axis):
    """

    :param angle: scalar, rotating degree
    :param axis: [3, ] axis to rotate with (will be normalized internally)
    :return: [4, 4] rotation matrix
    """
    angle = (angle / 180) * np.pi
    s = np.sin(angle)
    c = np.cos(angle)
    mc = 1 - c

    len = np.linalg.norm(axis)
    if (len == 0):
        return np.eye(4)

    axis = axis/len
    x = axis[0]
    y = axis[1]
    z = axis[2]

    return np.array([[ x * x * mc + c, x * y * mc - z * s, x * z * mc + y * s, 0],
                     [ x * y * mc + z * s, y * y * mc + c, y * z * mc - x * s, 0],
                     [ x * z * mc - y * s, y * z * mc + x * s, z * z * mc + c, 0],
                     [ 0, 0, 0, 1]], dtype=np.float)

def lookAtMatrix(eye=np.array([0,0,1]), at=np.array([0,0,0]), up=np.array([0,1,0])):

    up = up / np.linalg.norm(up)
    f = (at - eye)
    f = f / np.linalg.norm(f)

    r = np.cross(f, up)
    r = r / np.linalg.norm(r)

    u = np.cross(r, f)
    u = u / np.linalg.norm(u)

    return np.array([[r[0], r[1], r[2], -r @ eye],
                     [u[0], u[1], u[2], -u @ eye],
                     [-f[0], -f[1], -f[2], f @ eye],
                     [0, 0, 0, 1]], dtype=np.float)

def inverseMatrix(mat):
    """

    :param mat: [4, 4] rigid transformation
    :return:
    """
    R = mat[:3, :3]
    t = mat[:3, 3]
    R_inv = R.T
    t_inv = - (R.T @ t)
    mat_inv = np.eye(4)
    mat_inv[:3, :3] = R_inv
    mat_inv[:3, 3] = t_inv

    return mat_inv



def perspectiveMatrix(K, width, height, zNear, zFar, flipY):

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    w = width
    h = height
    n = zNear
    f = zFar

    if flipY:
        return np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0],
                         [0, -2 * fy / h, 1 - 2 * cy / h, 0],
                         [0, 0, (f + n) / (n - f), (2 * f * n) / (n - f)],
                         [0, 0, -1, 0]], dtype=np.float)

    return np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0],
                     [0, 2 * fy / h, 2 * cy / h - 1, 0],
                     [0, 0, (f + n) / (n - f), (2 * f * n) / (n - f)],
                     [0, 0, -1, 0]], dtype=np.float)

def perspectiveMatrixAspect(fovy, aspect, zNear, zFar):

    fovy = np.radians(fovy)
    focal = 1.0 / np.tan(fovy / 2.0)
    n = zNear
    f = zFar

    return np.array([[focal/aspect, 0, 0, 0],
                     [0, focal, 0, 0],
                     [0, 0, (f + n) / (n - f), (2 * f * n) / (n - f)],
                     [0, 0, -1, 0]], dtype=np.float)

def perspectiveMatrixOpenGL(left, right, top, bottom, near, far, flipY):
    if flipY:
        return np.array([[2*near/(right-left), 0, (right+left)/(right-left), 0],
                      [0, -2*near/(top-bottom), -(top+bottom)/(top-bottom), 0],
                      [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                      [0, 0, -1, 0]])

    return np.array([[2*near/(right-left), 0, (right+left)/(right-left), 0],
                  [0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0],
                  [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                  [0, 0, -1, 0]])

def negYZMat3(mat):
    """
    Negate the y, z (row 1, 2) value of a [3,3] matrix
    :param mat: [3, 3] matrix
    :return: same matrix but with row 1, 2 negated
    """

    return np.array([[mat[0, 0], mat[0, 1], mat[0, 2]],
                     [-mat[1, 0], -mat[1, 1], -mat[1, 2]],
                     [-mat[2, 0], -mat[2, 1], -mat[2, 2]]], dtype=np.float)

def negYZMat4(mat):
    """
    Negate the y, z (row 1, 2) value of a [4,4] matrix
    :param mat: [4, 4] matrix
    :return: same matrix but with row 1, 2 negated
    """
    return np.array([[mat[0, 0], mat[0, 1], mat[0, 2], mat[0, 3]],
                     [-mat[1, 0], -mat[1, 1], -mat[1, 2], -mat[1, 3]],
                     [-mat[2, 0], -mat[2, 1], -mat[2, 2], -mat[2, 3]],
                     [mat[3, 0], mat[3, 1], mat[3, 2], mat[3, 3]]], dtype=np.float)


# Tested with test_backproject_project
def project(vertices, K):
    """
    This function performs the opencv intrinsic matrix for perspective projection onto image plane
    (i.e. if the vertices are in the opengl coordinate, then user should manually negate y,z, coordinate)

    :param vertices: [N, 3]  vertices in camera coordinate (opencv)
    :param K: [3, 3] intrinsic matrix
    :return: uv_map: [N, 2] floating point (u,v) pixel location of the projected vertices
    """

    uv = K @ vertices.T
    uv[0, :] /= uv[2, :]
    uv[1, :] /= uv[2, :]
    uv = (uv.T)[:, :2]

    return uv.astype(np.float32)

# Tested with test_backproject_project
def projectMap(vertex_map, K):
    """
    This function performs the opencv intrinsic matrix for perspective projection onto image plane
    (i.e. if the vertex map is in the opengl coordinate, then user should manually negate y,z, coordinate)

    :param vertex_map: [H, W, 3] vertex map in camera coordinate (opencv)
    :param K: [3, 3] intrinsic matrix
    :return: uv_map: [H, W, 2] floating point (u,v) pixel location of the projected vertices
    """
    H, W, C = vertex_map.shape
    uv_map = project(vertex_map.reshape(H * W, 3), K).reshape(H, W, 2)

    return uv_map

# Tested with test_backproject_project
# TODO: Think if the depth of back project should be positive, otherwise the coordinates will be wrong
def backProjectMap(depth, K, opengl, mask=None):
    """
    Project depth into opengl camera coordinate (camera looking @ -Z axis)
    :param depth: [H, W] depth map in opengl coordinate
    :param K: [3, 3] intrinsic matrix
    :param mask: [H, W] if given, then the output map is masked
    :param opengl: boolean flag, if set, then it will use the opengl camera coordinate system. (i.e. +Y up, -Z forward)
    :return: cloud: [H, W, 3] point cloud in camera coordinate
    """
    H, W = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    xx, yy = np.meshgrid(range(W), range(H))
    pt0 = (xx - cx) * depth / fx
    pt1 = (yy - cy) * depth / fy

    if opengl:
        # negate the x coordinate, since negative depth flips the x coords.
        pt0 = - pt0

    if mask is not None:
        cloud = np.dstack((pt0 * mask, pt1 * mask, depth * mask))
    else:
        cloud = np.dstack((pt0, pt1, depth))

    return cloud


def calGradient2D(x):
    """
    Calculate the gradient of a 2D input data
    :param x: [H, W] or [H, W, C] input data
    :return: gy: [H, W] gradient respect to y axis
             gx: [H, W] gradient respect to x axis
    """
    h, w = x.shape[:2]
    gy = np.zeros_like(x)
    gx = np.zeros_like(x)

    gy[1:h - 1, :] = (x[2:h, :] - x[0:h - 2, :]) / 2
    gy[0, :] = x[1, :] - x[0, :]
    gy[h - 1, :] = x[h - 1, :] - x[h - 2, :]

    gx[:, 1: w - 1] = (x[:, 2:w] - x[:, 0:w - 2]) / 2
    gx[:, 0] = x[:, 1] - x[:, 0]
    gx[:, w - 1] = x[:, w - 1] - x[:, w - 2]

    return gy, gx

def calNormalMap(vertex_map, opengl=True):
    """
    Compute the normal map using vertex map, i.e. calculate normal using neighborhoods not actual geometry
    :param vertex_map: [H, W, 3] vertex map
    :return: normal_map: [H, W, 3] normal map, the normals is calculated s.t. it is pointing outward
    """
    gy_x, gx_x = np.gradient(vertex_map[..., 0])
    gy_y, gx_y = np.gradient(vertex_map[..., 1])
    gy_z, gx_z = np.gradient(vertex_map[..., 2])

    gx = np.dstack((gx_x, gx_y, gx_z))
    gy = np.dstack((gy_x, gy_y, gy_z))

    # y cross x,  since the image y is downward
    if opengl:
        normal_map = np.cross(gx, gy)
    else:
        normal_map = np.cross(gy, gx)

    n = np.linalg.norm(normal_map, axis=2)
    normal_map[:, :, 0] /= n
    normal_map[:, :, 1] /= n
    normal_map[:, :, 2] /= n

    NaNs = np.isnan(normal_map)
    normal_map[NaNs] = 0

    # # np.gradient is equivalent to below
    # gy_x2, gx_x2 = calGradient2D(vertex_map[..., 0])
    # gy_y2, gx_y2 = calGradient2D(vertex_map[..., 1])
    # gy_z2, gx_z2 = calGradient2D(vertex_map[..., 2])
    # gx2 = np.dstack((gx_x2, gx_y2, gx_z2))
    # gy2 = np.dstack((gy_x2, gy_y2, gy_z2))
    # assert np.allclose(gx,gx2)
    # assert np.allclose(gy, gy2)

    return normal_map


def warpRigidPoints(vertex, R, t):
    """

    :param vertex: [N, 3]
    :param R: [3, 3]
    :param t: [3,]
    :return:
    """
    x = R @ vertex.T # [3, N]
    x = x + t[...,np.newaxis] # [3, N]

    return x.T # [N, 3]

def cal_gradient2D(x):
    """

    :param x: [H, W, C]
    :return: gy: [H, W, C]
             gx: [H, W, C]
    """
    h, w, c = x.shape
    gy = np.zeros_like(x)
    gx = np.zeros_like(x)

    gy[1:h - 1, :, :] = (x[2:h, :, :] - x[0:h - 2, :, :]) / 2
    gy[0, :, :] = x[1, :, :] - x[0, :, :]
    gy[h - 1, :, :] = x[h - 1, :, :] - x[h - 2, :, :]

    gx[:, 1: w - 1, :] = (x[:, 2:w, :] - x[:, 0:w - 2, :]) / 2
    gx[:, 0, :] = x[:, 1, :] - x[:, 0, :]
    gx[:, w - 1, :] = x[:, w - 1, :] - x[:, w - 2, :]

    return gy, gx



def warpInverseRigidPoints(vertex, R, t):
    """

       :param vertex: [N, 3]
       :param R: [3, 3]
       :param t: [3,]
       :return:
       """
    x = R.T @ vertex.T  # [3, N]
    x = x - (R.T @ t)[...,np.newaxis]  # [3, N]

    return x.T  # [N, 3]


def toHomoMap(vertex):
    """
    Convert [H, W, 3] vertex map to [H, W, 4] vertex map in homogeneous coordinate form
    :param vertex: [H, W, 3] vertex map
    :return: [H, W, 4] vertex map in homogeneous coordinate form
    """
    return np.concatenate((vertex, np.ones((vertex.shape[0], vertex.shape[1], 1))), axis=2)


def toHomo(vectors):
    """
    Convert  N x 3 vectors to  4 x N vectors in homogeneous coordinate form
    :param vectors:  N x 3 vectors
    :return:  4 x N vectors
    """

    return np.concatenate((vectors, np.ones((vectors.shape[0], 1))), axis=1).T




def checkPointHomo(vectors):
    """
    Check if the input vector is N x 3 or  N x 4 homogeneous coordinate
    :param vectors:  N x 3 vectors
    :return:  N x 4 vectors
    """
    N, dim = vectors.shape
    if dim == 3:
        vec = np.ones((N,4),dtype=vectors.dtype)
        vec[:,:3] = vectors
        vec[:,3] = 1
    else:
        vec = vectors

    return vec

def skew(x):
    return np.array([[0, -x[2], x[1]],
                 [x[2], 0, -x[0]],
                 [-x[1], x[0], 0]])

def batch_skew(x):
    """
    computes the batched skew symmetric matrix
    :param x: shape (N,3,1) numpy array
    :return: shape (N,3,3) numpy array, skew symmetric matrix
    """
    O = np.zeros((x.shape[0],1))

    row1 = np.concatenate((O, -x[:,2], x[:,1]),axis=1)
    row2 = np.concatenate((x[:,2], O, -x[:,0]), axis=1)
    row3 = np.concatenate((-x[:,1], x[:,0], O), axis=1)
    out = np.stack((row1, row2, row3), axis=1)

    return out

def kabsch(_3d1, _3d2):
    """
        Calulate the rigid transformation  _3d2 =  T x  _3d1

        Reference: https://en.wikipedia.org/wiki/Kabsch_algorithm
    :param _3d1: N x 3 point cloud (source)
    :param _3d2: N x 3 point cloud  (destination)
    :return:  Rotation matrix and translation parameters from 1 ->  2  (Rx1 + t = x2)
    """
    c1 = np.average(_3d1,axis=0)
    c2 = np.average(_3d2,axis=0)
    y1 = _3d1 - c1
    y2 = _3d2 - c2
    H = np.array(np.matmul(np.mat(y1).T,np.mat(y2)))
    U, S, VT = np.linalg.svd(H)
    V = VT.T
    d = np.linalg.det(V.dot(U.T))
    I = np.diag([1,1,d])
    R = V.dot(I.dot(U.T))
    t = c2 - R.dot(c1)

    return R, t



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

class Region():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.h = h
        self.w = w





'''
def Rodrigues(w, u):
    
    Given axis vector w and translation vector u, generate a SE3 matrix
    :param w:  generator coefficients of Lie algebra [wx, wy, wz]
    :param u: translation vector [tx, ty, tz]
    :return: SE3 matrix (Lie group)

    if not np.any(u):
        u = np.zeros((3, 1))
    Wx = np.array([[0, -w[2], w[1]],
                   [w[2], 0, - w[0]],
                   [-w[1], w[0], 0]])
    ceta = math.sqrt(np.dot(w, np.transpose(w)))
    ceta_sq = ceta * ceta
    A = math.sin(ceta) / ceta
    B = (1 - math.cos(ceta)) / (ceta_sq)
    C = (1 - A) / ceta_sq

    R = np.eye(3) + A * Wx + B * np.dot(Wx, Wx)
    V = np.eye(3) + B * Wx + C * np.dot(Wx, Wx)

    # to SE3 matrix (Lie group)
    mat = np.block([[R, np.dot(V, np.reshape(u,(3,1)))],
                    [np.zeros(3), 1]])

    return mat
'''




#TODO: merge to Pose class
def rotatePose(rotation, pose1, radians):
    Ext1 = toExtMat(pose1, None, PoseParameterModel='Eulerzyx', radians=radians)
    Ext2 = toExtMat(rotation, None, PoseParameterModel='Eulerzyx', radians=radians)

    return np.dot(Ext2, Ext1)


#TODO: merge to Pose class
def setTranslation(Ext, translation):
    Ext[0:3, 3] = translation
    return Ext


def test_backproject_project(depth, K, mask):
    """

    :param depth: [H, W] depth map in opengl coordinate
    :param K: [3, 3] intrinsic matrix
    :return:
    """
    cam_coord = backProjectMap(depth, K, mask)
    uv = projectMap(cam_coord, K)
    depth_resampled = cv2.remap(depth, uv, None, interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(depth_resampled, depth)

def main():

    H = 480
    W = 640
    mask = np.ones((H,W)).astype(np.bool)
    depth = np.random.rand(H,W)
    K = np.array([[W//4, 0., W//2],
                  [0., H//4, H//2],
                  [0., 0., 1.]])

    test_backproject_project(depth, K, mask)


# import sys
# SOPHUS_ROOT = r'D:\MultimediaIClab\AR\Rendering\Sophus\py'
# sys.path.append(SOPHUS_ROOT)
#
# import sophus
# from sophus.se3 import Se3
#
#
# # [uvec, wvec]
# v = sophus.Vector6(0., 1, 0.5, 2., 1, 0.5)
# w = Se3.exp(v)
# mat = np.array(w.so3.matrix()._mat).reshape(3,3).astype(np.float)
# tvec = np.array(w.t._mat).astype(np.float)
#
# SO3 = cv2.Rodrigues(np.array([2., 1, 0.5]))[0]
# assert np.allclose(SO3,mat)
# #a = Pose(rvec= np.array([2., 1, 0.5]), tvec= np.array([0., 1, 0.5]))
# a = Pose()
# a.se3 = np.array([ 2., 1, 0.5, 0., 1, 0.5,])
# a.update()
#
#
#
# a.se3 += (-np.array([ 2., 1, 0.5, 0., 1, 0.5,]))
# a.update()
# a.se3 += np.array([2., 1, 0.5, 0., 1, 0.5, ])
# a.update()
#
# my_SO3 = a.SE3[:3,:3]
# my_tvec = a.SE3[:3,3]
# assert np.allclose(my_SO3 , SO3)
# assert np.allclose(my_tvec , tvec)
# # if __debug__:
# #     import sys
# #
# #     SOPHUS_ROOT = r'D:\MultimediaIClab\AR\Rendering\Sophus\py'
# #     sys.path.append(SOPHUS_ROOT)
# #
# #     import sophus
# #     from sophus.se3 import Se3
# #     from sophus.so3 import So3
# class Pose():
#     Tglc4 = np.array([[1, 0, 0, 0],
#                      [0, -1, 0, 0],
#                      [0, 0, -1, 0],
#                      [0, 0, 0, 1]])
#     Tglc3 = np.array([[1, 0, 0],
#                      [0, -1, 0],
#                      [0, 0, -1]])[np.newaxis, ...]

#     def __init__(self,rvec=np.zeros(3), tvec=np.zeros(3), PoseParamModel='axis', isRadian=True, SE3=None, isOpenGL=False):
#         """
#         All coordinates of the parameters are in OpenCV coordinate system, except for SE3_gl which is in OpenGL coordinate system
#         :param rvec:  axis angle (rotational parameter)
#         :param tvec:  translational parameter in meter
#         :param PoseParamModel:  'axis angle',  'euler angle'  and 'quaternion'
#         :param isRadian: whether the given rotational parameter is in radians
#         """
#         self.PoseParamModel = PoseParamModel
#         self.isRadian = isRadian
#         self.rvec = rvec
#         self.tvec = tvec

#         if SE3 is not None:
#             self.setSE3(SE3, isOpenGL=isOpenGL)
#         else:
#             self.SE3 = self.toSE3()

#         self.se3 = self.SE3toParam()
#         self.SE3_gl = self.convert_yz_mat(self.SE3)

#     def get_SE3_inv(self):
#         R_t = self.SO3.T
#         t_inv = -R_t.dot(self.tvec)
#         _mat = np.ones((4,4))
#         _mat[:3,:3] = R_t
#         _mat[:3,3] = t_inv

#         return _mat

#     @staticmethod
#     def convert_yz_mat(mat):
#         """
#         Transform the coordinates of opengl and opencv (add negation to both y, z axis)

#         Parameters
#         ----------
#         mat : array_like with shape (3x3) or (4x4)

#         Returns
#         -------
#          array_like with shape (3x3) or (4x4)
#             Same matrix but coordinate transformed
#         """

#         _mat = mat.copy()
#         _mat[1, :] = -_mat[1, :]
#         _mat[2, :] = -_mat[2, :]

#         return _mat

#     def setSE3(self, SE3, isOpenGL=False):
#         """
#         Set the internal parameters with an external SE3 matrix.
#         It will set the the matrix into self.SE3_gl and self.SE3 correctly according to the given isOpenGL Flag
#         :param SE3: 4x4 SE(3) rigid transformation matrix
#         :param isOpenGL: this is set to true if the SE3 input argument is in the OpenGL coordinate system
#         :return: None
#         """
#         if isOpenGL:
#             self.SE3_gl = SE3
#             self.SE3 = self.convert_yz_mat(SE3)
#         else:
#             self.SE3 = SE3
#             self.SE3_gl = self.convert_yz_mat(SE3)

#         self.SO3 = self.SE3[:3,:3]
#         self.se3 = self.SE3toParam()



#     def update(self,rvec=None, tvec=None, PoseParamModel=None):
#         if PoseParamModel != None:
#             self.PoseParamModel = PoseParamModel
#         if np.any(rvec != None):
#             self.rvec = rvec
#         if np.any(tvec != None):
#             self.tvec = tvec

#         if  np.all(rvec == None) and  np.all(tvec == None):
#             self.rvec = self.se3[:3]
#             self.so3 = self.se3[:3]
#             self.SE3 = self.__axixToSE3(hasUvec=True)
#             self.SE3_gl = self.convert_yz_mat(self.SE3)
#         else:
#             self.SE3 = self.toSE3()
#             self.se3 = self.__SE3Tose3()
#             temp = self.__axixToSE3(hasUvec=True)
#             assert np.allclose(temp, self.SE3)
#             self.SE3 = temp
#             self.SE3_gl = self.convert_yz_mat(self.SE3)


#     def toSE3(self, hasUvec=False):
#         # Euler Angle
#         if self.PoseParamModel.find('euler') >= 0:
#            return self.__eulerToSE3()
#         # Axis Angle
#         elif self.PoseParamModel == 'axis':
#             return self.__axixToSE3(hasUvec)
#         # Quaternion
#         elif self.PoseParamModel == 'quaternion':
#             raise NotImplementedError

#     '''
#     def toSO3(self):
#         if self.PoseParamModel.find('euler') >= 0:
#            return self.__eulerToSO3()
#         # Axis Angle
#         elif self.PoseParamModel == 'axis':
#             return self.__axixToSO3()
#         # Quaternion
#         elif self.PoseParamModel == 'quaternion':
#             raise NotImplementedError
    
#     def __eulerToSO3(self):
#         # Check if the angle is converted to radians
#         rvec = self.rvec if self.isRadian else np.radians(self.rvec)
#         sx, sy, sz = np.sin(rvec)
#         cx, cy, cz = np.cos(rvec)

#         Rx = np.array([[1, 0, 0, 0],
#                        [0, cx, -sx, 0],
#                        [0, sx, cx, 0],
#                        [0, 0, 0, 1]])
#         Ry = np.array([[cy, 0, sy, 0],
#                        [0, 1, 0, 0],
#                        [-sy, 0, cy, 0],
#                        [0, 0, 0, 1]])
#         Rz = np.array([[cz, -sz, 0, 0],
#                        [sz, cz, 0, 0],
#                        [0, 0, 1, 0],
#                        [0, 0, 0, 1]])
#         # Choose one order of rotation (Euler_3rd_2nd_1st)

#         if poseParm == 'eulerxyz':
#             mat = np.dot(np.dot(Rz, Ry), Rx)
#         elif poseParm == 'eulerxzy':
#             mat = np.dot(np.dot(Ry, Rz), Rx)
#         elif poseParm == 'euleryxz':
#             mat = np.dot(np.dot(Rz, Rx), Ry)
#         elif poseParm == 'euleryzx':
#             mat = np.dot(np.dot(Rx, Rz), Ry)
#         elif poseParm == 'eulerzxy':
#             mat = np.dot(np.dot(Ry, Rx), Rz)
#         elif poseParm == 'eulerzyx':
#             mat = np.dot(np.dot(Rx, Ry), Rz)
#         self.SO3 = mat
#         return mat
#     '''
#     def __eulerToSE3(self):
#         mat = np.eye(4)
#         mat[:3,:3] = self.SO3
#         # set translation
#         mat[0:3,3] = self.tvec

#         return mat

#     '''
#     def __axixToSO3(self):
#         self.SO3 = self.SE3[:3,:3]
#         return self.SO3
#     '''
#     def __axixToSE3(self, hasUvec=False):
#         """
#                 Calculate SE(3) from a given axis angle vector se(3) = (rvec, uvec)^T
#         :return:  SE(3) matrix
#         """
#         # Check if the angle is converted to radians
#         rvec = self.rvec if self.isRadian else np.radians(self.rvec)

#         #1. OpenCV method
#         SO3 = cv2.Rodrigues(np.array(rvec))[0]

#         #2.  R = cos(theta) * I + (1 - cos(theta)) * r * rT + sin(theta) * [r_x]
#         theta = np.linalg.norm(rvec)
#         #theta = np.arccos((np.trace(SO3) - 1) / 2)   Don't know why it is different from  np.linalg.norm(self.rvec)

#         itheta = 1. / theta if theta else 0.
#         r = rvec * itheta
#         if theta: assert np.allclose(np.linalg.norm(r), 1.)
#         r_x = skew(r)
#         c = np.cos(theta)
#         s = np.sin(theta)
#         rrt = np.array([[r[0] * r[0], r[1] * r[0], r[2] * r[0]],
#                         [r[0] * r[1], r[1] * r[1], r[2] * r[1]],
              
# # if __debug__:
# #     import sys
# #
# #     SOPHUS_ROOT = r'D:\MultimediaIClab\AR\Rendering\Sophus\py'
# #     sys.path.append(SOPHUS_ROOT)
# #
# #     import sophus
# #     from sophus.se3 import Se3
# #     from sophus.so3 import So3

# class Pose():

#     Tglc4 = np.array([[1, 0, 0, 0],
#                      [0, -1, 0, 0],
#                      [0, 0, -1, 0],
#                      [0, 0, 0, 1]])
#     Tglc3 = np.array([[1, 0, 0],
#                      [0, -1, 0],
#                      [0, 0, -1]])[np.newaxis, ...]

#     def __init__(self,rvec=np.zeros(3), tvec=np.zeros(3), PoseParamModel='axis', isRadian=True, SE3=None, isOpenGL=False):
#         """
#         All coordinates of the parameters are in OpenCV coordinate system, except for SE3_gl which is in OpenGL coordinate system
#         :param rvec:  axis angle (rotational parameter)
#         :param tvec:  translational parameter in meter
#         :param PoseParamModel:  'axis angle',  'euler angle'  and 'quaternion'
#         :param isRadian: whether the given rotational parameter is in radians
#         """
#         self.PoseParamModel = PoseParamModel
#         self.isRadian = isRadian
#         self.rvec = rvec
#         self.tvec = tvec

#         if SE3 is not None:
#             self.setSE3(SE3, isOpenGL=isOpenGL)
#         else:
#             self.SE3 = self.toSE3()

#         self.se3 = self.SE3toParam()
#         self.SE3_gl = self.convert_yz_mat(self.SE3)

#     def get_SE3_inv(self):
#         R_t = self.SO3.T
#         t_inv = -R_t.dot(self.tvec)
#         _mat = np.ones((4,4))
#         _mat[:3,:3] = R_t
#         _mat[:3,3] = t_inv

#         return _mat

#     @staticmethod
#     def convert_yz_mat(mat):
#         """
#         Transform the coordinates of opengl and opencv (add negation to both y, z axis)

#         Parameters
#         ----------
#         mat : array_like with shape (3x3) or (4x4)

#         Returns
#         -------
#          array_like with shape (3x3) or (4x4)
#             Same matrix but coordinate transformed
#         """

#         _mat = mat.copy()
#         _mat[1, :] = -_mat[1, :]
#         _mat[2, :] = -_mat[2, :]

#         return _mat

#     def setSE3(self, SE3, isOpenGL=False):
#         """
#         Set the internal parameters with an external SE3 matrix.
#         It will set the the matrix into self.SE3_gl and self.SE3 correctly according to the given isOpenGL Flag
#         :param SE3: 4x4 SE(3) rigid transformation matrix
#         :param isOpenGL: this is set to true if the SE3 input argument is in the OpenGL coordinate system
#         :return: None
#         """
#         if isOpenGL:
#             self.SE3_gl = SE3
#             self.SE3 = self.convert_yz_mat(SE3)
#         else:
#             self.SE3 = SE3
#             self.SE3_gl = self.convert_yz_mat(SE3)

#         self.SO3 = self.SE3[:3,:3]
#         self.se3 = self.SE3toParam()



#     def update(self,rvec=None, tvec=None, PoseParamModel=None):
#         if PoseParamModel != None:
#             self.PoseParamModel = PoseParamModel
#         if np.any(rvec != None):
#             self.rvec = rvec
#         if np.any(tvec != None):
#             self.tvec = tvec

#         if  np.all(rvec == None) and  np.all(tvec == None):
#             self.rvec = self.se3[:3]
#             self.so3 = self.se3[:3]
#             self.SE3 = self.__axixToSE3(hasUvec=True)
#             self.SE3_gl = self.convert_yz_mat(self.SE3)
#         else:
#             self.SE3 = self.toSE3()
#             self.se3 = self.__SE3Tose3()
#             temp = self.__axixToSE3(hasUvec=True)
#             assert np.allclose(temp, self.SE3)
#             self.SE3 = temp
#             self.SE3_gl = self.convert_yz_mat(self.SE3)


#     def toSE3(self, hasUvec=False):
#         # Euler Angle
#         if self.PoseParamModel.find('euler') >= 0:
#            return self.__eulerToSE3()
#         # Axis Angle
#         elif self.PoseParamModel == 'axis':
#             return self.__axixToSE3(hasUvec)
#         # Quaternion
#         elif self.PoseParamModel == 'quaternion':
#             raise NotImplementedError

#     '''
#     def toSO3(self):
#         if self.PoseParamModel.find('euler') >= 0:
#            return self.__eulerToSO3()
#         # Axis Angle
#         elif self.PoseParamModel == 'axis':
#             return self.__axixToSO3()
#         # Quaternion
#         elif self.PoseParamModel == 'quaternion':
#             raise NotImplementedError
    
#     def __eulerToSO3(self):
#         # Check if the angle is converted to radians
#         rvec = self.rvec if self.isRadian else np.radians(self.rvec)
#         sx, sy, sz = np.sin(rvec)
#         cx, cy, cz = np.cos(rvec)

#         Rx = np.array([[1, 0, 0, 0],
#                        [0, cx, -sx, 0],
#                        [0, sx, cx, 0],
#                        [0, 0, 0, 1]])
#         Ry = np.array([[cy, 0, sy, 0],
#                        [0, 1, 0, 0],
#                        [-sy, 0, cy, 0],
#                        [0, 0, 0, 1]])
#         Rz = np.array([[cz, -sz, 0, 0],
#                        [sz, cz, 0, 0],
#                        [0, 0, 1, 0],
#                        [0, 0, 0, 1]])
#         # Choose one order of rotation (Euler_3rd_2nd_1st)

#         if poseParm == 'eulerxyz':
#             mat = np.dot(np.dot(Rz, Ry), Rx)
#         elif poseParm == 'eulerxzy':
#             mat = np.dot(np.dot(Ry, Rz), Rx)
#         elif poseParm == 'euleryxz':
#             mat = np.dot(np.dot(Rz, Rx), Ry)
#         elif poseParm == 'euleryzx':
#             mat = np.dot(np.dot(Rx, Rz), Ry)
#         elif poseParm == 'eulerzxy':
#             mat = np.dot(np.dot(Ry, Rx), Rz)
#         elif poseParm == 'eulerzyx':
#             mat = np.dot(np.dot(Rx, Ry), Rz)
#         self.SO3 = mat
#         return mat
#     '''
#     def __eulerToSE3(self):
#         mat = np.eye(4)
#         mat[:3,:3] = self.SO3
#         # set translation
#         mat[0:3,3] = self.tvec

#         return mat

#     '''
#     def __axixToSO3(self):
#         self.SO3 = self.SE3[:3,:3]
#         return self.SO3
#     '''
#     def __axixToSE3(self, hasUvec=False):
#         """
#                 Calculate SE(3) from a given axis angle vector se(3) = (rvec, uvec)^T
#         :return:  SE(3) matrix
#         """
#         # Check if the angle is converted to radians
#         rvec = self.rvec if self.isRadian else np.radians(self.rvec)

#         #1. OpenCV method
#         SO3 = cv2.Rodrigues(np.array(rvec))[0]

#         #2.  R = cos(theta) * I + (1 - cos(theta)) * r * rT + sin(theta) * [r_x]
#         theta = np.linalg.norm(rvec)
#         #theta = np.arccos((np.trace(SO3) - 1) / 2)   Don't know why it is different from  np.linalg.norm(self.rvec)

#         itheta = 1. / theta if theta else 0.
#         r = rvec * itheta
#         if theta: assert np.allclose(np.linalg.norm(r), 1.)
#         r_x = skew(r)
#         c = np.cos(theta)
#         s = np          [r[0] * r[2], r[1] * r[2], r[2] * r[2]]])
#         self.SO3 = c * np.eye(3) + (1 - c) * rrt + s * r_x
#         self.SO3_gl = self.convert_yz_mat(self.SO3)

#         assert np.allclose(SO3, self.SO3)

#         if hasUvec:
#             sinc = np.sin(theta)/theta if theta else 1.
#             cosc = (1-np.cos(theta))/theta if theta else 0.
#             J = sinc* np.eye(3) + (1-sinc)*rrt + cosc*r_x
#             self.tvec = J.dot(self.se3[3:6])

#             if __debug__:
#                 if np.any(self.se3 != np.zeros(6)):
#                     v = sophus.Vector6(self.se3[3], self.se3[4], self.se3[5], self.se3[0], self.se3[1], self.se3[2])
#                     w = Se3.exp(v)
#                     mat = np.array(w.so3.matrix()._mat).reshape(3, 3).astype(np.float)
#                     tvec = np.array(w.t._mat).astype(np.float)

#                     assert np.allclose(self.SO3, mat)
#                     assert np.allclose(self.tvec, tvec)

#         mat = np.eye(4)
#         mat[:3, :3] = self.SO3
#         mat[:3, 3] = self.tvec

#         return mat
#     '''
#     def SO3toParam(self):
#         if self.PoseParamModel.find('euler') >= 0:
#             return self.__SO3ToEuler()
#             # Axis Angle
#         elif self.PoseParamModel == 'axis':
#             return self.__SO3Toso3()
#             # Quaternion
#         elif self.PoseParamModel == 'quaternion':
#             raise NotImplementedError
#         '''
#     def SE3toParam(self):
#         if self.PoseParamModel.find('euler') >= 0:
#             return self.__SE3ToEuler()
#             # Axis Angle
#         elif self.PoseParamModel == 'axis':
#             return self.__SE3Tose3()
#             # Quaternion
#         elif self.PoseParamModel == 'quaternion':
#             raise NotImplementedError
#     '''
#     def __SO3ToEuler(self):
#         if self.PoseParamModel  == 'eulerzyx':
#             r02 = self.SO3[0][2]
#             if r02 < 1:
#                 if r02 > -1:
#                     thetaY = math.asin(r02)
#                     thetaX = math.atan2(-self.SO3[1][2], self.SO3[2][2])
#                     thetaZ = math.atan2(-self.SO3[0][1], self.SO3[0][0])
#                 else:  # r02 = -1
#                     thetaY = -90
#                     thetaX = -math.atan2(self.SO3[1][0], self.SO3[1][1])
#                     thetaZ = 0
#             else:  # r02 = 1
#                 thetaY = 90
#                 thetaX = math.atan2(self.SO3[1][0], self.SO3[1][1])
#                 thetaZ = 0
#         else:
#             print('Decomposing euler angle to format other than zyx is currently not implemented!')
#             raise NotImplementedError

#         return np.array([thetaX, thetaY, thetaZ])
   
#     def __SE3ToEuler(self):
#         rvec = self.__SO3ToEuler()
#         tvec = self.SE3[:,3]
#         return np.hstack((rvec, tvec))
#      '''
#     '''
#     def __SO3Toso3(self):
#         ceta = arccos(np.trace(self.SO3) - 1) / 2
#         lnR = 0.5*ceta/np.sin(ceta)*(self.SO3-self.SO3.T)
#         self.so3 = np.array([lnR[1,2], lnR[2,0], lnR[0,1]])
#         return self.so3
#     '''

#     def __SE3Tose3(self):
#         """
#             Calculate se(3) = [ rvec, uvec] given SE(3)
#         :return:
#         """
#         self.rvec = cv2.Rodrigues(self.SO3)[0].squeeze()
#         self.tvec = self.SE3[:3, 3]


#         traceR = np.trace(self.SO3)
#         theta = np.arccos((traceR - 1) / 2)
#         # # TODO: 1. from scratch
#         # # https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio
#         # epsilon = 1e-5
#         # r = np.array([self.SO3[2,1]-self.SO3[1,2],self.SO3[0,2]-self.SO3[2,0],self.SO3[1,0]-self.SO3[0,1]])

#         # ub = 3-epsilon
#         # lb = -1+epsilon
#         # if traceR >= ub:
#         #     self.so3 = (1/2 - (traceR-3)/12)*r
#         # elif traceR <ub and traceR>lb:
#         #     self.so3 = theta*r / (2 * np.sin(theta))
#         #
#         # elif  traceR<=lb: #seems buggy here when theta == pi
#         #     a = np.diag(self.SO3).argmax()
#         #     b = (a + 1) % 3
#         #     c = (a + 2) % 3
#         #     s = np.sqrt(self.SO3[a, a] - self.SO3[b, b] - self.SO3[c, c] + 1)
#         #     v = np.array(
#         #         [s / 2, 0.5 * (self.SO3[b, a] + self.SO3[a, b]) / s, 0.5 * (self.SO3[c, a] + self.SO3[a, c]) / s])
#         #
#         #     self.so3 = np.pi*v/np.linalg.norm(v)
#         # else:
#         #     raise ValueError
#         #
#         # if __debug__:
#         #     rod = cv2.Rodrigues(self.SO3)[0].squeeze()
#         #     assert np.allclose(self.so3, rod)

#         # TODO: 2. still use cv2
#         self.so3 = self.rvec

#         itheta = 1. / theta if theta else 0.
#         r = self.so3 * itheta
#         if theta: assert np.allclose(np.linalg.norm(r), 1.)
#         r_x = skew(r)
#         rrt = np.array([[r[0] * r[0], r[1] * r[0], r[2] * r[0]],
#                         [r[0] * r[1], r[1] * r[1], r[2] * r[1]],
#                         [r[0] * r[2], r[1] * r[2], r[2] * r[2]]])

#         # J_inv =(theta/2)cot(theta/2) I + (1-(theta/2)cot(theta/2))r*rT - (theta/2)[r_x]
#         # since we can not eval x*cot(x) at x=pi directly, use x*cot(x)= x*cos(x)/sin(x) = cos(x)/sinc instead
#         thetaHalf = theta / 2
#         cotthetaH = np.cos(thetaHalf)/np.sinc(thetaHalf/np.pi)
#         J_inv = cotthetaH * np.eye(3) + (1 - cotthetaH) * rrt - thetaHalf * r_x

#         self.se3 = np.hstack((self.so3, np.dot(J_inv, self.tvec)))

#         #======== double check =============
#         if __debug__:
#             # This fails when theta = pi, due to the unstability of inverse sinc.
#             # Need to use special methods
#             # lnR = 0.5 * (self.SO3 - self.SO3.T) / np.sinc(theta / np.pi)
#             # so3 = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])
#             # assert np.allclose(so3, self.rvec)

#             if np.any(self.se3[:3] != np.zeros(3)):
#                 SSO3 = So3.exp(sophus.Vector3(self.rvec[0], self.rvec[1], self.rvec[2]))
#                 sSe3 =  Se3(SSO3,  sophus.Vector3(self.tvec[0], self.tvec[1], self.tvec[2]))
#                 sse3 = sSe3.log()._mat
#                 ssse3 = np.array([sse3[3],sse3[4],sse3[5], sse3[0], sse3[1], sse3[2]]).astype(float)
#                 # TODO: decide whether or not use Sophus log map output as gt
#                 if not np.allclose(ssse3[:3],self.se3[:3]):
#                     SSO3 = So3.exp(sophus.Vector3(-self.rvec[0], -self.rvec[1], -self.rvec[2]))
#                     sSe3 = Se3(SSO3, sophus.Vector3(self.tvec[0], self.tvec[1], self.tvec[2]))
#                     sse3 = sSe3.log()._mat
#                     ssse3 = np.array([sse3[3], sse3[4], sse3[5], sse3[0], sse3[1], sse3[2]]).astype(float)

#                 assert np.allclose(self.se3, ssse3)

#                 # TODO: add this back
#                 v = sophus.Vector6(self.se3[3], self.se3[4], self.se3[5],self.se3[0], self.se3[1], self.se3[2])
#                 w = Se3.exp(v)
#                 mat = np.array(w.so3.matrix()._mat).reshape(3, 3).astype(np.float)
#                 tvec = np.array(w.t._mat).astype(np.float)
#                 assert np.allclose(self.SO3, mat)
#                 assert np.allclose(self.tvec, tvec)


#         return self.se3


if __name__ == "__main__":
    main()