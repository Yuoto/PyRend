import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#for testing use only
RENDERER_ROOT = "/home/yuoto/AR/Renderings/PyRend"
sys.path.append(os.path.join(RENDERER_ROOT,'utiles'))
from transform import lookAtMatrix
import numpy as np


def nearest_point_sample(xyz, query_points, threshold):
    """
    Get the nearest points of the query_points from original xyz point cloud
    :param xyz: [B, N, 3] point cloud data
    :param query_points: [B, S, 3] queried point cloud
    :param threshold: threshold of the mask
    :return: sampled_query_idx: [B, S] the index of the sampled query points from xyz
             mask: [B, 1] indicates whether the query point is further than the threshold
    """
    # device = xyz.device
    B, S, _ = query_points.shape

    sqrdists = square_distance(query_points, xyz)
    sampled_query_sorted = sqrdists.topk(1, dim=-1, largest=False)

    sampled_query_idx = sampled_query_sorted[1][:, :, 0]
    sampled_distance = sampled_query_sorted[0][:, :, 0]
    mask = sampled_distance < threshold

    return sampled_query_idx, mask

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: [B, N, 3] pointcloud data
        npoint: number of samples
    Return:
        centroids: [B, npoint] sampled pointcloud index,
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) #[B,S]
    new_points = points[batch_indices, idx, :]
    return new_points

def batch_non_rigid_inverse_warp(deformed_pts, flow, rot_mat, trans):
    '''
      First apply inverse batched rigid transformation, then apply inverse flow
      :param deformed_pts: [B, 3, N] Tensor of 3D points
      :param flow: [B, 3, N] Tensor of 3D forward optical flow
      :param rot_mat: [B, 3, 3] Tensor of SO3
      :param trans: [B, 3] Tensor of translations R^3
      :return: deformed_pts: [B, 3, N] Tensor of xyz coordinates in the image coordinate
    '''
    trans = trans.unsqueeze(2)  # [B, 3, 1]
    rot_mat_inv = torch.transpose(rot_mat, 1, 2)
    trans_inv = torch.bmm(rot_mat_inv, -trans)

    pts = torch.baddbmm(trans_inv, rot_mat_inv, deformed_pts) - flow

    return pts

def batch_non_rigid_local_wrap(pts, rot_mat, trans):
    """
            W(x, R, t) = R(x-t) + x + t
    :param pts: [B, 3, N] Tensor of 3D points
    :param rot_mat: [B, 3, 3] Tensor of SO3
    :param trans: [B, 3, N] Tensor of translations R^3
    :return: deformed_pts: [B, 3, N] Tensor of xyz coordinates in the image coordinate
    """

    deformed_pts = pts + trans # [B, 3, N]
    local_pts = pts - trans # [B, 3, N]


    return torch.baddbmm(deformed_pts, rot_mat, local_pts, deformed_pts)


def batch_non_rigid_warp(pts, flow, rot_mat, trans):
    '''
      First apply flow, then apply batched rigid transformation of the given 3D points into the image coordinate
      :param pts: [B, 3, N] Tensor of 3D points
      :param flow: [B, 3, N] Tensor of 3D optical flow
      :param rot_mat: [B, 3, 3] Tensor of SO3
      :param trans: [B, 3] Tensor of translations R^3
      :return: deformed_pts: [B, 3, N] Tensor of xyz coordinates in the image coordinate
    '''

    trans = trans.unsqueeze(2)  # [B, 3, 1]
    deformed_pts = pts + flow

    return torch.baddbmm(trans, rot_mat, deformed_pts)


def batch_rigid_transform(pts, rot_mat, trans):
    '''
      Apply batched rigid transformation of the given 3D points into the image coordinate
      :param pts: [B, 3, N] Tensor of 3D points
      :param rot: [B, 3, 3] Tensor of SO3
      :param trans: [B, 3] Tensor of translations R^3
      :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image coordinate
    '''

    trans = trans.unsqueeze(2) # [B, 3, 1]

    return torch.baddbmm(trans, rot_mat, pts)


def batch_perspective(pts, K):
    '''
     Apply batched perspective transformation
     :param pts: [B, 3, N] Tensor of 3D points
     :param K: [3, 3] Tensor of intrinsic matrix
     :return: uv: [B, 2, N] Tensor of uv image coordinate (x,y)
   '''
    B = pts.size(0)
    K = K.view(1, 3, 3).repeat(B, 1, 1)  # [B, 3, 3]
    uv_homo = torch.bmm(K, pts)
    uv = uv_homo[:, :2, :]/uv_homo[:, 2:3, :]
    return uv

# Function tested
#TODO: decide if K should be batched K # [B, 3, 3]
def batch_back_project(depth_map, K):
    """ Back project the depth map from pixel coordinate into camera coordinate
    :param depth_map: [B, 1, H, W]
    :param K: [3, 3] instrinsic matrix
    :return xyz: [B, 3, H, W] camera coordinate
    """
    B, C, H, W = depth_map.shape

    x, y = generate_camera_coord_grid(B, H, W, K) # [B, 1, H, W], [B, 1, H, W]
    return torch.cat((x * depth_map, y * depth_map, depth_map), dim=1)

def index(feat, uv, padding_mode='zeros'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, S] uv coordinates in the image plane (x, y)
    :return: [B, C, S] image features at the uv coordinates
    '''

    # first transfrom the uv coordinates into range [-1,1]
    B, _, H, W = feat.shape
    _, _, S = uv.shape
    normalize = torch.cuda.FloatTensor([(W - 1) / 2, (H - 1) / 2]).view(1,2,1)
    uv = uv/normalize -1
    uv = uv.transpose(1, 2)  # [B, S, 2]
    uv = uv.view(B, S, 1, 2)  # [B, S, 1, 2]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True, padding_mode=padding_mode)  # [B, C, S, 1]
    return samples[:, :, :, 0]  # [B, C, S]


def cal_normal_map(vertex_map, opengl=True):
    """
    Compute the normal map using vertex map, i.e. calculate normal using neighborhoods not actual geometry
    :param vertex_map: [B, 3, H, W] Tensor of xyz vertex map
    :return: normal_map_normed: [B, 3, H, W] Tensor of normal map (outward)
    """


    gy_x, gx_x = torch.gradient(vertex_map[:,0:1,:,:], dim=[2,3])
    gy_y, gx_y = torch.gradient(vertex_map[:,1:2,:,:], dim=[2,3])
    gy_z, gx_z = torch.gradient(vertex_map[:,2:3,:,:], dim=[2,3])

    # before torch 1.9.0, there is no torch.gradient so we must implement it
    # gy_x1, gx_x1 = cal_gradient2D(vertex_map[:,0:1,:,:])
    # gy_y1, gx_y1 = cal_gradient2D(vertex_map[:,1:2,:,:])
    # gy_z1, gx_z1 = cal_gradient2D(vertex_map[:,2:3,:,:])
    #
    # assert torch.allclose(gy_x,gy_x1)
    # assert torch.allclose(gx_x, gx_x1)
    # assert torch.allclose(gy_y, gy_y1)
    # assert torch.allclose(gx_y, gx_y1)
    # assert torch.allclose(gy_z, gy_z1)
    # assert torch.allclose(gx_z, gx_z1)


    gx = torch.cat([gx_x, gx_y, gx_z],dim=1)
    gy = torch.cat([gy_x, gy_y, gy_z],dim=1)

    # y cross x, producing outward normal
    if opengl:
        normal_map = torch.cross(gx, gy, dim=1)
    else:
        normal_map = torch.cross(gy, gx, dim=1)

    normal_map_normed = F.normalize(normal_map, dim=1)
    # n = torch.linalg.norm(N, dim=2)
    # N[:, :, 0] = N[:, :, 0].clone() / n
    # N[:, :, 1] = N[:, :, 1].clone() / n
    # N[:, :, 2] = N[:, :, 2].clone() / n
    #
    # NaNs = torch.isnan(N)
    #
    # check if this is differentiable
    # N[NaNs] = 0

    return normal_map_normed

def cal_gradient2D(x):
    """

    :param x: [B, 1, H, W]
    :return: gy: [B, 1, H, W]
             gx: [B, 1, H, W]
    """
    B, _, h, w = x.shape
    gy = torch.zeros_like(x)
    gx = torch.zeros_like(x)

    gy[:, :, 1:h - 1, :] = (x[:, :, 2:h, :] - x[:, :, 0:h - 2, :]) / 2
    gy[:, :, 0, :] = x[:, :, 1, :] - x[:, :, 0, :]
    gy[:, :, h - 1, :] = x[:, :, h - 1, :] - x[:, :, h - 2, :]

    gx[:, :, :, 1: w - 1] = (x[:, :, :, 2:w] - x[:, :, :, 0:w - 2]) / 2
    gx[:, :, :, 0] = x[:, :, :, 1] - x[:, :, :, 0]
    gx[:, :, :, w - 1] = x[:, :, :, w - 1] - x[:, :, :, w - 2]

    return gy, gx

# tested in test_batch_lookat_mat()
def batch_lookat_mat(eye, at, up=None):
    """
    Generate batched lookAt matrix using eye, at input
    :param eye: [B, 3] position of the camera in world coordinate
    :param at: [B, 3] position of the camera in world coordinate
    :param up: [B, 3] position of the camera in world coordinate, default to +Y axis as in openGL system
    :return: T: [B, 4, 4] batched lookAt matrix
    """

    B, _ = eye.shape

    if up is None:
        up = torch.tensor([0., 1., 0.]).cuda().unsqueeze(0).repeat(B, 1)

    a = torch.linalg.norm(up, dim=1, keepdim=True)
    up = up / torch.linalg.norm(up, dim=1, keepdim=True)
    f = (at - eye)
    f = f / torch.linalg.norm(f, dim=1, keepdim=True)

    r = torch.cross(f, up, dim=1)
    r = r / torch.linalg.norm(r, dim=1, keepdim=True)

    u = torch.cross(r, f, dim=1)
    u = u / torch.linalg.norm(u, dim=1, keepdim=True)

    row1 = torch.cat([r, torch.sum(-r * eye, dim=1, keepdim=True)], dim=1).unsqueeze(1) # [B, 1, 4]
    row2 = torch.cat([u, torch.sum(-u * eye, dim=1, keepdim=True)], dim=1).unsqueeze(1)  # [B, 1, 4]
    row3 = torch.cat([-f, torch.sum(f * eye, dim=1, keepdim=True)], dim=1).unsqueeze(1)  # [B, 1, 4]
    row4 = torch.tensor([0., 0., 0., 1.]).cuda().unsqueeze(0).repeat(B, 1).unsqueeze(1)   # [B, 1, 4]

    T = torch.cat([row1, row2, row3, row4], dim=1) # [B, 4, 4]

    return T




def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def batch_inv_SE3(T):
    """

    :param T: [B, 4, 4] SE3 matrix
    :return: T_inv: [B, 4, 4] inverse of SE3 matrix
    """
    B = T.size(0)
    rot_mat = T[:, :3, :3] # [B, 3, 3]
    trans = T[:, :3, 3:4] # [B, 3, 1]

    rot_mat_inv = torch.transpose(rot_mat, 1, 2) # [B, 3, 3], R^T
    trans_inv = torch.bmm(rot_mat_inv, -trans) # [B, 3, 1], -R^T t

    T_inv = torch.zeros((B, 4, 4), dtype=T.dtype, device=T.device)
    T_inv[:, :3, :3] = rot_mat_inv
    T_inv[:, :3, 3:4] = trans_inv
    T_inv[:, 3, 3] = 1.

    return T_inv




# tested in test_batch_rot_mat_distance()
# however, this is not that robust due to rounding errors in acos input
# TODO: decide if this function can be used
def batch_rot_mat_distance(R1, R2):
    """
    Calculate the distance of two rotaiton matrices using trace formulation
    i.e. delta R = R1 R2^-1 = R1 R2^T
    :param R1: [B, 3, 3] source rotation matrix
    :param R2: [B, 3, 3] target rotation matrix
    :return: delta_radian: [B, 1]
    """

    delta_rot_mat = torch.bmm(R1, R2.permute(0,2,1)) # [B, 3, 3]
    trace_delta_rot_mat = torch.diagonal(delta_rot_mat, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) # [B, 1]
    delta_radian = torch.acos((trace_delta_rot_mat-1)*0.5) # [B, 1]

    return delta_radian


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-8
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # clipping is not important here; if q_abs is small, the candidate won't be picked
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clip(0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

# ================= Testing functions ================

def test_batch_inv_SE3():

    B = 8
    eye = torch.rand(B, 3).cuda()
    at = torch.rand(B, 3).cuda()
    T = batch_lookat_mat(eye, at)

    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        T_inv_torch = torch.linalg.inv(T)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        T_inv = batch_inv_SE3(T)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    torch.testing.assert_allclose(T_inv, T_inv_torch)



# TODO: move to test_geometry.py
def test_batch_lookat_mat():

    B = 4
    eye = torch.rand(B, 3).cuda()
    at = torch.rand(B, 3).cuda()

    T_np = [torch.tensor(lookAtMatrix(eye=eye[b].cpu().numpy().squeeze(),
                                      at=at[b].cpu().numpy().squeeze())).type(torch.cuda.FloatTensor).unsqueeze(0)  for b in range(B)]
    T_np = torch.cat(T_np, dim=0)

    T = batch_lookat_mat(eye, at)

    torch.testing.assert_allclose(T, T_np)


def test_matrix_axis_angle_conversion():
    B = 4
    rot = torch.rand(B, 3).cuda()*1e-3
    rot_mat = axis_angle_to_matrix(rot)  # [B, 3, 3]

    rot_recon = matrix_to_axis_angle(rot_mat)

    torch.testing.assert_allclose(rot, rot_recon)

def test_batch_rot_mat_distance():

    B = 4
    rot = torch.rand(B, 3).cuda()
    rot_mat = axis_angle_to_matrix(rot)  # [B, 3, 3]
    delta_radian = batch_rot_mat_distance(rot_mat, rot_mat)



    torch.testing.assert_allclose(delta_radian, torch.zeros(B, 1).cuda())

    rot1 = torch.tensor([0, np.pi/2, 0]).cuda().unsqueeze(0)
    rot2 = torch.tensor([0, -np.pi / 2, 0]).cuda().unsqueeze(0)

    rot_mat1 = axis_angle_to_matrix(rot1)  # [B, 3, 3]
    rot_mat2 = axis_angle_to_matrix(rot2)  # [B, 3, 3]
    delta_radian = batch_rot_mat_distance(rot_mat1, rot_mat2)
    torch.testing.assert_allclose(delta_radian, torch.tensor([np.pi]).cuda().unsqueeze(0))



if __name__=='__main__':
    test_batch_lookat_mat()
    for i in range(100):
        # test_batch_rot_mat_distance()
        # test_matrix_axis_angle_conversion()
        test_batch_inv_SE3()

    print('All test passed')








_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())



def meshgrid(H, W, B=None, is_cuda=False):
    """ torch version of numpy meshgrid function
    :input
    :param height
    :param width
    :param batch size
    :param initialize a cuda tensor if true
    -------
    :return
    :param meshgrid in column
    :param meshgrid in row
    """
    u = torch.arange(0, W)
    v = torch.arange(0, H)

    if is_cuda:
        u, v = u.cuda(), v.cuda()

    u = u.repeat(H, 1).view(1,H,W)
    v = v.repeat(W, 1).t_().view(1,H,W)

    if B is not None:
        u, v = u.repeat(B,1,1,1), v.repeat(B,1,1,1)
    return u, v

def generate_camera_coord_grid(B, H, W, K):
    """
    Generate a batch of image grids from pixel coordinates to camera coordinates
        px = (u - cx) / fx
        py = (y - cy) / fy
    :param B: Batch size
    :param H: image height
    :param W: image width
    :param K: [3, 3] intrinsic matrix
    :return: px: [B, 1, H, W]
             py: [B, 1, H, W]
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    x = torch.arange(0, W).cuda()
    y = torch.arange(0, H).cuda()

    v_grid, u_grid = torch.meshgrid(y, x)
    px = ((u_grid - cx) / fx).view(1, 1, H, W).repeat(B, 1, 1, 1)
    py = ((v_grid - cy) / fy).view(1, 1, H, W).repeat(B, 1, 1, 1)

    return px, py


def batch_inverse_Rt(R, t):
    """ The inverse of the R, t: [R' | -R't]
        function tested in 'test_geometry.py'
    :input
    :param rotation Bx3x3
    :param translation Bx3
    ----------
    :return
    :param rotation inverse Bx3x3
    :param translation inverse Bx3
    """
    R_t = R.transpose(1, 2)
    t_inv = -torch.bmm(R_t, t.contiguous().view(-1, 3, 1))

    return R_t, t_inv.view(-1, 3)


def batch_Rt_compose(d_R, d_t, R0, t0):
    """ Compose operator of R, t: [d_R*R | d_R*t + d_t]
        We use left-mulitplication rule here.
        function tested in 'test_geometry.py'

    :input
    :param rotation incremental Bx3x3
    :param translation incremental Bx3
    :param initial rotation Bx3x3
    :param initial translation Bx3
    ----------
    :return
    :param composed rotation Bx3x3
    :param composed translation Bx3
    """
    R1 = d_R.bmm(R0)
    t1 = d_R.bmm(t0.view(-1, 3, 1)) + d_t.view(-1, 3, 1)
    return R1, t1.view(-1, 3)


def batch_Rt_between(R0, t0, R1, t1):
    """ Between operator of R, t, transform of T_0=[R0, t0] to T_1=[R1, t1]
        which is T_1 \compose T^{-1}_0
        function tested in 'test_geometry.py'

    :input
    :param rotation of source Bx3x3
    :param translation of source Bx3
    :param rotation of target Bx3x3
    :param translation of target Bx3
    ----------
    :return
    :param incremental rotation Bx3x3
    :param incremnetal translation Bx3
    """
    R0t = R0.transpose(1, 2)
    dR = R1.bmm(R0t)
    dt = t1.view(-1, 3) - dR.bmm(t0.view(-1, 3, 1)).view(-1, 3)
    return dR, dt


def batch_skew(w):
    """ Generate a batch of skew-symmetric matrices.
        function tested in 'test_geometry.py'
    :input
    :param skew symmetric matrix entry Bx3
    ---------
    :return
    :param the skew-symmetric matrix Bx3x3
    """
    B, D = w.size()
    assert (D == 3)
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)


def batch_twist2Mat(twist):
    """ The exponential map from so3 to SO3
        Calculate the rotation matrix using Rodrigues' Rotation Formula
        http://electroncastle.com/wp/?p=39
        or Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (13)-(15)
        @todo: may rename the interface to batch_so3expmap(twist)
        functioned tested with cv2.Rodrigues implementation in 'test_geometry.py'
    :input
    :param twist/axis angle Bx3 \in \so3 space
    ----------
    :return
    :param Rotation matrix Bx3x3 \in \SO3 space
    """
    B = twist.size()[0]
    theta = twist.norm(p=2, dim=1).view(B, 1)
    w_so3 = twist / theta.expand(B, 3)
    W = batch_skew(w_so3)
    return torch.eye(3).repeat(B, 1, 1).type_as(W) \
           + W * sin(theta.view(B, 1, 1)) \
           + W.bmm(W) * (1 - cos(theta).view(B, 1, 1))


def batch_mat2angle(R):
    """ Calcuate the axis angles (twist) from a batch of rotation matrices
        Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (17)
        function tested in 'test_geometry.py'
    :input
    :param Rotation matrix Bx3x3 \in \SO3 space
    --------
    :return
    :param the axis angle B
    """
    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    R_trace = torch.stack(R1)
    # clamp if the angle is too large (break small angle assumption)
    # @todo: not sure whether it is absoluately necessary in training.
    angle = acos(((R_trace - 1) / 2).clamp(-1, 1))
    return angle


def batch_mat2twist(R):
    """ The log map from SO3 to so3
        Calculate the twist vector from Rotation matrix
        Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (18)
        @todo: may rename the interface to batch_so3logmap(R)
        function tested in 'test_geometry.py'
        @note: it currently does not consider extreme small values.
        If you use it as training loss, you may run into problems
    :input
    :param Rotation matrix Bx3x3 \in \SO3 space
    --------
    :param the twist vector Bx3 \in \so3 space
    """
    B = R.size()[0]

    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    tr = torch.stack(R1)
    theta = acos(((tr - 1) / 2).clamp(-1, 1))

    r11, r12, r13, r21, r22, r23, r31, r32, r33 = torch.split(R.view(B, -1), 1, dim=1)
    res = torch.cat([r32 - r23, r13 - r31, r21 - r12], dim=1)

    magnitude = (0.5 * theta / sin(theta))

    return magnitude.view(B, 1) * res


def batch_warp_inverse_depth(p_x, p_y, p_invD, pose, K):
    """ Compute the warping grid w.r.t. the SE3 transform given the inverse depth
    :input
    :param p_x the x coordinate map
    :param p_y the y coordinate map
    :param p_invD the inverse depth
    :param pose the 3D transform in SE3
    :param K the intrinsics
    --------
    :return
    :param projected u coordinate in image space Bx1xHxW
    :param projected v coordinate in image space Bx1xHxW
    :param projected inverse depth Bx1XHxW
    """
    [R, t] = pose
    B, _, H, W = p_x.shape

    I = torch.ones((B, 1, H, W)).type_as(p_invD)
    x_y_1 = torch.cat((p_x, p_y, I), dim=1)

    warped = torch.bmm(R, x_y_1.view(B, 3, H * W)) + \
             t.view(B, 3, 1).expand(B, 3, H * W) * p_invD.view(B, 1, H * W).expand(B, 3, H * W)

    x_, y_, s_ = torch.split(warped, 1, dim=1)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    u_ = (x_ / s_).view(B, -1) * fx + cx
    v_ = (y_ / s_).view(B, -1) * fy + cy

    inv_z_ = p_invD / s_.view(B, 1, H, W)

    return u_.view(B, 1, H, W), v_.view(B, 1, H, W), inv_z_


def batch_warp_affine(pu, pv, affine):
    # A = affine[:,:,:2]
    # t = affine[:,:, 2]
    B, _, H, W = pu.shape
    ones = torch.ones(pu.shape).type_as(pu)
    uv = torch.cat((pu, pv, ones), dim=1)
    uv = torch.bmm(affine, uv.view(B, 3, -1))  # + t.view(B,2,1)
    return uv[:, 0].view(B, 1, H, W), uv[:, 1].view(B, 1, H, W)


def check_occ(inv_z_buffer, inv_z_ref, u, v, thres=1e-1):
    """ z-buffering check of occlusion
    :param inverse depth of target frame
    :param inverse depth of reference frame
    """
    B, _, H, W = inv_z_buffer.shape

    inv_z_warped = warp_features(inv_z_ref, u, v)
    inlier = (inv_z_buffer > inv_z_warped - thres)

    inviews = inlier & (u > 0) & (u < W) & \
              (v > 0) & (v < H)

    return 1 - inviews


def warp_features(F, u, v):
    """
    Warp the feature map (F) w.r.t. the grid (u, v)
    """
    B, C, H, W = F.shape

    u_norm = u / ((W - 1) / 2) - 1
    v_norm = v / ((H - 1) / 2) - 1
    uv_grid = torch.cat((u_norm.view(B, H, W, 1), v_norm.view(B, H, W, 1)), dim=3)
    F_warped = nn.functional.grid_sample(F, uv_grid,
                                         mode='bilinear', padding_mode='border')
    return F_warped


def batch_transform_xyz(xyz_tensor, R, t, get_Jacobian=True):
    '''
    transform the point cloud w.r.t. the transformation matrix
    :param xyz_tensor: B * 3 * H * W
    :param R: rotation matrix B * 3 * 3
    :param t: translation vector B * 3
    '''
    B, C, H, W = xyz_tensor.size()
    t_tensor = t.contiguous().view(B, 3, 1).repeat(1, 1, H * W)
    p_tensor = xyz_tensor.contiguous().view(B, C, H * W)
    # the transformation process is simply:
    # p' = t + R*p
    xyz_t_tensor = torch.baddbmm(t_tensor, R, p_tensor)

    if get_Jacobian:
        # return both the transformed tensor and its Jacobian matrix
        J_r = R.bmm(batch_skew_symmetric_matrix(-1 * p_tensor.permute(0, 2, 1)))
        J_t = -1 * torch.eye(3).view(1, 3, 3).expand(B, 3, 3)
        J = torch.cat((J_r, J_t), 1)
        return xyz_t_tensor.view(B, C, H, W), J
    else:
        return xyz_t_tensor.view(B, C, H, W)


def flow_from_rigid_transform(depth, extrinsic, intrinsic):
    """
    Get the optical flow induced by rigid transform [R,t] and depth
    """
    [R, t] = extrinsic
    [fx, fy, cx, cy] = intrinsic


def batch_project(xyz_tensor, K):
    """ Project a point cloud into pixels (u,v) given intrinsic K
    [u';v';w] = [K][x;y;z]
    u = u' / w; v = v' / w
    :param the xyz points
    :param calibration is a torch array composed of [fx, fy, cx, cy]
    -------
    :return u, v grid tensor in image coordinate
    (tested through inverse project)
    """
    B, _, H, W = xyz_tensor.size()
    batch_K = K.expand(H, W, B, 4).permute(2, 3, 0, 1)

    x, y, z = torch.split(xyz_tensor, 1, dim=1)
    fx, fy, cx, cy = torch.split(batch_K, 1, dim=1)

    u = fx * x / z + cx
    v = fy * y / z + cy
    return torch.cat((u, v), dim=1)




def batch_euler2mat(ai, aj, ak, axes='sxyz'):
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    :param axes : Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    -------
    :return rotation matrix, array-like shape (B, 3, 3)
    Tested w.r.t. transforms3d.euler module
    """
    B = ai.size()[0]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    order = [i, j, k]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    # M = torch.zeros(B, 3, 3).cuda()
    if repetition:
        c_i = [cj, sj * si, sj * ci]
        c_j = [sj * sk, -cj * ss + cc, -cj * cs - sc]
        c_k = [-sj * ck, cj * sc + cs, cj * cc - ss]
    else:
        c_i = [cj * ck, sj * sc - cs, sj * cc + ss]
        c_j = [cj * sk, sj * ss + cc, sj * cs - sc]
        c_k = [-sj, cj * si, cj * ci]

    def permute(X):  # sort X w.r.t. the axis indices
        return [x for (y, x) in sorted(zip(order, X))]

    c_i = permute(c_i)
    c_j = permute(c_j)
    c_k = permute(c_k)

    r = [torch.stack(c_i, 1),
         torch.stack(c_j, 1),
         torch.stack(c_k, 1)]
    r = permute(r)

    return torch.stack(r, 1)


def batch_mat2euler(M, axes='sxyz'):
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param array-like shape (3, 3) or (4, 4). Rotation matrix or affine.
    :param  Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    --------
    :returns
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if repetition:
        sy = torch.sqrt(M[:, i, j] ** 2 + M[:, i, k] ** 2)
        # A lazy way to cope with batch data. Can be more efficient
        mask = ~(sy > 1e-8)
        ax = atan2(M[:, i, j], M[:, i, k])
        ay = atan2(sy, M[:, i, i])
        az = atan2(M[:, j, i], -M[:, k, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask], M[:, j, j][mask])
            ay[mask] = atan2(sy[mask], M[:, i, i][mask])
            az[mask] = 0.0
    else:
        cy = torch.sqrt(M[:, i, i] ** 2 + M[:, j, i] ** 2)
        mask = ~(cy > 1e-8)
        ax = atan2(M[:, k, j], M[:, k, k])
        ay = atan2(-M[:, k, i], cy)
        az = atan2(M[:, j, i], M[:, i, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask], M[:, j, j][mask])
            ay[mask] = atan2(-M[:, k, i][mask], cy[mask])
            az[mask] = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az