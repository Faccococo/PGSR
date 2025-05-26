#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def matrix_to_euler_angles(matrix: torch.Tensor, convention="XYZ") -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """

    def _index_from_letter(letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin( (matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)).clip(-1,1))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_x, valid_y = torch.meshgrid(valid_x, valid_y, indexing='xy')
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    # xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    # xyz_world = xyz_world[...,:3]

    return xyz_cam

def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape 
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        cross_distance = 3
        bottom_point = xyz[..., 2 * cross_distance:hd,   cross_distance:wd-cross_distance, :]
        top_point    = xyz[..., 0:hd-2 * cross_distance, cross_distance:wd-cross_distance, :]
        right_point  = xyz[..., cross_distance:hd-cross_distance, 2 * cross_distance:wd,   :]
        left_point   = xyz[..., cross_distance:hd-cross_distance, 0:wd-2 * cross_distance, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (cross_distance, cross_distance, cross_distance, cross_distance), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)        
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world, offset, gt_image)

    return xyz_normal

def normal_from_neareast(normal, offset):
    _, hd, wd = normal.shape 
    left_top_point = normal[..., 0:hd-2, 0:wd-2]
    top_point      = normal[..., 0:hd-2, 1:wd-1]
    right_top_point= normal[..., 0:hd-2, 2:wd]
    left_point   = normal[..., 1:hd-1, 0:wd-2]
    right_point  = normal[..., 1:hd-1, 2:wd]
    left_bottom_point   = normal[..., 2:hd, 0:wd-2]
    bottom_point = normal[..., 2:hd,   1:wd-1]
    right_bottom_point   = normal[..., 2:hd, 2:wd]
    normals = torch.stack((left_top_point,top_point,right_top_point,left_point,right_point,left_bottom_point,bottom_point,right_bottom_point),dim=0)
    new_normal = (normals * offset[:,None,1:-1,1:-1]).sum(0)
    new_normal = torch.nn.functional.normalize(new_normal, p=2, dim=0)
    new_normal = torch.nn.functional.pad(new_normal, (1,1,1,1), mode='constant').permute(1,2,0)
    return new_normal

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrixTorch(znear, zfar, fovX, fovY):
    tanHalfFovY = ((fovY / 2)).tan()
    tanHalfFovX = ((fovX / 2)).tan()

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4).cuda()

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    convention="XYZ"
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def lift_to_poses(poses_):
    poses = torch.zeros(*poses_.shape[:-1],4,4).cuda()
    poses[...,-1,-1]=1
    poses[...,:3,:3] = euler_angles_to_matrix(poses_[...,:3])
    poses[...,:3,-1] = poses_[...,3:]
    return poses

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fl_x, fl_y, w, h):
    top = cy / fl_y * znear
    bottom = -(h - cy) / fl_y * znear

    left = -(w - cx) / fl_x * znear
    right = cx / fl_x * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid
