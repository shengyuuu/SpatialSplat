from sympy import false
import torch
from torch import Tensor
from jaxtyping import Float, Bool
from typing import List, Tuple

def norm_pts(pts: Float[Tensor, "b n 3"],
            mask: Bool[Tensor, "b n"]) -> Float[Tensor, "b n 3"]:    
    pts[~mask] = 0
    all_dis = pts.norm(dim=-1)  
    nnz = mask.sum(dim=-1)
    norm_factor = all_dis.sum(dim=-1) / (nnz + 1e-6) 
    norm_factor = norm_factor.clip(min=1e-8)

    return norm_factor

def get_scene_scale(pts_world: Float[Tensor, "b n 3"],
                    pts_scene: Float[Tensor, "b n 3"],
                    near: float,
                    far: float,
                    edge_masks: Bool[Tensor, "b n"] = None) -> Float[Tensor,'b']:
    masks = (pts_world[:, :, 2] > near) & (pts_world[:, :, 2] < far)
    # remove the edge
    if edge_masks is not None:
        masks = masks * edge_masks
    factor_world = norm_pts(pts_world, masks)
    factor_scene = norm_pts(pts_scene, masks)
    
    return factor_scene / factor_world

def convert_depth2xyz(
    depth: Float[Tensor, "b v 1 height width"],
    intrinsics: Float[Tensor, "b v 3 3"],
    extrinsics: Float[Tensor, "b v 4 4"],
) -> Float[Tensor, "b v 3 height width"]:

    b, v, _, h, w = depth.shape
    device = depth.device

    # Step 0: 还原被归一化的相机内参
    intrinsics = intrinsics.clone()
    intrinsics[:, :, 0, :] *= w
    intrinsics[:, :, 1, :] *= h

    # Step 1: 创建像素网格
    y_range = torch.arange(h, device=device)
    x_range = torch.arange(w, device=device)
    v_grid, u_grid = torch.meshgrid(y_range, x_range, indexing='ij')  # [h, w]

    # 扩展成 [b, v, h, w]
    u_grid = u_grid.unsqueeze(0).unsqueeze(0).expand(b, v, h, w)
    v_grid = v_grid.unsqueeze(0).unsqueeze(0).expand(b, v, h, w)

    # Step 2: 提取相机内参 fx, fy, cx, cy
    fx = intrinsics[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # [b, v, 1, 1]
    fy = intrinsics[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1)

    # Step 3: 使用 pinhole 模型从像素坐标转换到相机坐标
    z = depth.squeeze(2)  # [b, v, h, w]
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy
    xyz_camera = torch.stack([x, y, z], dim=2)  # [b, v, 3, h, w]

    # Step 4: 转换到世界坐标系
    ones = torch.ones_like(z).unsqueeze(2)  # [b, v, 1, h, w]
    xyz_homogeneous = torch.cat([xyz_camera, ones], dim=2)  # [b, v, 4, h, w]
    xyz_homo_flat = xyz_homogeneous.permute(0, 1, 3, 4, 2).reshape(b, v, h * w, 4, 1)  # [b, v, N, 4, 1]
    # import ipdb; ipdb.set_trace()
    xyz_world = extrinsics.unsqueeze(2) @ xyz_homo_flat  # [b, v, N, 4, 1]
    xyz_world = xyz_world.squeeze(-1)[..., :3]  # [b, v, N, 3]
    xyz_world = xyz_world.permute(0, 1, 3, 2).reshape(b, v, 3, h, w)  # [b, v, 3, h, w]

    return xyz_world