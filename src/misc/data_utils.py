import os
import torch
from torch import Tensor
from jaxtyping import Float, Bool
from typing import List, Tuple
import numpy as np
import cv2
import open3d as o3d
# import rearagen
from einops import rearrange
from src.misc.camera_utils import convert_depth2xyz
from src.misc.image_utils import image_preprocess

def data_wrapper(imgs: Float[Tensor,'b v c h w'],
                 near:float,
                 far:float,
                 extrinsics: Float[Tensor, 'b v 4 4'] = None,
                 intrinsics: Float[Tensor, 'b v 3 3'] = None,
                 device : str = 'cuda',
                 ) -> dict:
    ''' warp data to specifical format for feed-forward model
    Args:
        imgs : 0-255
        near : the nearst plane to camera used to clip a scene
        far : the farest plane to camera used to clip a scene
        extrinsics : camera extrinsics relative to each image
        intrinsics : camera intrinsics relative to each image
    Returns:
        out : a data dict
    '''
    # parameters config 
    b, v, c_in, h, w = imgs.shape
    if extrinsics is None:
        extrinsics = torch.eye(4)[None,None].repeat(b, v, 1, 1).to(device) # [B, V, 4, 4]
        extrinsics[:, 1, 0, 3] = 1 # move the second camera to right at x axis
    if intrinsics is None:
        intrinsics = torch.tensor([
            [1.2, 0, 0.5],
            [0, 1.2, 0.5],
            [0, 0, 1]
            ]).repeat(b, v, 1, 1).to(device)  # [B, V, 4, 4]
    else:
        intrinsics[:,:,0,:] /= w
        intrinsics[:,:,1,:] /= h
        
    # resize and center crop
    # if need_preprocess:            
    imgs = rearrange(imgs, 'b v c h w -> (b v) c h w')
    imgs, intrinsics = image_preprocess(imgs, intrinsics, (256, 256))
    # points generate
    if c_in == 3:
        rgb = rearrange(imgs, '(b v) c h w -> b v c h w', b=b,v=v).to(device)
        xyzs = torch.zeros_like(rgb)
    elif c_in == 4:
        assert intrinsics is not None and extrinsics is not None, 'Need intrinsics and extrinsics to recover 3D points'
        rgb = rearrange(imgs[:,:3], '(b v) c h w -> b v c h w', b=b,v=v).to(device)
        depth = rearrange(imgs[:,3:4], '(b v) c h w -> b v c h w', b=b,v=v).to(device)
        xyzs = convert_depth2xyz(depth=depth, intrinsics=intrinsics.clone(),extrinsics=extrinsics.clone())

    # other parameters need for feed-forward splat
    near = torch.tensor([near]).to(device)[None].repeat(b,v) # [B, V,]
    far = torch.tensor([far]).to(device)[None].repeat(b,v) # [B, V,]
    index = [0, 1]
    scale = 1. #[1]
    extrinsics[:, :, :3, 3] /= scale # [B, V, 4, 4]
    overlap = 0.5

    return {"extrinsics": extrinsics.float(),
            "intrinsics": intrinsics.float(),
            "image": rgb.float(),
            "xyz": xyzs.float(),
            "mask_image": None,
            "near": near.float(),
            "far": far.float(),
            "index": index,
            "scale": scale,
            "overlap": overlap,
            }
    
def select_sub_warped_data(data: dict, 
                            index: List[int], 
                            ) -> dict:
        """
        select sub warped data
        Args:
            data: a dict contains all data
            index: a list of index to select
        Returns:    
            data: a dict contains selected data
        """
        sub_data = {}
        
        sub_data["image"] = data["image"][:,index]
        sub_data["xyz"] = data["xyz"][:,index]
        sub_data["extrinsics"] = data["extrinsics"][:,index]
        sub_data["intrinsics"] = data["intrinsics"][:,index]
        sub_data["near"] = data["near"][:,index]
        sub_data["far"] = data["far"][:,index]
        
        return sub_data
    
def add_new_view(view, data_warper):
    raise NotImplementedError
    pass

def generate_scene_and_object_pointcloud(gaussians, data, data_path, mask1, mask2):
    """
    从已提取的3D高斯和mask图像生成场景点云与目标点云，并保存为PLY文件。

    参数:
        gaussians: 模型输出的高斯对象，包含属性 means。
        data: 包含图像信息的字典，需含有 "image" 键，形状为 [v, c, h, w]。
        data_path: 保存PLY和mask图像的路径。
        mask1_name: mask1图像文件名，默认 '1mask.png'。
        mask2_name: mask2图像文件名，默认 '2mask.png'。
    """
    # 获取高斯点
    x = gaussians.means[0]  # [v*h*w, 3]

    # 图像颜色处理
    colors = data["image"][0]  # [v, c, h, w]
    v, c, h, w = colors.shape
    colors = rearrange(colors, "v c h w -> (v h w) c")
    input_std = torch.tensor([0.5, 0.5, 0.5], device=colors.device)
    input_mean = torch.tensor([0.5, 0.5, 0.5], device=colors.device)
    colors = colors * input_std + input_mean

    # 拆分前后两帧
    x1, x2 = x[h * w:], x[:h * w]
    colors1, colors2 = colors[h * w:], colors[:h * w]

    # 生成场景点云
    scene_points = torch.cat([x1, x2], dim=0)
    scene_colors = torch.cat([colors1, colors2], dim=0)
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points.detach().cpu().numpy())
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors.detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(data_path, "scene.ply"), scene_pcd)

    mask1 = cv2.resize(mask1, (w, h)) > 0
    mask2 = cv2.resize(mask2, (w, h)) > 0
    mask1 = mask1.flatten()
    mask2 = mask2.flatten()

    # 提取目标点云
    object_points1 = x2[mask1]
    object_colors1 = colors2[mask1]
    object_points2 = x1[mask2]
    object_colors2 = colors1[mask2]
    object_points = torch.cat([object_points1, object_points2], dim=0)
    object_colors = torch.cat([object_colors1, object_colors2], dim=0)

    # 保存目标点云
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points.detach().cpu().numpy())
    object_pcd.colors = o3d.utility.Vector3dVector(object_colors.detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(data_path, "object.ply"), object_pcd)

    print("点云保存完成:scene.ply 和 object.ply")
    
def select_paired_images(data_budget:dict, method: str = 'default') -> List[int]:
    """
    select image pairs which has largest overlap with the latest view
    output is a list of ids
    """
    if method == 'default':
        # default method
        # select the first two images
        return [0, -1]
    else:
        raise NotImplementedError(f"Method {method} not implemented")    