from dataclasses import dataclass

import torch
from einops import reduce, rearrange
from jaxtyping import Float, Bool
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import open3d as o3d


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    
    def __init__(self, cfg: LossDepthCfgWrapper):
        super().__init__(cfg)
        # set a learnable temperature
        self.alpha = 0.2
        self.reg_loss = torch.nn.MSELoss(reduction='none')
    
    def get_conf_log(self, x):
        return x, torch.log(x)
    
    def norm_pts(self, pts: Float[Tensor, "b n 3"],
                   mask: Bool[Tensor, "b n"]) -> Float[Tensor, "b n 3"]:
        pts[~mask] = 0
        all_dis = pts.norm(dim=-1)  
        nnz = mask.sum(dim=-1) # [b]
        norm_factor = all_dis.sum(dim=-1) / (nnz + 1e-6) # [b]
        norm_factor = norm_factor.clip(min=1e-8)
        # import ipdb; ipdb.set_trace()
        res = pts / norm_factor[:,None,None] # [b,n,3]
        return res
        
    
    def convert_depth2xyz(
        self,
        depth: Float[Tensor, "b v c height width"],
        intrinsics: Float[Tensor, "b v 3 3"],
        extrinsics: Float[Tensor, "b v 4 4"],
    ) -> Float[Tensor, "b v c height width"]:

        b,v,_,h,w = depth.shape
        x = torch.arange(w, device=depth.device, dtype=torch.float32)
        y = torch.arange(h, device=depth.device, dtype=torch.float32)
        x, y = torch.meshgrid(x, y)
        x = x.T
        y = y.T
        xyz = torch.stack([x, y, torch.ones_like(x)], dim=0)
        
        intrinsics = intrinsics.reshape(b*v, 3, 3) #[bv, 3, 3]
        intrinsics[:,0,:] = intrinsics[:,0,:] * w
        intrinsics[:,1,:] = intrinsics[:,1,:] * h
        extrinsics = extrinsics.reshape(b*v, 4, 4) #[bv, 4, 4]
        xyz = xyz.reshape(3, -1)[None].repeat(b*v, 1, 1) # [bv, 3, h*w]
        xyz = intrinsics.inverse().bmm(xyz)
        xyz = xyz * depth.reshape(b*v, 3, -1)

        xyz_tr = extrinsics[:, :3, :3].bmm(xyz) + extrinsics[:, :3, 3:4]
        return xyz_tr.reshape(b, v, 3, h, w)
        
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        masks: Tensor = None,
        clip_feats: Tensor = None,
    ) -> Float[Tensor, ""]:
        # get the xyz from the depth
        xyz_gt = batch["context"]["xyz"]
        # debug
        # color = batch["context"]["image"]
        # pcd = o3d.geometry.PointCloud()
        # pts = rearrange(xyz_gt[2,0], "c h w -> (h w) c")
        # colors = rearrange(color[2,0], "c h w -> (h w) c")
        # pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy()*0.5 + 0.5)
        # pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        # o3d.io.write_point_cloud("data/demo/pcd1.ply", pcd)
        # import ipdb; ipdb.set_trace()
        xyz_pred = gaussians.means # [b,n,3]
        conf_pred = gaussians.active # [b,n]
        xyz_gt = rearrange(xyz_gt, 'b v c h w -> b (v h w) c')
        xyz_mask = (xyz_gt[:,:,2] > batch["context"]["near"].min()) * (xyz_gt[:,:,2] < batch["context"]["far"].max()) # [b,n]
        norm_xyz_gt = self.norm_pts(xyz_gt, xyz_mask)
        norm_xyz_pred = self.norm_pts(xyz_pred, xyz_mask)
        xyz_gt = norm_xyz_gt[xyz_mask]
        xyz_pred = norm_xyz_pred[xyz_mask]
        # apply l2 loss on the xyz
        loss_xyz = self.reg_loss(xyz_pred, xyz_gt)
        # apply confidence-aware depth loss
        conf_pred = conf_pred[xyz_mask]
        conf_pred, log_conf_pred = self.get_conf_log(conf_pred)
        conf_loss = loss_xyz * conf_pred[:,None] - self.alpha * log_conf_pred[:,None]   
        
        return self.cfg.weight * conf_loss.mean()