from sympy import true
import torch
from torch import Tensor, device
from jaxtyping import Float
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from einops import rearrange, repeat, pack
import cv2
import os
from pathlib import Path
from src.visualization.camera_trajectory.wobble import (
    generate_wobble,
)
from src.visualization.pca_map import apply_pca_colormap

from src.misc.data_utils import data_wrapper
from src.model.decoder import get_decoder, DecoderCfg
from src.model.encoder import get_encoder, EncoderCfg


@dataclass
class SpatialSplatConfig:
    encoder: EncoderCfg
    decoder: DecoderCfg
    ckpt_path : str
    near: float
    far: float

class SpatialSplat(torch.nn.Module):
    def __init__(self, config: SpatialSplatConfig,
                 devices = 'cuda:0', 
                 eval : bool =True):
        self.cfg = config
        self.device = devices
        super().__init__()
        self.init_model(eval=eval)

    def init_model(self, eval : bool = True
                   ) -> None:
        self.encoder,_ = get_encoder(self.cfg.encoder)
        self.decoder = get_decoder(self.cfg.decoder)
        ckpt_weights = torch.load(self.cfg.ckpt_path, map_location='cpu')
        ckpt_weights = ckpt_weights['state_dict']
        ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(ckpt_weights, strict=False)    
        # freeze encoder
        if self.cfg.encoder.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.encoder.to(device=self.device)
        self.decoder.to(device=self.device)
        self.gaussians_list = None
        if eval:
            self.encoder.eval()
            self.decoder.eval()
    
    @torch.no_grad()
    def inference(self, img_list: List[str],
                  output_path: str = './output',
                  intrinsics: Float[Tensor, 'b v 3 3'] = None,
                  extrinsics: Float[Tensor, 'b v 4 4'] = None,
                  ) -> Float[Tensor, 'b n c']:
        '''
        Input:
            img_list: list of img paths
            intrinsics: [b, v, 3, 3]
            extrinsics: [b, v, 4, 4]
        Output: 
            extracted features: [b, n, c]
        '''
        imgs = []
        for img_path in img_list:
            im = cv2.imread(img_path)
            im = im[:,:,[2,1,0]] # BGR to RGB
            imgs.append(im)
        imgs = torch.tensor(np.stack(imgs, axis=0)).permute(0,3,1,2)[None]
        data_dict = data_wrapper(imgs, self.cfg.near, self.cfg.far, 
                                 intrinsics=intrinsics, 
                                 extrinsics=extrinsics, device=self.device)   
        self.render_video_wobble(data_dict, output_path)
        
        
    def render_video_wobble(self, data_dict,
                            output_path : str = './output') -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = data_dict["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = data_dict["extrinsics"][:, 0, :3, 3]
            origin_b = data_dict["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                data_dict["extrinsics"][:, 0],
                delta,
                t,
            )
            intrinsics = repeat(
                data_dict["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(data_dict, trajectory_fn, "wobble", num_frames=60, output_path=output_path)
        
    def render_video_generic(
        self,
        data_dict,
        trajectory_fn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        output_path : str = './output'
    ) -> None:
        gaussians = self.encoder(data_dict)[0]
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = data_dict["image"].shape

        near = repeat(data_dict["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(data_dict["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w)
        )
        instance = output.instance_image[0]
        
        instance = [apply_pca_colormap(frame.permute(1,2,0)) for frame in instance]
        instance_images = [frame.permute(2,0,1) for frame in instance]

        # create instance features video
        instance_video = torch.stack(instance_images)
        instance_video = (instance_video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            instance_video = pack([instance_video, instance_video[::-1][1:-1]], "* c h w")[0]
        dir = Path(output_path)
        dir.mkdir(exist_ok=True, parents=True)
        size = (256,256)
        mp4_handle = cv2.VideoWriter(output_path + f'/{name}_instance.avi',
                                     cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 30,size)
        for frame in instance_video:
            frame = frame.transpose(1,2,0)[:,:,[2,1,0]]
            mp4_handle.write(frame)
        mp4_handle.release()
        print(f'Saved instance feature video to {output_path}/{name}_instance.avi')
        
        # create rgb video
        rgb_images = [frame for frame in output.color[0]]
        rgb_video = torch.stack(rgb_images)
        rgb_video = (rgb_video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            rgb_video = pack([rgb_video, rgb_video[::-1][1:-1]], "* c h w")[0]  
        dir = Path(output_path)
        dir.mkdir(exist_ok=True, parents=True)
        size = (256,256)
        mp4_handle = cv2.VideoWriter(output_path + f'/{name}_rgb.avi',
                                     cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 30,size)
        for frame in rgb_video:
            frame = frame.transpose(1,2,0)[:,:,[2,1,0]]
            mp4_handle.write(frame)
        mp4_handle.release()
        print(f'Saved rgb video to {output_path}/{name}_rgb.avi')
        
        cv2.destroyAllWindows()