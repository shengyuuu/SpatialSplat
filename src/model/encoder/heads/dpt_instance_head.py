# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from .head_modules import UnetExtractor
from .postprocess import postprocess
from ...encodings.positional_encoding_3d import PositionalEncoding3D
from ...transformer.spatial_attention import SpatialTransformer, Normalize


class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)

    def forward(self, x, condition):
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition) 
        y = gamma * x + beta
        return y

class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768, feature_dim = 256, **kwargs):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.feats_upsample = Interpolate(scale_factor=2,mode='bilinear',align_corners=True)
        # self.feats_proj = nn.Sequential(
        #     nn.Conv2d(feature_dim + 32, feature_dim, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True),
    
        if self.language_scale:
            self.feats_proj = FiLM(feature_dim, self.language_dim)
            # language head
            self.language_feats_head = nn.Sequential(
            nn.Conv2d(feature_dim, self.language_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            )
            

    def forward(self, encoder_tokens: List[torch.Tensor], xyzs, imgs, language_feats=None, image_size=None, conf=None, feats_ex=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0]) # [bv, c, h, w] h = H // 2, w = W // 2

        # full resolution path
        path_1_up = self.feats_upsample(path_1) # [bv, c, h*2, w*2] 
        instance_full = self.head(path_1_up)
        # simple version
        # language_feats = self.language_feats_proj(language_feats)
        # language_feats = F.interpolate(language_feats, size=[H,W], mode='bilinear', align_corners=True)    
        # fusion_feats = path_1_up + language_feats
        # instance_full = self.head(fusion_feats)
        
        if not self.language_scale:
            return instance_full, None, None
        
        # 1/8 resolution path
        else:
            instance_low = F.interpolate(instance_full.clone().detach(), 
                size=[H//self.language_scale,W//self.language_scale], mode='nearest')
            # language_feats = self.language_feats_proj(language_feats)
            b,c,h,w = path_1.shape
            language_feats = F.interpolate(language_feats, size=[h,w], mode='bilinear', align_corners=True) # [bv, c, h, w]
            language_feats = rearrange(language_feats, 'bv c h w -> bv h w c') # [bv, c, h, w]
            img_feats = rearrange(path_1, 'bv c h w -> bv h w c') # [bv, c, h, w]
            fusion_feats = self.feats_proj(img_feats, language_feats) # [bv, c, h, w]
            fusion_feats = rearrange(fusion_feats, 'bv h w c -> bv c h w') # [bv, c, h, w]
            language = self.language_feats_head(fusion_feats) # [bv, 512, 32, 32]
            return instance_full, instance_low, language
        
        
        

        # exchange_feats
        language_feats_up = F.interpolate(feats_ex, size=[H,W], mode='nearest')    
        path_1_up = torch.cat([path_1_up, language_feats_up], dim=1)
        refine_feats = self.feats_proj(path_1_up)
        instance_full = self.head(refine_feats)
        
        if not self.language_scale:
            return instance_full, None, None
        
        # 1/8 resolution path
        else:
            instance_low = F.interpolate(instance_full.clone().detach(), 
                size=[H//self.language_scale,W//self.language_scale], mode='nearest') # [bv, c, H/16, W/16]            
            path_1_low = F.interpolate(refine_feats.clone(), size=[H // self.language_scale,W // self.language_scale], mode='bilinear', align_corners=True) # [bv, c, H/16, W/16]
            language_feats_low = F.interpolate(language_feats.clone(), size=[H // self.language_scale,W // self.language_scale], mode='bilinear', align_corners=True)
        
        # language 
            q = language_feats_low # [bv, c, h, w] h = H // 8, w = W // 8
            kv = rearrange(path_1_low, 'bv c h w -> bv (h w) c') # [bv, c, h, w]
            q = self.language_feats_attn(q, kv)
            language = self.language_feats_head(q)

            return instance_full, instance_low, language


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, language_dim = 0,language_scale = 0. ,postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        language_dim = language_dim,
                        language_scale = language_scale,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, xyz, imgs, language_feats = None, img_info = None, conf=None, feats_ex = None):
        out = self.dpt(x, xyz, imgs, language_feats = language_feats, image_size=(img_info[0], img_info[1]), conf=conf, feats_ex = feats_ex)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_instance_dpt_head(net, has_conf=False, out_nchan=3, language_dim = 0, language_scale = 0., postprocess_func=postprocess):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                language_dim = language_dim,
                                language_scale = language_scale,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='gs_instance')
