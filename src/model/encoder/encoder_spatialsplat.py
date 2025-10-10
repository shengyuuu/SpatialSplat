from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import time
from torchvision import transforms

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ..language_extractor import build_language_extractor
from ..transformer.unet import UNetModel
import concurrent.futures
inf = float('inf')


transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderSpatialSplatCfg:
    name: Literal["spatialsplat"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    gs_instance_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    language_scale: int = 8


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderSpatialSplat(Encoder[EncoderSpatialSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderSpatialSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        # 需要language监督
        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in + 1# 1 for opacity, 1 for active
        self.language_scale = cfg.language_scale

        self.gs_params_head_type = cfg.gs_params_head_type
        self.gs_instance_head_type = cfg.gs_instance_head_type

        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        self.set_gs_params_head(cfg, cfg.gs_params_head_type, self.gs_instance_head_type)
        
        if self.gaussian_adapter.d_instance == 64:
            self.lang_feats_conv = nn.Sequential(
                nn.Conv2d(64, 512, 3, 1, 1),
            )
        
    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
    
    def set_gs_params_head(self, cfg, head_type, instance_head_type):
        if head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim,
                                                    language_scale=self.language_scale)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim,
                                                    language_scale=self.language_scale)
        elif head_type == 'bi_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim,
                                                    language_scale=self.language_scale)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
        
        # instance channel head # 8
        if self.gaussian_adapter.d_instance > 0:
            if instance_head_type == 'dpt_instance':
                self.instance_head = head_factory(instance_head_type, 'gs_instance', self.backbone, has_conf=False, 
                                                out_nchan=self.gaussian_adapter.d_instance, language_dim=self.gaussian_adapter.d_language,
                                                language_scale=self.language_scale)
                self.instance_head2 = head_factory(instance_head_type, 'gs_instance', self.backbone, has_conf=False, 
                                                out_nchan=self.gaussian_adapter.d_instance, language_dim=self.gaussian_adapter.d_language,
                                                language_scale=self.language_scale)
            elif instance_head_type == 'bi_instance':
                self.instance_head = head_factory(instance_head_type, 'gs_instance', self.backbone, has_conf=False, 
                                                out_nchan=8, language_dim=self.gaussian_adapter.d_language,
                                                language_scale=self.language_scale,
                                                )
            else:
                raise NotImplementedError(f"unexpected {instance_head_type=}")
            self.need_instance = True
        else:
            self.need_instance = False
        
        # language
        if self.gaussian_adapter.d_language > 0 or self.gaussian_adapter.d_instance == 64:
            self.need_language = True
            self.is_language_extractor_on_cuda = False
        else:
            self.need_language = False
            self.is_language_extractor_on_cuda = False
            
    def language_extractor_on_cuda(self,device):
        self.language_extractor = build_language_extractor(
            model_name=self.gaussian_adapter.language_model_type,
            device=device)
        self.is_language_extractor_on_cuda = True

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def _extract_language_features(self, context):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        if self.need_language:
            if not self.is_language_extractor_on_cuda:
                self.language_extractor_on_cuda(device)        
            l_input = context["image"].reshape(b * v, 3, h, w)* 0.5 + 0.5
            language_features = self.language_extractor.extract_language_features(
                l_input)
            language_features = rearrange(language_features, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            language_features = None
        
        return language_features
        
    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        # Encode the context images.
        dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
            
        # language features
        language_features = self._extract_language_features(context)
        
        # for means head
        res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        # for the 3DGS params heads
        if self.gs_params_head_type == 'dpt_gs':
            GS_res1, GS_res1_low = self.gaussian_param_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), 
                                                                view1['img'][:, :3], shape1[0].cpu().tolist())
            GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
            GS_res1_low = rearrange(GS_res1_low, "b d h w -> b (h w) d") if GS_res1_low is not None else None
            
            GS_res2, GS_res2_low = self.gaussian_param_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), 
                                                                view2['img'][:, :3], shape2[0].cpu().tolist())
            GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            GS_res2_low = rearrange(GS_res2_low, "b d h w -> b (h w) d") if GS_res2_low is not None else None
        else:
            raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

        # for the instance head
        if self.need_instance:

            language_features_ex = language_features

            if self.gs_instance_head_type == 'dpt_instance':
                instance_res_1, instance_res_1_low, language_1 = self.instance_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2),
                                                    view1['img'][:, :3], language_features[:,0] if language_features is not None else None, 
                                                    shape1[0].cpu().tolist(),
                                                    feats_ex = language_features_ex[:,0] if language_features is not None else None
                                                    )   
                instance_res_1 = rearrange(instance_res_1, "b d h w -> b (h w) d")
                instance_res_1_low = rearrange(instance_res_1_low, "b d h w -> b (h w) d") if instance_res_1_low is not None else None
                language_1 = rearrange(language_1, "b d h w -> b (h w) d") if language_1 is not None else None
                
                instance_res_2, instance_res_2_low, language_2 = self.instance_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2),
                                                    view2['img'][:, :3], language_features[:,1] if language_features is not None else None, 
                                                    shape2[0].cpu().tolist(), 
                                                    feats_ex = language_features_ex[:,1] if language_features is not None else None
                                                    )
                instance_res_2 = rearrange(instance_res_2, "b d h w -> b (h w) d")
                instance_res_2_low = rearrange(instance_res_2_low, "b d h w -> b (h w) d") if instance_res_2_low is not None else None
                language_2 = rearrange(language_2, "b d h w -> b (h w) d") if language_2 is not None else None
            else:
                raise NotImplementedError(f"unexpected {self.gs_instance_head_type=}")

        # for means
        pts3d_list = []
        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces
        pts3d_list.append(pts_all)
        
        # for gaussian parameters
        gaussians_list = []
        gaussians = torch.stack([GS_res1, GS_res2], dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        gaussians[..., 0] = gaussians[..., 0].sigmoid()
        gaussians_list.append(gaussians)
    
        # for instance
        if self.need_instance:
            instances = torch.stack([instance_res_1, instance_res_2], dim=1)
            instances = rearrange(instances, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            # Normalize the instance features for cosine similarity.
            instances = F.normalize(instances, p=2, dim=-1)
            gaussians_list[0] = torch.cat([gaussians_list[0][..., 0:1], instances, gaussians_list[0][..., 1:]], dim=-1)
        
        # for language
        languages_list = [None]
        if self.language_scale:
            pts3d1_low = F.interpolate(res1['pts3d'].permute(0, 3, 1, 2), scale_factor=1. / self.language_scale, mode='nearest')
            pts3d1_low = rearrange(pts3d1_low, "b d h w -> b (h w) d")
            pts3d2_low = F.interpolate(res2['pts3d'].permute(0, 3, 1, 2), scale_factor=1. / self.language_scale, mode='nearest')
            pts3d2_low = rearrange(pts3d2_low, "b d h w -> b (h w) d")
            pts_all_low = torch.stack((pts3d1_low, pts3d2_low), dim=1)
            pts_all_low = pts_all_low.unsqueeze(-2)
            pts3d_list.append(pts_all_low)
            
            
            gaussians_low = torch.stack([GS_res1_low, GS_res2_low], dim=1)
            gaussians_low = rearrange(gaussians_low, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            gaussians_low[..., 0] = gaussians_low[..., 0].sigmoid()
            gaussians_list.append(gaussians_low)
            
            instances_low = torch.stack([instance_res_1_low, instance_res_2_low], dim=1)
            instances_low = rearrange(instances_low, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            instances_low = F.normalize(instances_low, p=2, dim=-1)
            gaussians_list[1] = torch.cat([gaussians_list[1][..., 0:1], instances_low, gaussians_list[1][..., 1:]], dim=-1)
            
            languages = torch.stack([language_1, language_2], dim=1)
            languages = rearrange(languages, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            languages_list.append(languages.unsqueeze(-2))

        # not need depth
        depths = pts_all[..., -1].unsqueeze(-1)
        
        # Convert the features and depths into Gaussians.
        if self.pose_free:
            gaussians_list = [self.gaussian_adapter.forward(
                pts.unsqueeze(-2),
                depths,
                gaussians[..., 0].unsqueeze(-1),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                language=languages,
            ) for pts, gaussians, languages in zip(pts3d_list ,gaussians_list, languages_list)]
        else:
            raise NotImplementedError(f"unexpected {self.pose_free=}")

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians_list[0].scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians_list[0].rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians_list[0].means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians_list[0].opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )

        return [Gaussians(
            rearrange(
                gaussians.means.float(),
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances.float(),
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics.float(),
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities.float(),
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(
                gaussians.active.float(),
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(
                gaussians.instance_feats.float(),
                "b v r srf spp c -> b (v r srf spp) c",
            ) if gaussians.instance_feats is not None else None,
            rearrange(
                gaussians.language_feats.float(),
                "b v r srf spp c -> b (v r srf spp) c",
            ) if gaussians.language_feats is not None else None,
        ) for gaussians in gaussians_list]

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim