from dataclasses import dataclass
from typing import Literal

from sympy import use
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, pose_refine
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color= [0.0, 0.0, 0.0]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    # force to use float32
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        scales: Float[Tensor, "batch"] | None = None,
        scale_invariant: bool = False,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        modify: bool = False,
        use_sh: bool = True,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        if scales is not None:
            assert not modify, "Scales should not be modified"
            extrinsics[:, :, :3, 3] *= scales[...,None, None]
        if cam_rot_delta is not None:
            color, depth = pose_refine(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=scale_invariant,
            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
        )
            language_image = None
            instance_image = None
        elif modify:
            color, depth, raddi, gaussian_counter = render_cuda_modify(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            )
            language_image = None
            instance_image = None
        else:
            
            color, language_image, instance_image = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                instance_feats=repeat(gaussians.instance_feats, "b g c_instance -> (b v) g c_instance", v=v) if gaussians.instance_feats is not None else None,
                # active=repeat(gaussians.active, "b g -> (b v) g", v=v) if gaussians.active is not None else None,
                scale_invariant=scale_invariant,  
                use_sh=use_sh,  
            )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        if instance_image is not None:
            instance_image = rearrange(instance_image, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput(
            color,
            None,
            instance_image,
            language_image
        )