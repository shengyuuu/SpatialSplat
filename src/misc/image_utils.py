import torch
from torch import Tensor
from PIL import Image
from jaxtyping import Float
import numpy as np
# import rearagen
from einops import rearrange
from torch.nn import functional as F

def normalize_image(tensor:Float[torch.Tensor, 'v c h w'], 
                    mean=(0.5, 0.5, 0.5), 
                    std=(0.5, 0.5, 0.5)
                    ) -> Float[torch.Tensor, 'v c h w']:
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)[None]
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)[None]
    return (tensor - mean) / std

def denormalize_image(tensor:Float[torch.Tensor, 'v c h w'],
                        mean=(0.5, 0.5, 0.5), 
                        std=(0.5, 0.5, 0.5)
                        ) -> Float[torch.Tensor, 'v c h w']:
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)[None]
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)[None]
        return tensor * std + mean

def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
    mode=None
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS if mode is None else mode)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]
    
    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics

def image_preprocess(image: Float[Tensor, 'v c h w'], 
                     intrinsics: Float[Tensor, 'v 3 3'], 
                     shape: tuple[int, int]
                     ) -> tuple[Float[Tensor, 'v c h w'], Float[Tensor, 'v 3 3']]:
    _,c_in, h_in, w_in = image.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    if c_in == 3:
        image = image.float() / 255. 
        image = F.interpolate(image, size=(h_scaled, w_scaled), mode='bilinear', align_corners=True)
        image, intrinsics = center_crop(image, intrinsics, shape)
        image = normalize_image(image)
    elif c_in == 4:
        rgb = image[:,:3]
        depth = image[:,3:4]
        rgb = rgb.float() / 255.
        rgb = F.interpolate(rgb, size=(h_scaled, w_scaled), mode='bilinear', align_corners=True)
        depth = F.interpolate(depth, size=(h_scaled, w_scaled), mode='nearest')
        image = torch.cat([rgb, depth], dim=1)
        image, intrinsics = center_crop(image, intrinsics, shape)
        normalized_rgb = normalize_image(image[:, :3])
        image = torch.cat([normalized_rgb, image[:, 3:4]], dim=1)
    else:
        raise ValueError(f"Invalid image channel {c_in}, only support 3 or 4 channels")
    
    return image, intrinsics