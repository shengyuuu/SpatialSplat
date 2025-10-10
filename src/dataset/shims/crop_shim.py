import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from typing import Union
import torch.nn.functional as F

from ..types import AnyExample, AnyViews


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
    masks: Float[Tensor, "*#batch 1 h w"] = None,
    xyz: Float[Tensor, "*#batch 3 h w"] = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Union[Float[Tensor, "*#batch 1 h_out w_out"],
          None],  # updated masks
    Union[Float[Tensor, "*#batch 3 h_out w_out"],
            None],  # updated xyz
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]
    
    
    # Center-crop the mask.
    if masks is not None:
        masks = masks[..., :, row : row + h_out, col : col + w_out]
    else:
        masks = torch.zeros_like(images[:, :1, :, :])
        
    # Center-crop the depth_map.
    if xyz is not None:
        xyz = xyz[..., row : row + h_out, col : col + w_out]
    else:
        xyz = torch.zeros_like(images)

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, masks, xyz, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    masks: Float[Tensor, "*#batch 1 h w"] = None,
    xyz: Float[Tensor, "*#batch 3 h w"] = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 1 h_out w_out"],  # updated masks
    Float[Tensor, "*#batch 3 h w"],  # updated xyz
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)
    
    # Reshape the masks to the correct size.
    if masks is not None:
        *batch, _, h, w = masks.shape
        masks = masks.repeat(1, 3, 1, 1)
        masks = masks.reshape(-1, 3, h, w)
        masks = torch.stack([rescale(mask, (h_scaled, w_scaled),mode=Image.NEAREST) for mask in masks])
        masks = masks.reshape(*batch, 3, h_scaled, w_scaled)[:,:1,:,:]
        
    # Reshape the xyz to the correct size.
    if xyz is not None:
        *batch, _, h, w = xyz.shape
        xyz = F.interpolate(xyz, (h_scaled, w_scaled), mode='nearest')

    return center_crop(images, intrinsics, shape, masks, xyz)

def refine_masks(masks: Float[Tensor, "*#batch 1 h w"]):
    masks = (masks * 255).int()
    splited_masks_batch = []
    for mask in masks:
        splited_mask = []
        for id in range(mask.max() + 1):
            mask_id = (mask == id)
            if id == 0:
                splited_mask.append(mask_id)
                continue
            if torch.nonzero(mask_id).shape[0] < 16*16:
                continue
            splited_mask.append(mask_id)
        splited_masks_batch.append(splited_mask)
    return splited_masks_batch


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    if "xyz" not in views.keys():
        images, masks, xyz, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], 
                                                             shape, views["mask_image"])
    else:
        images, masks, xyz, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], 
                                                             shape, views["mask_image"], views["xyz"])
    # masks = refine_masks(masks)
    return {
        **views,
        "image": images,
        "mask_image": masks,
        "xyz": xyz,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
