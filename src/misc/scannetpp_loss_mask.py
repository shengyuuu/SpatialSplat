import einops
import torch
from jaxtyping import Float, Bool
from .geometry import unproject_depth, world_space_to_camera_space, camera_space_to_pixel_space


@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    # import ipdb; ipdb.set_trace()
    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    # Unproject the depth to get the 3D points in world space
    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

    # Project the 3D points into the pixel space of all the second views simultaneously
    camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

    # Calculate the depth of each point
    rendered_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

    # We use three conditions to determine if a point should be masked

    # Condition 1: Check if the points are in the frustum of any of the v2 views
    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )  # (b, v1, v2, h, w)
    in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

    # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
    non_zero_depth = depth_1 > 1e-6

    # Condition 3: Check if the points have matching depth to any of the v2
    # views torch.nn.functional.grid_sample expects the input coordinates to
    # be normalized to the range [-1, 1], so we normalize first
    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
    for b in range(depth_1.shape[0]):
        for i in range(v1):
            for j in range(v2):
                depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j].float(), sampled_depths.float(), atol=1e-1)

    matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)
    del points_3d, camera_points, points_2d, rendered_depth
    mask = in_frustum_mask & non_zero_depth & matching_depth
    return mask #[b, v1, h, w]


@torch.no_grad()
def calculate_loss_mask(batch) -> Bool[torch.Tensor, "b v h w"]:
    '''Calcuate the loss mask for the target views in the batch'''
    h,w = batch["target"]["depthmap"].shape[-2:]
    target_depth = batch["target"]["depthmap"] # (b, v, h, w)
    target_intrinsics = batch["target"]["intrinsics"] # (b, v, 3, 3)
    target_c2w = batch["target"]["extrinsics"] # (b, v, 4, 4)
    context_depth = batch["context"]["depthmap"] # (b, v, h, w)
    context_intrinsics = batch["context"]["intrinsics"]
    context_c2w = batch["context"]["extrinsics"]
    
    target_intrinsics[:,:,0,:] *= w
    target_intrinsics[:,:,1,:] *= h
    context_intrinsics[:,:,0,:] *= w
    context_intrinsics[:,:,1,:] *= h
    # import ipdb; ipdb.set_trace()

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    # import ipdb; ipdb.set_trace()
    return mask
