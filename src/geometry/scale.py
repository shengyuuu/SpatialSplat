import numpy as np
from jaxtyping import Float
import cv2
import torch

def compute_projection_matrix(object_points, image_points):
    num_points = object_points.shape[0]
    A = []

    for i in range(num_points):
        X, Y, Z = object_points[i]
        u, v = image_points[i]
        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    return P

def resi_calcu(P: Float[np.ndarray, "3 4"],
               object_points: Float[np.ndarray, "n 3"],
               image_points: Float[np.ndarray, "n 2"]
               ) -> Float[np.ndarray, "n"]:
    proj_pts_3d = np.dot(P, np.hstack([object_points, 
                                    np.ones((object_points.shape[0], 1))]).T)
    proj_pts_3d /= proj_pts_3d[2]
    proj_pts_2d = proj_pts_3d[:2].T
    
    return np.linalg.norm(image_points - proj_pts_2d, axis=-1)

def solveDLTransac(object_points: Float[np.ndarray, "n 3"],
                   image_points: Float[np.ndarray, "n 2"],
                   iteration = 1e3, 
                   eps = 1,
                   min_num_points = 6,
                   in_percent = 0.9) -> Float[np.ndarray, "3 4"]:
    num_points = object_points.shape[0]
    
    best_P = None
    last_percent = 0
    for _ in range(iteration):
        
        # Randomly select min_num_points points
        idx = np.random.choice(num_points, min_num_points, replace=False)
        
        P = compute_projection_matrix(object_points[idx], image_points[idx])
        
        residual = resi_calcu(P, object_points, image_points)
        
        num_in_liner = np.sum(residual < eps)
        
        percent = num_in_liner / num_points
        
        if percent > last_percent:
            last_percent = percent
            best_P = P
        
            if percent > in_percent:
                break        
    
    return best_P

def solveDLTransac_batch(
    object_points: Float[np.ndarray, "batch n 3"],
    image_points: Float[np.ndarray, "batch n 2"],
    intrinsic_matrix: Float[np.ndarray, "batch 3 3"],
    img_size,
    iteration = 1e3,
    eps = 1,
    min_num_points = 6,
    in_percent = 0.9
) -> Float[np.ndarray, "batch 3 4"]:
    
    return np.array([
        solveDLTransac(object_points[i], 
                       image_points[i], 
                       iteration, 
                       eps, 
                       min_num_points, 
                       in_percent)
                     for i in range(object_points.shape[0])])
    
def solvePnPransac(object_points: Float[np.ndarray, "n 3"],
                     image_points: Float[np.ndarray, "n 2"],
                     camera_matrix: Float[np.ndarray, "3 3"],
                     iteration = 1e3, 
                     eps = 1,
                     min_num_points = 6,
                     in_percent = 0.9) -> Float[np.ndarray, "3 4"]:
    num_points = object_points.shape[0]
    
    best_T = None
    last_percent = 0
    for _ in range(int(iteration)):  
        # Randomly select min_num_points points
        idx = np.random.choice(num_points, min_num_points, replace=False)
        # solve PnP    
        retval, rvec, tvec= cv2.solvePnP(object_points[idx], 
                                            image_points[idx], 
                                            camera_matrix, 
                                            None)
        
        R, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack([R, tvec])
        P = np.dot(camera_matrix, extrinsic_matrix)
        
        residual = resi_calcu(P, object_points, image_points)
        num_in_liner = np.sum(residual < eps)
        percent = num_in_liner / num_points
        # update best_T
        if percent > last_percent:
            last_percent = percent
            best_T = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])
            if percent > in_percent:
                break 
    
    if last_percent < 0.6:
        best_T = None       
    # print(last_percent)
    return best_T
    
def solvePnPransac_batch(object_points: Float[np.ndarray, "batch n 3"],
    image_points: Float[np.ndarray, "batch n 2"],
    intrinsic_matrix: Float[np.ndarray, "batch 3 3"],
    extrinsic_matrix: Float[np.ndarray, "batch 4 4"],
    img_size,
    device=None,
    iteration = 1e3,
    eps = 1,
    min_num_points = 6,
    in_percent = 0.9
) -> Float[torch.Tensor, "batch"]:
    assert object_points.shape[1] == image_points.shape[1] == img_size[0] * img_size[1], \
        "The number of points should be equal to the image size"
    
    # denorm intrinsic_matrix
    intrinsic_matrix[:, 0] *= img_size[1]
    intrinsic_matrix[:, 1] *= img_size[0]
    
    extrinsics = [
        solvePnPransac(object_points[i], 
                       image_points[i], 
                       intrinsic_matrix[i], 
                       iteration, 
                       eps, 
                       min_num_points, 
                       in_percent)
                     for i in range(object_points.shape[0])
    ]
    
    for i, ext in enumerate(extrinsics):
        if ext is None:
            extrinsics[i] = extrinsic_matrix[i]
        else:
            extrinsics[i] = np.linalg.inv(ext)
    extrinsics = np.array(extrinsics)
    # import ipdb; ipdb.set_trace()
    scale = np.linalg.norm(extrinsics[:, :3, 3], axis=-1) / np.linalg.norm(extrinsic_matrix[:, :3, 3], axis=-1)
    
    return torch.from_numpy(scale).to(device)