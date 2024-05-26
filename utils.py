import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
import open3d as o3d

# https://arxiv.org/pdf/2307.03043#:~:text=For%20any%20two%20point%20sets,the%20Euclidean%20or%20Manhattan%20distance).
class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()
        
    def forward(self, x, y):
        x = x.unsqueeze(1)  # (B, 1, M, 3)
        y = y.unsqueeze(2)  # (B, N, 1, 3)
        
        # Calculate pairwise distances
        # dist = torch.sum((x - y) ** 2, -1)  # (B, N, M)
        dist = torch.norm(x - y, dim=-1)  # (B, N, M)
        
        # Minimum distance from each point in y to x and from x to y
        dist_y_to_x = torch.min(dist, dim=2)[0]  # (B, N)
        dist_x_to_y = torch.min(dist, dim=1)[0]  # (B, M)
        
        # Mean distance
        loss = torch.mean(dist_y_to_x) + torch.mean(dist_x_to_y)
        
        return loss
    
def compute_receptive_fields(data, p, nnbrs, L=4):
    hops = 2 * L
    
    knn_i = set([p.cpu().numpy().tolist()])
    knn_l = set(knn_i) # union of all knn_i
    for _ in range(hops):
        hop_indices = nnbrs[list(knn_i)].flatten().cpu().numpy()
        knn_l.update(hop_indices)
        knn_i = np.unique(hop_indices)
    
    return data[list(knn_l)]


def chamfer_loss(D_f_p, points, point_indices, nnbrs, device):
    # D_f_p: (B, M, 3)
    # points: (B, N, 3)
    # point_indices: (16,)
    chamfer_dist = ChamferDistance().to(device)
    total_loss = 0.0
    for i, P in enumerate(points):
        # for p_idx in point_indices:
        #     # R_p = compute_receptive_fields(P, P[p_idx], nnbrs[i])
        #     R_p = compute_receptive_fields(P, p_idx, nnbrs[i])
        #     R_bar_p = R_p - torch.mean(R_p, axis=0)
        #     loss = chamfer_dist(D_f_p[i].unsqueeze(0).to(device), R_bar_p.unsqueeze(0).to(device))
        #     total_loss += loss
        
        receptive_fields_list = []
        for p_idx in point_indices:
            R_p = compute_receptive_fields(P, p_idx, nnbrs[i])
            receptive_fields_list.append(R_p)
            
            # visualize the receptive fields on the ground truth point cloud
            # display P in blue and R_p in red
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(P.cpu().numpy())
            # pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * P.shape[0]))
            
            # pcd_r = o3d.geometry.PointCloud()
            # pcd_r.points = o3d.utility.Vector3dVector(R_p.cpu().numpy())
            # pcd_r.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * R_p.shape[0]))
            
            # o3d.visualization.draw_geometries([pcd, pcd_r])

        # Compute the Chamfer distance for each receptive field
        for R_p in receptive_fields_list:
            R_bar_p = R_p - torch.mean(R_p, axis=0)
            loss = chamfer_dist(D_f_p[i].unsqueeze(0).to(device), R_bar_p.unsqueeze(0).to(device))
            total_loss += loss
        
    return total_loss / (16 * len(points))
    
    
def get_teacher_train_features_distr(teacher, points, nearest_neighbors):
    teacher.eval()
    with torch.no_grad():
        features = teacher(points, torch.zeros(points.shape[0], 16000, 64), nearest_neighbors) # (B, N, 64)
        
    mean = torch.mean(features, dim=(0, 1))
    std = torch.std(features, dim=(0, 1))
    
    return mean.cpu(), std.cpu()


def normalized_mse_loss(f_S, f_T, mean, std):
    # transform f_T to be centered around 0 with unit variance
    f_T = f_T - mean
    f_T = f_T / std
    score = nn.functional.mse_loss(f_S, f_T)
    
    return score


def get_anomaly_scores(f_S, f_T, mean, std):
    # compute unit-wise normalized L2 distance
    f_T = f_T - mean
    f_T = f_T / std
    scores = torch.norm(f_S - f_T, dim=-1)
    
    return scores
    

def get_mvtec_filepaths(split='train'):    
    if split == 'train':
        subpath_base = 'train/good/xyz'
    elif split == 'val':
        subpath_base = 'validation/good/xyz'
    else:
        raise ValueError("Invalid split. Choose 'train' or 'val'.")
                
    f_paths = []
    for obj in os.listdir('data/mvtec_3d_anomaly_detection'):
        path = os.path.join('data/mvtec_3d_anomaly_detection', obj)
        if os.path.isdir(path):
            for file in os.listdir(os.path.join(path, subpath_base)):
                if file.endswith('.tiff'):
                    f_paths.append(os.path.join(path, subpath_base, file))
                    
        break
                    
    return f_paths

#### NOTE: This function is deprecated, use open3d instead
def farthest_point_sampling(point_cloud, num_samples):
    points = np.asarray(point_cloud.points)
    num_points = points.shape[0]

    if num_samples >= num_points:
        return point_cloud

    sampled_indices = [np.random.randint(num_points)]
    distances = np.linalg.norm(points - points[sampled_indices[0]], axis=1)

    for _ in range(num_samples - 1):
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)
        distances = np.minimum(distances, np.linalg.norm(points - points[next_index], axis=1))

    sampled_points = points[sampled_indices]
    sampled_point_cloud = o3d.geometry.PointCloud()
    sampled_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

    return sampled_point_cloud
