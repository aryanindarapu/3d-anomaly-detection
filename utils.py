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

    # def forward(self, pred, gt):
    #     # batch_size, num_points, _ = pred.shape
    #     num_points = pred.shape[1]
    #     expanded_pred = pred.unsqueeze(1).expand(num_points, num_points, 3)
    #     expanded_gt = gt.unsqueeze(2).expand(num_points, num_points, 3)
    #     # expanded_pred = pred.unsqueeze(1).expand(batch_size, num_points, num_points, 3)
    #     # expanded_gt = gt.unsqueeze(2).expand(batch_size, num_points, num_points, 3)

    #     # this is using L1 norm # TODO: what would happen if I use L2 norm?
    #     dist = torch.norm(expanded_pred - expanded_gt, dim=3, p=2)
    #     dist1, _ = torch.min(dist, dim=2)
    #     dist2, _ = torch.min(dist, dim=1)

    #     chamfer_loss = (dist1.mean(dim=1) + dist2.mean(dim=1)).mean()
    #     return chamfer_loss
    def forward(self, x, y):
        x = x.unsqueeze(1)  # (B, 1, M, 3)
        y = y.unsqueeze(2)  # (B, N, 1, 3)
        
        # Calculate pairwise distances
        dist = torch.sum((x - y) ** 2, -1)  # (B, N, M)
        
        # Minimum distance from each point in y to x and from x to y
        dist_y_to_x = torch.min(dist, dim=2)[0]  # (B, N)
        dist_x_to_y = torch.min(dist, dim=1)[0]  # (B, M)
        
        # Mean distance
        loss = torch.mean(dist_y_to_x) + torch.mean(dist_x_to_y)
        
        return loss

# def compute_receptive_fields(data, p, k):
#     # TODO: current taking 2 hops. Should I take 2 * 4 = 8 hops (equal to num executed LFA blocks)?
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
#     distances, indices = nbrs.kneighbors([p])
#     N1 = indices.flatten()
#     N2 = set()
#     for idx in N1:
#         _, hop_indices = nbrs.kneighbors([data[idx]])
#         N2.update(hop_indices.flatten())
    
#     all_points = set(N1).union(N2)
    
#     return data[list(all_points)]

def compute_receptive_fields(data, p, nnbrs, L=4):
    hops = 4 * L
    # nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    
    distances, indices = nnbrs.kneighbors([p])
    current_points = set(indices.flatten())
    
    for _ in range(hops):
        next_points = set()
        for idx in current_points:
            _, hop_indices = nnbrs.kneighbors([data[idx]])
            next_points.update(hop_indices.flatten())
        
        current_points.update(next_points)
    
    return data[list(current_points)]


def chamfer_loss(D_f_p, points, point_indices, k, nnbrs, device):
    # D_f_p: (B, M, 3)
    # points: (B, N, 3)
    # point_indices: (16,)
    chamfer_dist = ChamferDistance().to(device)
    # Q = points[np.random.choice(points.shape[0], 16, replace=False)]
    total_loss = 0.0
    for i, P in enumerate(points):
        for p_idx in point_indices:
            R_p = compute_receptive_fields(P, P[p_idx], nnbrs[i])
            R_bar_p = R_p - torch.mean(R_p, axis=0)
            loss = chamfer_dist(D_f_p[i].unsqueeze(0).to(device), R_bar_p.unsqueeze(0).to(device))
            total_loss += loss
        
    return total_loss / (16 * len(points))

# def chamfer_loss(D_f_p, points, point_indices, k=32):
#     # D_f_p: (B, M, 3)
#     # points: (B, N, 3)
#     # point_indices: (16,)
    
#     chamfer_dist = ChamferDistance()
#     # Select points at the given indices across all batches
#     selected_points = points[:, point_indices, :]  # (B, 16, 3)
    
#     # Compute receptive fields for the selected points
#     R_p_list = []
#     # for P, indices in zip(points, point_indices):
#     for P in points:
#         R_p = torch.stack([compute_receptive_fields(P, P[idx], k) for idx in point_indices], dim=0)
#         R_p_list.append(R_p)
    
#     # Concatenate receptive fields across batches
#     R_p_all = torch.cat(R_p_list, dim=0)  # (B*16, k, 3)
    
#     # Normalize receptive fields
#     R_bar_p_all = R_p_all - torch.mean(R_p_all, dim=1, keepdim=True)
    
#     # Reshape D_f_p for loss calculation
#     D_f_p_repeated = D_f_p.repeat_interleave(len(point_indices), dim=0)  # (B*16, M, 3)
    
#     # Calculate Chamfer loss
#     loss = chamfer_dist(D_f_p_repeated, R_bar_p_all)
    
#     return loss / (16 * len(points))


def normalized_mse_loss(f_S, f_T):
    # transform f_T to be centered around 0 with unit variance
    f_T = f_T - torch.mean(f_T, axis=0)
    f_T = f_T / torch.std(f_T, axis=0)
    score = nn.functional.mse_loss(f_S, f_T)
    
    return score


def get_anomaly_scores(f_S, f_T):
    # compute unit-wise normalized L2 distance
    f_T = f_T - torch.mean(f_T, axis=0)
    f_T = f_T / torch.std(f_T, axis=0)
    scores = torch.norm(f_S - f_T, dim=1)
    
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
                    
    return f_paths


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
