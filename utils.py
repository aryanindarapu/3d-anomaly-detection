import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np

# https://arxiv.org/pdf/2307.03043#:~:text=For%20any%20two%20point%20sets,the%20Euclidean%20or%20Manhattan%20distance).
class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, pred, gt):
        batch_size, num_points, _ = pred.shape
        expanded_pred = pred.unsqueeze(1).expand(batch_size, num_points, num_points, 3)
        expanded_gt = gt.unsqueeze(2).expand(batch_size, num_points, num_points, 3)

        # this is using L1 norm # TODO: what would happen if I use L2 norm?
        dist = torch.norm(expanded_pred - expanded_gt, dim=3, p=2)
        dist1, _ = torch.min(dist, dim=2)
        dist2, _ = torch.min(dist, dim=1)

        chamfer_loss = (dist1.mean(dim=1) + dist2.mean(dim=1)).mean()
        return chamfer_loss

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

def compute_receptive_fields(data, p, k, L=4):
    hops = 4 * L
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    
    distances, indices = nbrs.kneighbors([p])
    current_points = set(indices.flatten())
    
    for _ in range(hops):  # already took the first hop
        next_points = set()
        for idx in current_points:
            _, hop_indices = nbrs.kneighbors([data[idx]])
            next_points.update(hop_indices.flatten())
        
        current_points.update(next_points)
    
    return data[list(current_points)]


def chamfer_loss(D_f_p, points, point_indices):
    chamfer_dist = ChamferDistance()
    # Q = points[np.random.choice(points.shape[0], 16, replace=False)]
    total_loss = 0.0
    for p_idx in point_indices:
        R_p = compute_receptive_fields(points, points[p_idx], k=32)
        R_bar_p = R_p - torch.mean(R_p, axis=0)
        loss = chamfer_dist(D_f_p, R_bar_p)
        total_loss += loss
        
    return total_loss / 16


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
        subpath_base = os.path.join(path, 'train/good/xyz')
    elif split == 'val':
        subpath_base = os.path.join(path, 'validation/good/xyz')
    else:
        raise ValueError("Invalid split. Choose 'train' or 'val'.")
                
    f_paths = []
    for obj in os.listdir('data/mvtec_3d_anomaly_detection'):
        path = os.path.join('data/mvtec_3d_anomaly_detection', obj)
        if os.path.isdir(path):
            for file in os.listdir(subpath_base):
                if file.endswith('.tiff'):
                    f_paths.append(os.path.join(subpath_base, file))
                    
    return f_paths
