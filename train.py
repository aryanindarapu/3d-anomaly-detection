import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import generate_mn10_data, generate_mvtec_data, ADDataset
from model import Model, Decoder
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse



class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, pred, gt):
        batch_size, num_points, _ = pred.shape
        expanded_pred = pred.unsqueeze(1).expand(batch_size, num_points, num_points, 3)
        expanded_gt = gt.unsqueeze(2).expand(batch_size, num_points, num_points, 3)

        dist = torch.norm(expanded_pred - expanded_gt, dim=3, p=2)
        dist1, _ = torch.min(dist, dim=2)
        dist2, _ = torch.min(dist, dim=1)

        chamfer_loss = (dist1.mean(dim=1) + dist2.mean(dim=1)).mean()
        return chamfer_loss

def compute_receptive_fields(data, p, k):
    # TODO: current taking 2 hops. Should I take 2 * 4 = 8 hops (equal to num executed LFA blocks)?
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors([p])
    N1 = indices.flatten()
    N2 = set()
    for idx in N1:
        _, hop_indices = nbrs.kneighbors([data[idx]])
        N2.update(hop_indices.flatten())
    
    all_points = set(N1).union(N2)
    
    return data[list(all_points)]

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

# def compute_receptive_field(p, k):
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(p)
#     _, indices = nbrs.kneighbors(p)
#     receptive_field = p[indices]
#     return receptive_field
    
import open3d as o3d    

if __name__ == "__main__":
    # train_data, val_data = generate_data(n_points=64000, n_train=500, n_val=25, save=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default=None, help="Path to data files")
    args = parser.parse_args()
    
    if args.data:
        mn10_train_data = np.load(args.data + '/train_point_clouds.npy')
        mn10_val_data = np.load(args.data + '/val_point_clouds.npy')
        
        # visualize some point clouds using open3d
        # for i in range(5):
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(mn10_train_data[i])
        #     o3d.visualization.draw_geometries([pcd])
                
        mvtec_train_data = np.load(args.data + '/mvtec_train_point_clouds.npy')
        mvtec_val_data = np.load(args.data + '/mvtec_val_point_clouds.npy')
    else:
        # mn10_train_data, mn10_val_data = generate_mn10_data(n_points=64000, n_train=5, n_val=5, save=True)
        mvtec_train_data, mvtec_val_data = generate_mvtec_data(n_points=64000, save=True)
        
        
    mn10_train_dataset = ADDataset(mn10_train_data, k=32)
    mn10_val_dataset = ADDataset(mn10_val_data, k=32)
    
    mn10_train_loader = DataLoader(mn10_train_dataset, batch_size=1, shuffle=True)
    mn10_val_loader = DataLoader(mn10_val_dataset, batch_size=1, shuffle=False)
    
    teacher = Model()
    decoder = Decoder()
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-6)
    
    n_teacher_epochs = 250
    
    # Teacher-decoder loss function - minimize Chamfer distance to train D
    # Q = 16 randomly sampled points from input point cloud
    # Chamfer(D(f_p), Rbar(p)) for each p in Q
    
    teacher.train()
    print("Training teacher model")
    for epoch in range(n_teacher_epochs):
        for i, (points, nearest_neighbors) in enumerate(mn10_train_loader):
            # features = torch.zeros(64000, 64) # n points, dimension 64; TODO: should I reinitialize for every point cloud?
            teacher_optimizer.zero_grad()
            features = teacher(points, torch.zeros(64000, 64), nearest_neighbors)
            
            indices = np.random.choice(features.shape[0], 16)
            output = features[indices, :]
            
            loss = chamfer_loss(output, indices)
            loss.backward()
            teacher_optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
                
    torch.save(teacher.state_dict(), 'models/teacher.pth')
    torch.save(decoder.state_dict(), 'models/decoder.pth')
    
    mvtec_train_dataset = ADDataset(mvtec_train_data, k=32)
    mvtec_val_dataset = ADDataset(mvtec_val_data, k=32)
    
    mvtec_train_loader = DataLoader(mvtec_train_dataset, batch_size=1, shuffle=True)
    mvtec_val_loader = DataLoader(mvtec_val_dataset, batch_size=1, shuffle=False)
    
    student = Model()
    student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)
    
    n_student_epochs = 100
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    criterion = nn.MSELoss()
    student.train()
    print("Training student model")
    for epoch in range(n_student_epochs):
        for i, (points, nearest_neighbors) in enumerate(mvtec_train_loader):
            # f_S = torch.zeros(64000, 64) # TODO: reinit the student model?
            student_optimizer.zero_grad()
            f_S = student(points, torch.zeros(64000, 64), nearest_neighbors)
            f_T = teacher(points, torch.zeros(64000, 64), nearest_neighbors)
            
            # transform f_T to be centered around 0 with unit variance
            f_T = f_T - torch.mean(f_T, axis=0)
            f_T = f_T / torch.std(f_T, axis=0)
            
            loss = criterion(f_S, f_T)
            loss.backward()            
            student_optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
                
                
    torch.save(student.state_dict(), 'models/student.pth')
    