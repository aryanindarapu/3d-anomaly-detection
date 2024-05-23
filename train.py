import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_mn10_data, get_mvtec_data, ADDataset, custom_collate_fn
from model import Model, Decoder
import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import chamfer_loss, normalized_mse_loss
from tqdm import tqdm

if __name__ == "__main__":
    # train_data, val_data = generate_data(n_points=64000, n_train=500, n_val=25, save=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default=None, help="Path to data files")
    args = parser.parse_args()
    
    n_points = 16000
    k = 8
    
    if args.data:
        mn10_train_data = np.load(args.data + '/train_point_clouds.npy')
        mn10_val_data = np.load(args.data + '/val_point_clouds.npy')
        
        # visualize some point clouds using open3d
        # for i in range(5):
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(mn10_train_data[i])
        #     o3d.visualization.draw_geometries([pcd])
                
        mvtec_train_data = np.load(args.data + '/mvtec_train_point_clouds.npy')
    else:
        mn10_train_data, mn10_val_data = get_mn10_data(n_points=n_points, n_train=500, n_val=25, save=True)
        mvtec_train_data = get_mvtec_data(n_points=n_points, save=True)
    
    # exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "mps"
    
    # Initialize ModelNet10 dataset with normalized point clouds
    mn10_train_dataset = ADDataset(mn10_train_data, k=k, normalize=True)
    mn10_val_dataset = ADDataset(mn10_val_data, k=k, normalize=True)
    
    mn10_train_loader = DataLoader(mn10_train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    mn10_val_loader = DataLoader(mn10_val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize the teacher and decoder models
    teacher = Model(k=k).to(device)
    decoder = Decoder().to(device)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-6)
    
    n_teacher_epochs = 10 # 250
    
    # Teacher-decoder loss function - minimize Chamfer distance to train D
    # Q = 16 randomly sampled points from input point cloud
    # Chamfer(D(f_p), Rbar(p)) for each p in Q
    
    teacher.train()
    print("Training teacher model")
    teacher_train_losses = []
    teacher_val_losses = []
    for epoch in tqdm(range(n_teacher_epochs)):
        total_loss = 0.0
        for i, (points, nearest_neighbors, nnbrs) in enumerate(mn10_train_loader):
            # features = torch.zeros(64000, 64) # n points, dimension 64; TODO: should I reinitialize for every point cloud?
            teacher_optimizer.zero_grad()
            features = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
            
            # indices = np.random.choice(features.shape[1], 16)
            indices = torch.randint(0, features.shape[1], (16,))
            output = decoder(features[:, indices, :])
            
            loss = chamfer_loss(output, points.to(device), indices, k, nnbrs, device)
            loss.backward()
            total_loss += loss.item()
            teacher_optimizer.step()            
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
        teacher_train_losses.append(loss.item())
        
        # Validation
        with torch.no_grad():
            val_loss = 0.0
            for points, nearest_neighbors, nnbrs in mn10_val_loader:
                features = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
                indices = torch.randint(0, features.shape[1], (16,))
                output = decoder(features[:, indices, :])
                loss = chamfer_loss(output, points.to(device), indices, k, nnbrs, device)
                val_loss += loss.item()
                
            teacher_val_losses.append(val_loss)
            
            
    # plot loss curve
    plt.plot(teacher_train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Teacher Model Loss")
    plt.show()
    
    plt.plot(teacher_val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Teacher Model Validation Loss")
    plt.show()
                
    torch.save(teacher.state_dict(), 'models/teacher.pth')
    torch.save(decoder.state_dict(), 'models/decoder.pth')
    
    mvtec_train_dataset = ADDataset(mvtec_train_data, k=k)
    
    mvtec_train_loader = DataLoader(mvtec_train_dataset, batch_size=1, shuffle=True)
    
    # Initialize student model with random weights    
    student = Model(k=k, is_student=True).to(device)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)
    
    n_student_epochs = 5 # 100
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # criterion = nn.MSELoss()
    student.train()
    print("Training student model")
    student_losses = []
    for epoch in tqdm(range(n_student_epochs)):
        loss = 0.0
        for i, (points, nearest_neighbors) in enumerate(mvtec_train_loader):
            # f_S = torch.zeros(64000, 64) # TODO: reinit the student model?
            student_optimizer.zero_grad()
            with torch.no_grad():
                f_T = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
            
            f_S = student(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
            
            loss += normalized_mse_loss(f_S, f_T)
            # loss = criterion(f_S, f_T)
            loss.backward()            
            student_optimizer.step()
            
        student_losses.append(loss.item())
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
            
            
    # plot loss curve
    plt.plot(student_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Student Model Loss")
    plt.show()

    torch.save(student.state_dict(), 'models/student.pth')
