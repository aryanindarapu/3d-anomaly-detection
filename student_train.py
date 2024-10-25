import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_mvtec_data, ADDataset
from model import Model, Decoder
import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import chamfer_loss, normalized_mse_loss, get_teacher_train_features_distr
from tqdm import tqdm
import open3d as o3d

if __name__ == "__main__":
    # train_data, val_data = generate_data(n_points=64000, n_train=500, n_val=25, save=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default=None, help="Path to data files")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to models")
    args = parser.parse_args()
    
    n_points = 16000
    k = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.data:       
        mvtec_train_data = np.load(args.data + '/mvtec_train_pcdata_single.npy')
        mvtec_val_data = np.load(args.data + '/mvtec_val_pcdata_single.npy')
        print(mvtec_train_data.shape, mvtec_val_data.shape)
        # visualize some point clouds using open3d
        # for i in range(5):
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(mvtec_val_data[i])
        #     o3d.visualization.draw_geometries([pcd])
    else:
        # mvtec_train_data = get_mvtec_data(n_samples=10, n_points=n_points, save=True, split='train')
        mvtec_train_data = get_mvtec_data(n_samples=None, n_points=n_points, save=True, split='train')
        # mvtec_val_data = get_mvtec_data(n_samples=50, n_points=n_points, save=True, split='val')
        mvtec_val_data = get_mvtec_data(n_samples=None, n_points=n_points, save=True, split='val')
    
    # exit()
    if args.model:
        teacher = Model(d=32, k=k, R=3).to(device)
        teacher.load_state_dict(torch.load(args.model + '/teacher_best.pth', map_location=device))
    else:
        raise ValueError("Teacher model not found")


    mvtec_train_dataset = ADDataset(mvtec_train_data, k=k)
    mvtec_train_loader = DataLoader(mvtec_train_dataset, batch_size=1, shuffle=True)
    
    # Initialize student model with random weights    
    student = Model(d=32, k=k, R=3, is_student=True).to(device)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)
    
    n_student_epochs = 100
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    mvtec_train_dataset = ADDataset(mvtec_train_data, k=k)
    mvtec_train_tmp = DataLoader(mvtec_train_dataset, batch_size=len(mvtec_train_dataset), shuffle=False)
    with torch.no_grad():
        all_features = []
        # all_mean = []
        # all_std = []
        # teacher.eval()
        for i, (points, nearest_neighbors) in enumerate(mvtec_train_tmp):
            features = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 32).to(device), nearest_neighbors.to(device)) # (B, N, 64)
            all_features.append(features)
            # mean, std = get_teacher_train_features_distr(teacher, points, nearest_neighbors, d, device)
            # all_mean.append(mean)
            # all_std.append(std)
        
        # mean = torch.mean(torch.stack(all_mean), dim=0)
        # std = torch.mean(torch.stack(all_std), dim=0)
        all_features = torch.cat(all_features, dim=0)
        mean = torch.mean(all_features, dim=(0, 1))
        std = torch.std(all_features, dim=(0, 1))
        
        print(f"Mean Shape: {mean.shape}, Std Shape: {std.shape}")
                    
        # save the mean and std
        np.save("data/mean_potato.npy", mean.cpu().numpy())
        np.save("data/std_potato.npy", std.cpu().numpy())

    # exit()
    student.train()
    print("Training student model")
    student_train_losses = []
    student_val_losses = []
    best_val_loss = float('inf')
    for epoch in tqdm(range(n_student_epochs)):
        total_loss = 0.0
        for i, (points, nearest_neighbors, nnbrs) in enumerate(mvtec_train_loader):
            # f_S = torch.zeros(64000, 64) # TODO: reinit the student model?
            student_optimizer.zero_grad()
            with torch.no_grad():
                f_T = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
            
            f_S = student(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
            
            loss = normalized_mse_loss(f_S, f_T, mean.to(device), std.to(device))
            # loss = criterion(f_S, f_T)
            loss.backward()
            total_loss += loss.item()
            student_optimizer.step()
            
            if i % 50 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

        student_train_losses.append(total_loss)
        
        with torch.no_grad():
            val_loss = 0.0
            for points, nearest_neighbors, nnbrs in mvtec_val_data:
                f_S = student(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
                f_T = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
                val_loss += normalized_mse_loss(f_S, f_T, mean.to(device), std.to(device)).item()
                
            student_val_losses.append(val_loss)
        
        if total_loss < best_val_loss:
            print(f"Saving best student model at epoch {epoch} with loss {loss}")
            torch.save(student.state_dict(), 'models/student_best.pth')
            
            
    # plot training loss curve
    plt.plot(student_train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Student Model Training Loss")
    # plt.show()
    plt.savefig('student_loss.png')
    
    # plot validation loss curve
    plt.plot(student_val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Student Model Validation Loss")

    torch.save(student.state_dict(), 'models/student.pth')
