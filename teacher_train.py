import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_mn10_data, ADDataset
from model import Model, Decoder
import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import chamfer_loss
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
    
    if args.data:
        mn10_train_data = np.load(args.data + '/train_point_clouds.npy')
        mn10_val_data = np.load(args.data + '/val_point_clouds.npy')
        
        # visualize some point clouds using open3d
        # for i in range(5):
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(mn10_train_data[i])
        #     o3d.visualization.draw_geometries([pcd])
    else:
        mn10_train_data, mn10_val_data = get_mn10_data(n_points=n_points, n_train=500, n_val=25, save=True)
    
    # exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize ModelNet10 dataset with normalized point clouds
    mn10_train_dataset = ADDataset(mn10_train_data, k=k, normalize=True)
    mn10_val_dataset = ADDataset(mn10_val_data, k=k, normalize=True)
    
    mn10_train_loader = DataLoader(mn10_train_dataset, batch_size=1, shuffle=True)
    mn10_val_loader = DataLoader(mn10_val_dataset, batch_size=1, shuffle=False)
    
    # Initialize the teacher and decoder models
    teacher = Model(k=k).to(device)
    decoder = Decoder().to(device)
    if args.model:
        teacher.load_state_dict(torch.load(args.model + '/teacher.pth', map_location=device))
        decoder.load_state_dict(torch.load(args.model + '/decoder.pth', map_location=device))
        
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-6)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-6)
    
    
    n_teacher_epochs = 250
    
    # Teacher-decoder loss function - minimize Chamfer distance to train D
    # Q = 16 randomly sampled points from input point cloud
    # Chamfer(D(f_p), Rbar(p)) for each p in Q
    
    teacher.train()
    print("Training teacher model")
    teacher_train_losses = []
    teacher_val_losses = []
    best_val_loss = float('inf')
    for epoch in tqdm(range(n_teacher_epochs)):
        total_loss = 0.0
        for i, (points, nearest_neighbors) in enumerate(mn10_train_loader):
            # features = torch.zeros(64000, 64) # n points, dimension 64; TODO: should I reinitialize for every point cloud?
            teacher_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            features = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
            
            indices = torch.randint(0, features.shape[1], (16,))
            output = decoder(features[:, indices, :]).detach()
            
            loss = chamfer_loss(output, points.to(device), indices, nearest_neighbors, device)
            # loss = nn.MSELoss()(output, points.to(device))
            loss.backward()
            total_loss += loss.item()
            teacher_optimizer.step()
            decoder_optimizer.step()
            if i % 50 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
            
        # print(f"Epoch {epoch} time: {time.perf_counter() - batch_loop}. Loss: {total_loss}")
        teacher_train_losses.append(loss.item())
        
        # Validation
        with torch.no_grad():
            val_loss = 0.0
            for points, nearest_neighbors in mn10_val_loader:
                features = teacher(points.to(device), torch.zeros(points.shape[0], n_points, 64).to(device), nearest_neighbors.to(device))
                indices = torch.randint(0, features.shape[1], (16,))
                output = decoder(features[:, indices, :])
                loss = chamfer_loss(output, points.to(device), indices, nearest_neighbors, device)
                val_loss += loss.item()
                
            print(f"Epoch {epoch}. Validation Loss: {val_loss}")
            teacher_val_losses.append(val_loss)
            
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model at epoch {epoch} with train loss {total_loss} and val loss {val_loss}")
            torch.save(teacher.state_dict(), 'models/teacher_best.pth')
            torch.save(decoder.state_dict(), 'models/decoder_best.pth')
            
    # plot training loss curve
    plt.plot(teacher_train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Teacher Model Training Loss")
    # plt.show()
    plt.savefig('teacher_loss_curve.png')
    
    # plot validation loss curve
    plt.plot(teacher_val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Teacher Model Validation Loss")
    # plt.show()
    plt.savefig('teacher_val_loss_curve.png')
                
    torch.save(teacher.state_dict(), 'models/teacher_final.pth')
    torch.save(decoder.state_dict(), 'models/decoder_final.pth')
