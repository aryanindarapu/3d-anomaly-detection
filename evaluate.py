import numpy as np
import argparse
from dataset import get_mvtec_test_data, ADDataset, get_mvtec_folder_data
from torch.utils.data import DataLoader
from model import Model
import torch
from utils import get_anomaly_scores, get_teacher_train_features_distr
import open3d as o3d
import matplotlib.pyplot as plt

from config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default=None, help="Path to data files")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to models")
    args = parser.parse_args()
    
    if args.data:
        # mvtec_train_data = np.load(args.data + '/mvtec_train_data.npy')
        mvtec_anomaly_test_data = np.load(args.data + '/mvtec_anomaly_test.npy')
        mvtec_anomaly_test_data_files = np.load(args.data + '/mvtec_anomaly_test_files.npy')
        print(mvtec_anomaly_test_data.shape)
        # mvtec_gt_test_data = np.load(args.data + '/mvtec_gt_test.npy')
    else:
        mvtec_anomaly_test_data, mvtec_anomaly_test_data_files = get_mvtec_folder_data(n_points=n_points)
        # _, mvtec_anomaly_test_data, mvtec_anomaly_test_data_files = get_mvtec_test_data(n_samples=5, n_points=n_points)
        
        
    # print(mvtec_anomaly_test_data_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model:
        teacher = Model(d=d, k=k, R=R).to(device)
        student = Model(d=d, k=k, R=R).to(device)
        teacher.load_state_dict(torch.load(args.model + '/teacher_best.pth', map_location=device))
        student.load_state_dict(torch.load(args.model + '/student_best.pth', map_location=device))
        # student.load_state_dict(torch.load(args.model + '/student_best_working.pth', map_location=device))
        
        mean = torch.tensor(np.load(args.model + '/mean.npy'))
        std = torch.tensor(np.load(args.model + '/std.npy'))
    else:
        raise ValueError("Models not found")
    
    test_dataset = ADDataset(mvtec_anomaly_test_data, k=k, normalize=True)
    # test_dataset = ADDataset(mvtec_anomaly_test_data, k=k, normalize=True, inference=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    student.eval()
    teacher.eval()
    
    # with torch.no_grad():
    #     mvtec_train_dataset = ADDataset(mvtec_train_data, k=k)
    #     mvtec_train_tmp = DataLoader(mvtec_train_dataset, batch_size=len(mvtec_train_dataset), shuffle=False)
    #     for i, (points, nearest_neighbors) in enumerate(mvtec_train_tmp):
    #         mean, std = get_teacher_train_features_distr(teacher, points.to(device), nearest_neighbors.to(device))

    with torch.no_grad():
        for i, (points, nearest_neighbors) in enumerate(test_loader):
            f_S = student(points.to(device), torch.zeros(points.shape[0], n_points, d).to(device), nearest_neighbors.to(device))
            f_T = teacher(points.to(device), torch.zeros(points.shape[0], n_points, d).to(device), nearest_neighbors.to(device))
            scores = get_anomaly_scores(f_S, f_T, mean.to(device), std.to(device)).squeeze().cpu().numpy()
            
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
            # scores_transformed = np.power(scores_normalized, 2)
            # scores_transformed = (scores_transformed - scores_transformed.min()) / (scores_transformed.max() - scores_transformed.min())

            
            ### draw model output geometries (i.e. model detected anomalies) ###            
            # o3d.visualization.draw_geometries([pcd])
            colormap = plt.get_cmap('jet')
            colors = colormap(scores_normalized)[:, :3]  # Get RGB values, ignore alpha
            # colors = np.zeros((scores_normalized.shape[0], 3))
            # colors[scores_normalized > 0.2] = [1, 0, 0]  # Red for scores > 0.2
            # colors[scores_normalized <= 0.2] = [0, 0, 1] 
            
            # Create Open3D point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points.squeeze().cpu().numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # Visualize point 
            print(mvtec_anomaly_test_data_files[i])
            o3d.visualization.draw_geometries([point_cloud])
                
            ### draw ground truth geometries ###
            # pcd_gt = o3d.geometry.PointCloud()
            # pcd_gt.points = o3d.utility.Vector3dVector(mvtec_anomaly_test_data[i])
            # gt_normalized = mvtec_gt_test_vals[i] / 255.0
            # pcd_gt.colors = o3d.utility.Vector3dVector(gt_normalized)
            # o3d.visualization.draw_geometries([pcd_gt])
        