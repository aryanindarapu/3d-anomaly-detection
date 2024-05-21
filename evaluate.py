import numpy as np
import argparse
from dataset import get_mvtec_test_data, ADDataset
from torch.utils.data import DataLoader
from model import Model
import torch
from utils import get_anomaly_scores
import open3d as o3d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # train_data, val_data = generate_data(n_points=64000, n_train=500, n_val=25, save=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default=None, help="Path to data files")
    args = parser.parse_args()
    
    if args.data:
        mvtec_anomaly_test_data = np.load(args.data + '/mvtec_anomaly_test.npy')
        mvtec_gt_test_data = np.load(args.data + '/mvtec_gt_test.npy')
    else:
        mvtec_gt_test_vals, mvtec_anomaly_test_data = get_mvtec_test_data(n_samples=5)
    
    teacher = Model()
    student = Model()
    teacher.load_state_dict(torch.load('models/teacher.pth'))
    student.load_state_dict(torch.load('models/student.pth'))
    
    test_dataset = ADDataset(mvtec_anomaly_test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    student.eval()
    teacher.eval()

    with torch.no_grad():
        for i, (points, nearest_neighbors) in enumerate(test_loader):
            f_S = student(points, torch.zeros(64000, 64), nearest_neighbors)
            f_T = teacher(points, torch.zeros(64000, 64), nearest_neighbors)
            scores = get_anomaly_scores(f_S, f_T)
            
            ### draw model output geometries (i.e. model detected anomalies) ###
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.squeeze().numpy())
            
            colors = plt.cm.jet(scores_normalized)[:, :3]  # Using jet colormap, discard the alpha channel
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.visualization.draw_geometries([pcd])
            
            ### draw ground truth geometries ###
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(mvtec_anomaly_test_data[i])
            gt_normalized = mvtec_gt_test_vals[i] / 255.0
            pcd_gt.colors = o3d.utility.Vector3dVector(gt_normalized)
            o3d.visualization.draw_geometries([pcd_gt])
        