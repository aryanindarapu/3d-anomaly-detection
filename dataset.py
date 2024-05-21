# generate synthetic 3D scenes using objects of the ModelNet10 dataset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from tifffile import tifffile
import os
import open3d as o3d
from utils import get_mvtec_filepaths
import time

# return 500 training and 25 validation point clouds, choose n=64000 input points from each scene
def get_mn10_data(n_points=64000, n_train=500, n_val=25, save=True):
    df = pd.read_csv('data/metadata_modelnet10.csv')
    # print(df['split'].unique())
    # train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    data = df[df['split'] == 'train']
    path_prefix = 'data/ModelNet10/'

    train_point_clouds = []
    val_point_clouds = []
    # for i in tqdm(range(n_train + n_val)):
    i = 0
    print(f"Generating {n_train} training and {n_val} validation point clouds.")
    
    num_continues = 0
    big_loop_time = time.perf_counter()
    while i < n_train + n_val:
        time_start = time.perf_counter()
        
        if i == n_train:
            data = df[df['split'] == 'test']
        
        random_train = data.sample(n=10)
        # print(random_train.head())
        
        # TODO: do i generate a new scene each time?
        o3d_meshes = []
        # combined_train_pcd = o3d.geometry.PointCloud()
        for file in random_train['object_path']:           
            mesh = o3d.io.read_triangle_mesh(path_prefix + file)
            # if not mesh.has_vertex_normals():
            #     mesh.compute_vertex_normals()
                
            mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())

            # Apply a random rotation
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3) * 2 * np.pi)
            mesh.rotate(rotation, center=mesh.get_center())
            
            translation_vector = np.random.uniform(-3, 3, size=3)
            mesh.translate(translation_vector, relative=False)

            # Convert to point cloud
            pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            pcd.points = mesh.vertices
            
            o3d_meshes.append(pcd)
            # combined_train_pcd += pcd
            # combined_train_pcd += mesh.vertices

        # visualize the meshes
        combined_train_pcd = o3d.geometry.PointCloud()
        for pcd in o3d_meshes:
            combined_train_pcd += pcd
            
        combined_train_pcd.remove_duplicated_points()
            
        # if num points < 64000, retry with a new scene
        if len(combined_train_pcd.points) < n_points:
            continue
        
        # https://medium.com/@sim30217/farthest-point-sampling-43ddedc25628
        combined_train_pcd = combined_train_pcd.farthest_point_down_sample(n_points)
        # farthest_points = farthest_point_sampling(np.asarray(combined_train_pcd.points), n_points)
        
        # axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-3, -3, -3), max_bound=(3, 3, 3))
        # bounding_box.color = (1, 0, 0)  
        # o3d.visualization.draw_geometries([combined_train_pcd, axis_frame, bounding_box])
        
        # o3d.visualization.draw_geometries([combined_train_pcd, axis_frame, bounding_box])
        farthest_points = np.asarray(combined_train_pcd.points)
        
        if i < n_train:
            train_point_clouds.append(farthest_points)
        else:
            val_point_clouds.append(farthest_points)
        
        i += 1
        
        if i % 10 == 0:
            print(f"Generated {i} point clouds.")
            print(f"Time taken for the last point clouds: {time.perf_counter() - time_start:.2f} seconds.")
            print(f"Total time taken: {time.perf_counter() - big_loop_time:.2f} seconds.")
            print()
            

    train_point_clouds = np.array(train_point_clouds)
    val_point_clouds = np.array(val_point_clouds)
    print(train_point_clouds.shape, val_point_clouds.shape)

    if save:
        np.save('data/train_point_clouds.npy', train_point_clouds)
        np.save('data/val_point_clouds.npy', val_point_clouds)
    
    return train_point_clouds, val_point_clouds


def get_mvtec_data(n_points=64000, save=True, split='train'):
    pc_list = []
    c = 0
    for file in tqdm(get_mvtec_filepaths(split)):
        img = tifffile.imread(file).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(img)
        
        # https://medium.com/@sim30217/farthest-point-sampling-43ddedc25628
        new_pcd = pcd.farthest_point_down_sample(n_points)
        # o3d.visualization.draw_geometries([new_pcd])
        farthest_points = np.asarray(new_pcd.points)
        pc_list.append(farthest_points)
        if c == 5:
            break
        
        c += 1
    
    pc_list = np.array(pc_list)
    print(f"{split} Dataset Shape:", pc_list.shape)
    
    if save:
        np.save(f'data/mvtec_{split}_point_clouds.npy', pc_list)
        
    return pc_list


def get_mvtec_test_data(n_samples=10, n_points=64000):
    gt_data, pcds = [], []
    
    anomaly_labels = ['combined', 'contamination', 'crack', 'cut', 'hole']
    i = 0
    for obj in os.listdir('data/mvtec_3d_anomaly_detection'):
        path = os.path.join('data/mvtec_3d_anomaly_detection', obj)
        if os.path.isdir(path):
            label = np.random.choice(anomaly_labels)
            img_choice = np.random.choice(os.listdir(os.path.join(path, f'test/{label}/gt')))
            if img_choice.endswith('.tiff'):
                gt_path = os.path.join(path, f'test/{label}/gt', img_choice)
                img_path = os.path.join(path, f'test/{label}/rgb', img_choice)
                
                img = tifffile.imread(img_path).reshape(-1, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(img)
                new_pcd = pcd.farthest_point_down_sample(n_points)
                # o3d.visualization.draw_geometries([new_pcd])
                farthest_points = np.asarray(new_pcd.points)
                pcds.append(farthest_points)
                
                # read PNG image and flatten
                gt_img = tifffile.imread(gt_path).reshape(-1, 3)
                gt_data.append(gt_img)
                
                i += 1
                if i == n_samples:
                    break
                
    gt_data = np.array(gt_data)
    pcds = np.array(pcds)
    
    return gt_data, pcds
        
    

class ADDataset(Dataset):
    def __init__(self, data, k, normalize=False):
        self.k = k
        self.data = torch.tensor(data, dtype=torch.float32)
        self.nearest_neighbors = self.compute_nearest_neighbors(data, k)
        
        # P - point cloud (n_points, 3)
        # p - point (3,)
        # knn_p - k nearest neighbors of p (k, 3)
        # q - nearest neighbor of p (3,)        
        # TODO: how does this normalization work?
        if normalize:
            # s = 0.0
            N = 0 # total number of points
            # for P_idx, P in enumerate(self.data):
            #     for i, p in enumerate(P):
            #         knn_p = P[self.nearest_neighbors[P_idx, i]]
            #         for q in knn_p:
            #             s += torch.norm(p - q)
            #         N += 1
            
                # P = self.data[P_idx]
                # indices = self.nearest_neighbors[P_idx]
                
                # Calculate distances and sum them
                # dists = torch.norm(P.unsqueeze(1) - P[indices], dim=2, p=2)  # Shape: (n_points, k)
                # s += torch.sum(dists)
            # s /= N * k
            # self.data /= s
            
            # self.s = s
                
            all_distances = []
    
            for P_idx, P in enumerate(self.data):
                knn_indices = self.nearest_neighbors[P_idx]
                knn_points = P[knn_indices]
                
                # Compute distances in a vectorized manner
                distances = torch.norm(P[:, None, :] - knn_points, dim=-1)
                all_distances.append(distances)
                N += P.shape[0]
            
            # Concatenate all distances and calculate the sum
            all_distances = torch.cat(all_distances, dim=0)
            s = all_distances.sum().item()
            
            # Calculate the normalization factor
            s /= N * k
            
            # Normalize the data
            for P in self.data:
                P /= s
                
        
    def compute_nearest_neighbors(self, data, k):
        nearest_neighbors_list = []
        for P in data:
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(P)
            _, indices = nbrs.kneighbors(P)
            # Exclude the point itself
            indices = indices[:, 1:]
            nearest_neighbors_list.append(indices)
            
        return torch.tensor(np.array(nearest_neighbors_list))         

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.nearest_neighbors[idx]
