# generate synthetic 3D scenes using objects of the ModelNet10 dataset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from tifffile import tifffile
import os
# import trimesh
import cv2
from PIL import Image
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
    
    for i in tqdm(range(n_train + n_val)):        
        if i == n_train:
            data = df[df['split'] == 'test']
        
        random_train = data.sample(n=10)
        # print(random_train.head())
        
        # TODO: do i generate a new scene each time?
        o3d_meshes = []
        # combined_train_pcd = o3d.geometry.PointCloud()
        for file in random_train['object_path']:           
            mesh = o3d.io.read_triangle_mesh(path_prefix + file) # NOTE: did you sample points from the mesh or use the file's vertices?
                
            mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())

            # Apply a random rotation
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.random.rand(3) * 2 * np.pi)
            mesh.rotate(rotation, center=mesh.get_center())
            
            translation_vector = np.random.uniform(-3, 3, size=3)
            mesh.translate(translation_vector, relative=False)

            # Convert to point cloud
            pcd = mesh.sample_points_uniformly(n_points)
            o3d_meshes.append(pcd)
            
        # axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-3, -3, -3), max_bound=(3, 3, 3))
        # bounding_box.color = (1, 0, 0)  
        # o3d.visualization.draw_geometries([*o3d_meshes, axis_frame, bounding_box])

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
                
        # o3d.visualization.draw_geometries([combined_train_pcd, axis_frame, bounding_box])
        farthest_points = np.asarray(combined_train_pcd.points)
        
        if i < n_train:
            train_point_clouds.append(farthest_points)
        else:
            val_point_clouds.append(farthest_points)
            

    train_point_clouds = np.array(train_point_clouds)
    val_point_clouds = np.array(val_point_clouds)
    print(train_point_clouds.shape, val_point_clouds.shape)

    if save:
        np.save('data/train_point_clouds.npy', train_point_clouds)
        np.save('data/val_point_clouds.npy', val_point_clouds)
    
    return train_point_clouds, val_point_clouds


def get_mvtec_data(n_samples, n_points=64000, save=True, split='train'):
    pc_list = []
    f_list = np.array(get_mvtec_filepaths(split))
    # shuffle the file list
    # np.random.shuffle(f_list)
    for file in tqdm(f_list[:n_samples]):
        img = tifffile.imread(file).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(img)
        
        # https://medium.com/@sim30217/farthest-point-sampling-43ddedc25628
        new_pcd = pcd.farthest_point_down_sample(n_points)
        # o3d.visualization.draw_geometries([new_pcd])
        farthest_points = np.asarray(new_pcd.points)
        pc_list.append(farthest_points)
    
    pc_list = np.array(pc_list)
    print(f"{split} Dataset Shape:", pc_list.shape)
    
    if save:
        np.save(f'data/mvtec_{split}_pcdata_single.npy', pc_list)
        
    return pc_list

def get_mvtec_folder_data(n_points=64000):
    pcds, labels = [], []
    for obj in os.listdir('data/mvtec_test'):
        print(obj)
        # check if ends with .tiff
        if obj.endswith('.tiff'):
            path = os.path.join('data/mvtec_test', obj)
            img = tifffile.imread(path).reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(img)
            new_pcd = pcd.farthest_point_down_sample(n_points)
            farthest_points = np.asarray(new_pcd.points)
            pcds.append(farthest_points)
            labels.append(obj)
                
    pcds = np.array(pcds)
    labels = np.array(labels)
    
    # if save:
    #     np.save(f'data/mvtec_{split}_pcdata_single.npy', pcds)
    #     np.save(f'data/mvtec_{split}_labels.npy', labels)
        
    return pcds, labels                


def get_mvtec_test_data(n_samples=10, n_points=64000, save=True):
    gt_data, pcds, pcd_labels = [], [], []
    
    i = 0
    for obj in os.listdir('data/mvtec_3d_anomaly_detection'):
        print(f"Iteration {i}")
        path = os.path.join('data/mvtec_3d_anomaly_detection', obj)
        if os.path.isdir(path):
            anomaly_labels = os.listdir(os.path.join(path, 'test'))
            anomaly_labels.remove('good')
            label = np.random.choice(anomaly_labels)
            img_choice = np.random.choice(os.listdir(os.path.join(path, f'test/{label}/gt')))[:-4]
            
            gt_path = os.path.join(path, f'test/{label}/gt', f"{img_choice}.png")
            img_path = os.path.join(path, f'test/{label}/xyz', f"{img_choice}.tiff")
            
            img = tifffile.imread(img_path).reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(img)
            new_pcd = pcd.farthest_point_down_sample(n_points)
            # o3d.visualization.draw_geometries([new_pcd])
            farthest_points = np.asarray(new_pcd.points)
            pcds.append(farthest_points)
            pcd_labels.append(img_path)
            
            # read PNG image and flatten
            # gt_img = tifffile.imread(gt_path).reshape(-1, 3)
            # gt_data.append(gt_img)
            gt_img = Image.open(gt_path)
            gt_img = np.array(gt_img)
            gt_data.append(gt_img)
            
            i += 1
            if i == n_samples:
                break
                
    # gt_data = np.array(gt_data)
    pcds = np.array(pcds)
    pcd_labels = np.array(pcd_labels)
    
    if save:
        # np.save('data/mvtec_gt_test.npy', gt_data)
        np.save('data/mvtec_anomaly_test.npy', pcds)
        np.save('data/mvtec_anomaly_test_files.npy', pcd_labels)
    
    return gt_data, pcds, pcd_labels

        
def custom_collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    nearest_neighbors = torch.stack([item[1] for item in batch])
    nnbrs_obj_list = [item[2] for item in batch]
    return data, nearest_neighbors, nnbrs_obj_list    


class ADDataset(Dataset):
    def __init__(self, data, k, normalize=False, inference=False):
        self.k = k
        self.data = torch.tensor(data, dtype=torch.float32)
        self.nearest_neighbors = self.compute_nearest_neighbors(data, k)
        
        # P - point cloud (n_points, 3)
        # p - point (3,)
        # knn_p - k nearest neighbors of p (k, 3)
        # q - nearest neighbor of p (3,)    
        if inference:
            # normalize each input point cloud individually
            # self.data /= 0.0014349749565124512
            self.data /= 0.0015581955507577184
            
        elif normalize:
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
                
                # Compute distances between each point and its k nearest neighbors
                distances = torch.norm(P[:, None, :] - knn_points, dim=-1)
                all_distances.append(distances)
                N += P.shape[0]
            
            # Concatenate all distances and calculate the sum
            all_distances = torch.cat(all_distances, dim=0)
            s = all_distances.sum().item()
            
            # Calculate the normalization factor
            s /= N * k
            
            # Normalize the data
            for idx in range(len(self.data)):
                self.data[idx] /= s
                
            
            print(f"Normalization factor: {s}")
            # for i in range(5):
            #     # visualize the normalized point cloud
            #     pc = o3d.geometry.PointCloud()
            #     pc.points = o3d.utility.Vector3dVector(self.data[i].numpy())
            #     o3d.visualization.draw_geometries([pc])
            
                
        
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
    
    def get_nnbrs(self):
        return self.nearest_neighbors
