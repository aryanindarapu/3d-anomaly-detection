# generate synthetic 3D scenes using objects of the ModelNet10 dataset
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import trimesh
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from tifffile import tifffile
import os
import open3d as o3d

# def parse_off_file(file_path):
#     with open(file_path, 'r') as file:
#         first_line = file.readline().strip()
#         n_verts, n_faces, _ = map(int, file.readline().strip().split())
#         verts = [list(map(float, file.readline().strip().split())) for _ in range(n_verts)]
#         faces = [list(map(int, file.readline().strip().split()[1:])) for _ in range(n_faces)]
#     return np.array(verts), np.array(faces)

# https://medium.com/@sim30217/farthest-point-sampling-43ddedc25628
def farthest_point_sampling(points, n_points):
    # farthest_points = [np.random.randint(points.shape[0])] # choose a random point

    # for _ in range(n_points - 1):
    #     distances = np.sqrt(((points - points[farthest_points[-1]])**2).sum(axis=1))
    #     farthest_points.append(np.argmax(distances)) # choose point with max distance
        
    # return points[farthest_points]
    
    indices = np.random.choice(len(points), 1)
    farthest_points = points[indices, :]
    distances = np.linalg.norm(points - farthest_points[0], axis=1)
    
    for _ in range(1, n_points):
        # Select the point with maximum distance to the nearest already selected point
        farthest_point_index = np.argmax(distances)
        farthest_point = points[farthest_point_index:farthest_point_index+1]
        farthest_points = np.vstack([farthest_points, farthest_point])
        # Update the minimum distances
        new_distances = np.linalg.norm(points - farthest_point, axis=1)
        distances = np.minimum(distances, new_distances)
    
    return farthest_points

# return 500 training and 25 validation point clouds, choose n=64000 input points from each scene
def generate_mn10_data(n_points=64000, n_train=500, n_val=25, save=True):
    df = pd.read_csv('data/metadata_modelnet10.csv')
    # print(df['split'].unique())
    # train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    data = df[df['split'] == 'train']
    path_prefix = 'data/ModelNet10/'

    train_point_clouds = []
    val_point_clouds = []
    for i in tqdm(range(n_train + n_val)):
        if i == n_train:
            data = df[df['split'] == 'test']
        
        random_train = data.sample(n=10)
        # print(random_train.head())
        
        # TODO: do i generate a new scene each time?
        scene = trimesh.Scene()
        for file in random_train['object_path']:
            # verts, faces = parse_off_file(file)
            # print(verts, faces)
            mesh = trimesh.load(path_prefix + file)
            # mesh.show()
            
            # Scale longest side of bounding box to 1
            scale_factor = 1.0 / mesh.bounding_box.extents.max()
            mesh.apply_scale(scale_factor)

            # Rotate mesh
            rotation = trimesh.transformations.random_rotation_matrix()
            mesh.apply_transform(rotation)
            # mesh.show()
            
            # Place the mesh at location in 3D space
            translation_vector = np.random.uniform(-3, 3, size=3)
            # print(translation_vector)
            mesh.apply_translation(translation_vector)
            # print(mesh.vertices.shape)

            # Add the mesh to scene
            scene.add_geometry(mesh)
                    
        combined_mesh = scene.dump(concatenate=True)
        # print(type(combined_mesh.vertices))
        farthest_points = farthest_point_sampling(combined_mesh.vertices, n_points)
        
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
    
def generate_mvtec_data(n_points=64000, save=True):
    train_point_clouds = []
    val_point_clouds = []
    for obj in os.listdir('data/mvtec_3d_anomaly_detection'):
        path = os.path.join('data/mvtec_3d_anomaly_detection', obj)
        if os.path.isdir(path):
            path = os.path.join(path, 'train/good/xyz')
            for file in os.listdir(path):
                if file.endswith('.tiff'):
                    # img = Image.open(os.path.join(path, file))
                    img = tifffile.imread(os.path.join(path, file)).reshape(-1, 3)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(img)
                    new_pcd = pcd.farthest_point_down_sample(n_points)
                    # o3d.visualization.draw_geometries([new_pcd])
                    farthest_points = np.asarray(new_pcd.points)
                    train_point_clouds.append(farthest_points)
                    
            path = os.path.join(path, 'validation/good/xyz')
            for file in os.listdir(path):
                if file.endswith('.tiff'):
                    img = tifffile.imread(os.path.join(path, file)).reshape(-1, 3)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(img)
                    new_pcd = pcd.farthest_point_down_sample(n_points)
                    # o3d.visualization.draw_geometries([new_pcd])
                    farthest_points = np.asarray(new_pcd.points)
                    val_point_clouds.append(farthest_points)
    
    train_point_clouds = np.array(train_point_clouds)
    val_point_clouds = np.array(val_point_clouds)
    print(train_point_clouds.shape, val_point_clouds.shape)
    
    if save:
        np.save('data/mvtec_train_point_clouds.npy', train_point_clouds)
        np.save('data/mvtec_val_point_clouds.npy', val_point_clouds)
        
    return train_point_clouds, val_point_clouds

class ADDataset(Dataset):
    def __init__(self, data, k, normalize=False):
        self.k = k
        self.data = torch.tensor(data, dtype=torch.float32)
        self.nearest_neighbors = self.compute_nearest_neighbors(data, k)
        
        # P - point cloud (n_points, 3)
        # p - point (3,)
        # knn_p - k nearest neighbors of p (k, 3)
        # q - nearest neighbor of p (3,)
        # for P_idx, P in enumerate(self.data):
        #     s = 0.0
        #     for p_idx, p in enumerate(P):
        #         knn_p = P[self.nearest_neighbors[P_idx, p_idx]]
        #         for q in knn_p:
        #             s += np.linalg.norm(p - q)
                                    
        #     s /= len(P) * k
        #     self.s_list.append(s)
        #     self.data[P_idx] = P / s
            
        s_list = []
        
        if normalize:
            n_point_clouds, n_points, _ = self.data.shape
            for P_idx in range(n_point_clouds):
                P = self.data[P_idx]
                indices = self.nearest_neighbors[P_idx]
                
                # Calculate distances and sum them
                dists = torch.norm(P.unsqueeze(1) - P[indices], dim=2, p=2)  # Shape: (n_points, k)
                s = torch.sum(dists) / (n_points * self.k)
                
                s_list.append(s.item())
                self.data[P_idx] = P / s
        
        self.s_list = s_list
        # self.receptive_fields = self.compute_receptive_fields(data, k)
        
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
