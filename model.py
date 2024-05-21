import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SharedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
            
    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(x, 0.2)
        return x


class LFABlock(nn.Module):
    def __init__(self, k, d):
        super(LFABlock, self).__init__()
        self.k = k
        self.d = d
        self.smlp = SharedMLP(4, d)
        
    def forward(self, points, features, knn_indices):
        """Runs the forward pass of the LFABlock.

        Args:
            points (torch.tensor): Shape: (B, N, D)
            features (torch.tensor): Shape: (F_N, F_D) 
            knn_indices (torch.tensor): Shape: (B, N, k)

        Returns:
            torch.tensor: Shape: (B, N, 2*F_D)
        """
        
        # knn_indices: for each point, the indices of its k=32 nearest neighbors
        # points: (B, N, D)
        B, N, D = points.shape
        F_B, F_N, F_D = features.shape
        
        # compute kNN of each point
        # dist_matrix = torch.cdist(points, points)
        # _, knn_indices = torch.topk(dist_matrix, self.k, largest=False)
        
        knn_points = torch.gather(points.unsqueeze(1).expand(B, N, N, D), 2, knn_indices.unsqueeze(-1).expand(B, N, self.k, D))
        
        # compute geometric features, G: (B, N, k, 4)
        # for each point, compute the difference between it and its k nearest neighbors
        p_expanded = points.unsqueeze(2).expand(B, N, self.k, D)
        geometric_features = torch.cat([p_expanded - knn_points, torch.norm(p_expanded - knn_points, dim=-1, keepdim=True)], dim=-1)
        
        geometric_features = geometric_features.view(B * N * self.k, -1)
        mlp_output = self.smlp(geometric_features)
        mlp_output = mlp_output.view(B, N, self.k, self.d)
        
        features_expanded = torch.gather(features.unsqueeze(1).expand(B, N, N, F_D), 2, knn_indices.unsqueeze(-1).expand(B, N, self.k, F_D))
        
        concatenated_features = torch.cat([mlp_output, features_expanded], dim=-1)
        
        average_pooled_features = torch.mean(concatenated_features, dim=2)
        
        return average_pooled_features.squeeze()


class ResidualBlock(nn.Module):
    def __init__(self, d=64, k=32):
        super(ResidualBlock, self).__init__()
        self.initial_smlp = SharedMLP(d, d // 4)
        self.lfa1 = LFABlock(k, d // 4)
        self.lfa2 = LFABlock(k, d // 2)
        self.final_smlp = SharedMLP(d, d)
        self.res_smlp = SharedMLP(d, d)
    
    def forward(self, points, features, knn_indices):
        x = self.initial_smlp(features)
        x = self.lfa1(points, x, knn_indices)
        x = self.lfa2(points, x, knn_indices)
        x = self.final_smlp(x)
        residual = self.res_smlp(features)
        x = x + residual
        # x = F.relu(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, d=64, m=1024):
        super(Decoder, self).__init__()
        
        self.input = nn.Linear(d, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, m*3)
        
    def forward(self, x):
        B, _, _ = x.shape
        x = self.input(x)
        x = F.leaky_relu(self.hidden1(x), -0.05)
        x = F.leaky_relu(self.hidden1(x), -0.05)
        x = self.output(x)
        
        return x.view(B, -1, 3)
    
    
class Model(nn.Module):
    def __init__(self, is_student=False):
        super(Model, self).__init__()
        self.residual_blocks = nn.ModuleList([ResidualBlock() for _ in range(4)])
        
        if is_student:
            # initialize with uniformly distributed random weights
            for block in self.residual_blocks:
                for param in block.parameters():
                    nn.init.uniform_(param, -0.1, 0.1) # TODO: is this the correct range?
        
    def forward(self, points, features, knn_indices):
        # run point cloud input through residual blocks
        for block in self.residual_blocks:
            features = block(points, features, knn_indices)
        
        return features
    
        # x = None
        # if self.is_teacher:
        #     # randomly select 16 descriptors from the last residual block to pass to the decoder
            
        #     x = self.decoder(x)
        
        # return x, features, indices
        
# 4 residual blocks, d = 64
# shared MLPs are dense layer and leaky ReLU with alpha = 0.2
# LFA uses nearest neighbor search with k = 32

### Teacher network
# 250 epochs, Adam (lr=0.001, weight decay=1e-6)
# feed single input at a time

### Student network
# 100 epochs, Adam (lr=0.001, weight decay=1e-5)