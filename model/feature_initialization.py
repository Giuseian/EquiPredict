import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class FeatureInitialization(nn.Module):
    """Class for initializing features of input data"""
    def __init__(self, in_node_nf, hidden_nf, act_fn=nn.ReLU):
        super().__init__()
        # Embedding layers to transform input node features into a hidden feature space.
        self.embedding = nn.Linear(in_node_nf, hidden_nf // 2)
        self.embedding2 = nn.Linear(in_node_nf, hidden_nf // 2)
        
        self.act_fn = act_fn() # Activation function for non-linearity in feature transformation.
        self._initialize_weights() # Initialize weights with a specific strategy for better training performance.

    def initialize_weights(self):
        """Method to initialize weights of embedding layers using Xavier initialization"""
        init.xavier_uniform_(self.embedding.weight)
        init.xavier_uniform_(self.embedding2.weight)

    def forward(self, h, vel_angle):
        """Forward pass to compute the combined feature vector from input features and velocity angles"""
        h = self.embedding(h)
        vel_angle_embedding = self.embedding2(vel_angle)
        return torch.cat([h, vel_angle_embedding], dim=-1)