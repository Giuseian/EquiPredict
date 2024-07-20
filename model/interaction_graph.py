import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class InteractionGraph(nn.Module):
    """Class to model interaction graphs between agents using learned features"""
    def __init__(self, hidden_nf, hid_channel, act_fn=nn.ReLU, category_num=4):
        super().__init__()
        self.hidden_nf = hidden_nf  # Number of features in hidden layers.
        self.hid_channel = hid_channel  # Number of channels in hidden layers.
        self.category_num = category_num  # Number of categories to classify interactions.
        self.tao = 1  # Parameter, possibly a threshold or scaling factor, to be defined.

        # Define MLPs for processing different aspects of agent interaction.
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hid_channel, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, hid_channel * 2),
            act_fn()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, hidden_nf)
        )

        self.category_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, category_num),
            act_fn()
        )

    def calc_category(self, h, coord, valid_mask):
        """Method to calculate interaction categories based on combined features using clustering"""
        batch_size, agent_num = coord.shape[:2]
        
        # Flatten and concatenate features for clustering
        h_flat = h.view(batch_size * agent_num, -1)
        coord_flat = coord.view(batch_size * agent_num, -1)
        features = torch.cat([h_flat, coord_flat], dim=-1).detach().cpu().numpy()
        
        # Perform K-means clustering to categorize interactions
        kmeans = KMeans(n_clusters=self.category_num)
        kmeans.fit(features)
        cluster_labels = kmeans.labels_
        
        # Reshape cluster labels to match the batch and agent dimensions
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=h.device)
        cluster_labels = cluster_labels.view(batch_size, agent_num)
        
        # Create one-hot encoding for cluster labels
        interaction_category = F.one_hot(cluster_labels, num_classes=self.category_num).float()
        
        # Expand dimensions to match expected output shape
        interaction_category = interaction_category.unsqueeze(2).expand(-1, -1, agent_num, -1)
        
        return interaction_category