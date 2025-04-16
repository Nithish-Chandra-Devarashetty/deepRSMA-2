import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GINEConv

class RNA_feature_extraction(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(RNA_feature_extraction, self).__init__()
        # Store dimensions for debugging
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Create a linear layer to project node features to hidden_channels
        self.node_encoder = nn.Linear(1, hidden_channels)  # RNA features are 1-dimensional

        # Create a linear layer to project edge features to hidden_channels
        self.edge_encoder = nn.Linear(1, hidden_channels)  # Edge features are 1-dimensional

        # First GNN layer
        nn1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINEConv(nn1, edge_dim=hidden_channels)

        # Second GNN layer
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINEConv(nn2, edge_dim=hidden_channels)

        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        # Print shapes for debugging
        # print(f"RNA node features shape: {x.shape}, dtype: {x.dtype}")
        # print(f"RNA edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
        # print(f"RNA edge_attr shape: {edge_attr.shape}, dtype: {edge_attr.dtype}")
        # print(f"RNA batch shape: {batch.shape}, dtype: {batch.dtype}")

        # Ensure correct data types
        x = x.float()
        edge_attr = edge_attr.float()

        # Reshape node features to [num_nodes, 1]
        x = x.view(-1, 1)

        # Encode node features to hidden_channels dimension
        x = self.node_encoder(x)

        # Encode edge features to match node feature dimension
        edge_attr = self.edge_encoder(edge_attr)

        # Apply first GINEConv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        # Apply second GINEConv layer
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Pool node features to get graph-level representation
        x = self.pool(x, batch)

        return x
