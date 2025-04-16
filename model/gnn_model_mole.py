import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GCNNet(nn.Module):  # Renamed back to original class name
    def __init__(self, in_channels, hidden_channels):
        super(GCNNet, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels))
        # Create a linear layer to project edge features to the same dimension as node features
        self.edge_encoder = nn.Linear(3, hidden_channels)
        self.conv1 = GINEConv(nn1, edge_dim=hidden_channels)  # Set edge_dim to match hidden_channels

        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=hidden_channels)  # Set edge_dim to match hidden_channels

        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        # Encode edge features to match node feature dimension
        edge_attr = self.edge_encoder(edge_attr)

        # Apply GINEConv layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Pool node features
        x = self.pool(x, batch)
        return x
