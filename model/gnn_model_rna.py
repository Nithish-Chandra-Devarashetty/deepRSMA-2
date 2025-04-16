import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GINEConv

class RNA_feature_extraction(nn.Module):  # Renamed here
    def __init__(self, in_channels, hidden_channels):
        super(RNA_feature_extraction, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=1)  # Set edge_dim to 1

        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=1)  # Set edge_dim to 1

        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.pool(x, batch)
        return x
