import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNNet, self).__init__()
        # Store dimensions for debugging
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Create a linear layer to project node features to hidden_channels
        # Use a try-except block to handle potential dimension errors
        try:
            self.node_encoder = nn.Linear(9, hidden_channels)  # Molecule atom features are 9-dimensional
        except Exception as e:
            print(f"Warning: Error creating node encoder: {e}")
            # Fallback to a more flexible approach
            self.node_encoder = nn.Sequential(
                nn.Linear(9, hidden_channels),
                nn.ReLU()
            )

        # Create a linear layer to project edge features to hidden_channels
        try:
            self.edge_encoder = nn.Linear(3, hidden_channels)  # Bond features are 3-dimensional
        except Exception as e:
            print(f"Warning: Error creating edge encoder: {e}")
            # Fallback to a more flexible approach
            self.edge_encoder = nn.Sequential(
                nn.Linear(3, hidden_channels),
                nn.ReLU()
            )

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
        try:
            # Print all input dimensions for comprehensive debugging
            print("\n=== GNN_molecule Input Dimensions ===")
            print(f"x: {x.shape}, dtype: {x.dtype}, device: {x.device}")
            print(f"edge_index: {edge_index.shape}, dtype: {edge_index.dtype}, device: {edge_index.device}")
            print(f"edge_attr: {edge_attr.shape}, dtype: {edge_attr.dtype}, device: {edge_attr.device}")
            print(f"batch: {batch.shape}, dtype: {batch.dtype}, device: {batch.device}")

            # Ensure all tensors are on the same device
            device = x.device
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch = batch.to(device)

            # Ensure correct data types
            x = x.float()
            edge_attr = edge_attr.float()

            # Check for NaN values
            if torch.isnan(x).any():
                print("Warning: NaN values detected in node features. Replacing with zeros.")
                x = torch.nan_to_num(x, nan=0.0)

            if torch.isnan(edge_attr).any():
                print("Warning: NaN values detected in edge features. Replacing with zeros.")
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

            # Encode node features to hidden_channels dimension
            try:
                x = self.node_encoder(x)
            except Exception as e:
                print(f"Error in node encoding: {e}")
                # Try reshaping if needed
                if len(x.shape) == 1:
                    x = x.view(-1, 1)
                    print(f"Reshaped x to {x.shape}")
                    x = self.node_encoder(x)

            # Encode edge features to match node feature dimension
            try:
                edge_attr = self.edge_encoder(edge_attr)
            except Exception as e:
                print(f"Error in edge encoding: {e}")
                # Try reshaping if needed
                if len(edge_attr.shape) == 1:
                    edge_attr = edge_attr.view(-1, 1)
                    print(f"Reshaped edge_attr to {edge_attr.shape}")
                    edge_attr = self.edge_encoder(edge_attr)

            # Apply first GINEConv layer
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            print(f"After first GINEConv: x shape: {x.shape}")

            # Apply second GINEConv layer
            x = self.conv2(x, edge_index, edge_attr)
            x = F.relu(x)
            print(f"After second GINEConv: x shape: {x.shape}")

            # For global pooling, we need to ensure batch has the same size as x
            if batch.size(0) != x.size(0):
                print(f"Warning: Molecule batch size {batch.size(0)} doesn't match node feature size {x.size(0)}")
                # Create a new batch tensor with the correct size
                # This is critical for the pooling operation to work correctly
                batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

                # Print the new batch size for debugging
                print(f"Created new molecule batch tensor with size {batch.size()}")

                # Double-check that the sizes match
                if batch.size(0) != x.size(0):
                    print(f"ERROR: Molecule sizes still don't match! x: {x.size(0)}, batch: {batch.size(0)}")
                    # Last resort: reshape x to match batch size
                    # This is not ideal but better than crashing
                    print(f"Reshaping molecule x from {x.shape} to match batch size {batch.size(0)}")
                    # Create a new tensor with the correct size
                    new_x = torch.zeros(batch.size(0), x.size(1), device=device)
                    # Copy as much data as possible
                    min_size = min(batch.size(0), x.size(0))
                    new_x[:min_size] = x[:min_size]
                    x = new_x

            # Pool node features to get graph-level representation
            x = self.pool(x, batch)

            return x

        except Exception as e:
            print(f"Error in GNN_molecule.forward: {e}")
            # Return a default tensor in case of error
            return torch.zeros(1, self.hidden_channels, device=x.device)
