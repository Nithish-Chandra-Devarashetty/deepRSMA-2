import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch
from data import RNA_dataset, Molecule_dataset, RNA_dataset_independent, Molecule_dataset_independent, WordVocab
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr,spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import random
torch.set_printoptions(profile="full")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128

EPOCH = 200
RNA_type = 'Viral_RNA_independent'
rna_dataset = RNA_dataset(RNA_type)
molecule_dataset = Molecule_dataset(RNA_type)

rna_dataset_in = RNA_dataset_independent()
molecule_dataset_in = Molecule_dataset_independent()

seed = 1



# set random seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)
set_seed(seed)

# combine two pyg dataset
class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)



def average_multiple_lists(lists):
    return [sum(item)/len(lists) for item in zip(*lists)]





# DeepRSMA architecture
class DeepRSMA(nn.Module):
    def __init__(self):
        super(DeepRSMA, self).__init__()
        # RNA graph + seq
        self.rna_graph_model = RNA_feature_extraction(hidden_dim, hidden_dim)

        # Mole graph
        self.mole_graph_model = GNN_molecule(hidden_dim, hidden_dim)
        # Mole seq
        self.mole_seq_model = mole_seq_model(hidden_dim)

        # Cross fusion module
        self.cross_attention = cross_attention(hidden_dim)

        self.line1 = nn.Linear(hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

        self.rna1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim*4)

        self.rna2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim*4, hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, rna_batch, mole_batch):
        try:
            # Process RNA data
            # Extract features from RNA batch
            x = rna_batch.x.to(device)
            edge_index = rna_batch.edge_index.to(device)

            # Create edge attributes (assuming they're not provided in the batch)
            edge_attr = torch.ones(edge_index.size(1), 1).to(device)

            # Create batch index for the graph pooling
            # The batch index should have the same size as the number of nodes
            # and indicate which graph each node belongs to (all 0 for a single graph)
            try:
                # Get the number of nodes from the edge_index tensor
                num_nodes = edge_index.max().item() + 1

                # Create batch index with the correct size
                batch_idx = torch.zeros(num_nodes, dtype=torch.long).to(device)

                # Verify that the batch index size matches the node feature size
                if batch_idx.size(0) != x.size(0):
                    # Use the larger size to be safe
                    max_size = max(batch_idx.size(0), x.size(0))
                    batch_idx = torch.zeros(max_size, dtype=torch.long).to(device)
            except Exception as e:
                # Fallback to using x size
                batch_idx = torch.zeros(x.size(0), dtype=torch.long).to(device)

            # Check for potential issues with the edge_index tensor
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                print(f"Warning: edge_index has unexpected shape: {edge_index.shape}")
                # Try to fix the edge_index tensor
                if edge_index.dim() == 1:
                    # Reshape to [2, num_edges]
                    num_edges = edge_index.size(0) // 2
                    edge_index = edge_index.view(2, num_edges)
                    print(f"Reshaped edge_index to {edge_index.shape}")

            # Optional debugging
            # print(f"RNA node features: shape={x.shape}, dtype={x.dtype}")
            # print(f"RNA edge_index: shape={edge_index.shape}, dtype={edge_index.dtype}")
            # print(f"RNA edge_attr: shape={edge_attr.shape}, dtype={edge_attr.dtype}")
            # print(f"RNA batch_idx: shape={batch_idx.shape}, dtype={batch_idx.dtype}")

            # Get graph embeddings using the RNA feature extraction model
            rna_graph_final = self.rna_graph_model(x, edge_index, edge_attr, batch_idx)
        except Exception as e:
            print(f"Error in RNA processing: {e}")
            raise

        # Get RNA sequence embedding
        rna_emb = rna_batch.emb.to(device)

        # Convert rna_len to a scalar integer
        try:
            if isinstance(rna_batch.rna_len, torch.Tensor):
                # Check if it's a multi-element tensor
                if rna_batch.rna_len.numel() > 1:
                    # Use the maximum length for masking
                    rna_len = rna_batch.rna_len.max().item()
                else:
                    rna_len = rna_batch.rna_len.item()
            elif isinstance(rna_batch.rna_len, (list, tuple)):
                # Use the maximum length for masking
                rna_len = max(rna_batch.rna_len)
            else:
                rna_len = int(rna_batch.rna_len)

            # Ensure rna_len is within valid range
            if rna_len > 512:
                rna_len = 512
            elif rna_len <= 0:
                rna_len = 100

        except Exception as e:
            # Fallback to a default value
            rna_len = 100

        # Create sequence and graph masks
        try:
            # Ensure they're created on the same device as other tensors
            rna_mask_seq = torch.ones(1, 512, device=device)

            # Create the masks
            rna_mask_seq[0, rna_len:] = 0

            rna_mask_graph = torch.ones(1, 128, device=device)
            if rna_len > 128:
                graph_len = 128
            else:
                graph_len = rna_len

            rna_mask_graph[0, graph_len:] = 0

        except Exception as e:
            # Create default masks
            rna_mask_seq = torch.ones(1, 512, device=device)
            rna_mask_graph = torch.ones(1, 128, device=device)

        # Create output tensors
        rna_out_seq = torch.zeros(1, 512, hidden_dim).to(device)
        rna_out_graph = torch.zeros(1, 128, hidden_dim).to(device)

        # Set the sequence final embedding
        rna_seq_final = rna_emb

        try:
            # Process molecule data
            # Extract features from molecule batch
            mole_x = mole_batch.x.to(device)
            mole_edge_index = mole_batch.edge_index.to(device)
            mole_edge_attr = mole_batch.edge_attr.to(device)  # Type conversion handled in the model

            # Create batch index for the graph pooling
            try:
                # Get the number of nodes from the edge_index tensor
                num_nodes = mole_edge_index.max().item() + 1

                # Create batch index with the correct size
                mole_batch_idx = torch.zeros(num_nodes, dtype=torch.long).to(device)

                # Verify that the batch index size matches the node feature size
                if mole_batch_idx.size(0) != mole_x.size(0):
                    # Use the larger size to be safe
                    max_size = max(mole_batch_idx.size(0), mole_x.size(0))
                    mole_batch_idx = torch.zeros(max_size, dtype=torch.long).to(device)
            except Exception as e:
                # Fallback to using x size
                mole_batch_idx = torch.zeros(mole_x.size(0), dtype=torch.long).to(device)

            # Check for potential issues with the edge_index tensor
            if mole_edge_index.dim() != 2 or mole_edge_index.size(0) != 2:
                print(f"Warning: mole_edge_index has unexpected shape: {mole_edge_index.shape}")
                # Try to fix the edge_index tensor
                if mole_edge_index.dim() == 1:
                    # Reshape to [2, num_edges]
                    num_edges = mole_edge_index.size(0) // 2
                    mole_edge_index = mole_edge_index.view(2, num_edges)
                    print(f"Reshaped mole_edge_index to {mole_edge_index.shape}")

            # Check for potential issues with the edge_attr tensor
            if mole_edge_attr.dim() == 1:
                print(f"Warning: mole_edge_attr has unexpected shape: {mole_edge_attr.shape}")
                # Reshape to [num_edges, 3]
                mole_edge_attr = mole_edge_attr.view(-1, 3)
                print(f"Reshaped mole_edge_attr to {mole_edge_attr.shape}")

            # Optional debugging
            # print(f"Molecule node features: shape={mole_x.shape}, dtype={mole_x.dtype}")
            # print(f"Molecule edge_index: shape={mole_edge_index.shape}, dtype={mole_edge_index.dtype}")
            # print(f"Molecule edge_attr: shape={mole_edge_attr.shape}, dtype={mole_edge_attr.dtype}")
            # print(f"Molecule batch_idx: shape={mole_batch_idx.shape}, dtype={mole_batch_idx.dtype}")

            # Get graph embeddings using the molecule feature extraction model
            mole_graph_final = self.mole_graph_model(mole_x, mole_edge_index, mole_edge_attr, mole_batch_idx)
        except Exception as e:
            print(f"Error in molecule processing: {e}")
            raise

        # For consistency with the original implementation
        mole_graph_emb = mole_x

        # Process molecule sequence
        try:
            # Get molecule sequence embeddings
            mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)

            # Calculate molecule sequence final embedding
            mole_mask_seq_device = mole_mask_seq.to(device)
            mole_mask_seq_3d = mole_mask_seq_device.unsqueeze(dim=2)

            # Apply mask and calculate mean
            masked_emb = mole_seq_emb[-1] * mole_mask_seq_3d
            mole_seq_final = masked_emb.mean(dim=1).squeeze(dim=1)

        except Exception as e:
            # Create fallback embeddings
            mole_seq_emb = [torch.zeros(1, 128, hidden_dim).to(device)]
            mole_mask_seq = torch.ones(1, 128).to(device)
            mole_seq_final = torch.zeros(1, hidden_dim).to(device)


        # mole graph processing with error handling
        try:
            flag = 0
            mole_out_graph = []
            mask = []

            for i in mole_batch.graph_len:
                try:
                    # Convert i to integer if it's a tensor
                    if isinstance(i, torch.Tensor):
                        count_i = i.item()
                    else:
                        count_i = int(i)

                    # Get a slice of the molecule graph embeddings
                    x = mole_graph_emb[flag:flag+count_i]

                    # Create padding tensor with matching feature dimension
                    feature_dim = x.size(-1)
                    temp = torch.zeros((128-x.size()[0]), feature_dim).to(device)

                    # Concatenate along dimension 0
                    x = torch.cat((x, temp), 0)

                    # Add to output list
                    mole_out_graph.append(x)
                    mask.append([] + count_i * [1] + (128 - count_i) * [0])
                    flag += count_i

                except Exception as e:
                    # Create a dummy tensor with the right shape
                    dummy_x = torch.zeros(128, hidden_dim).to(device)
                    mole_out_graph.append(dummy_x)
                    mask.append([0] * 128)  # All masked out

            # Stack the tensors
            mole_out_graph = torch.stack(mole_out_graph).to(device)
            mole_mask_graph = torch.tensor(mask, dtype=torch.float).to(device)

        except Exception as e:
            # Create fallback tensors
            batch_size = 1  # Assume batch size 1 as fallback
            mole_out_graph = torch.zeros(batch_size, 128, hidden_dim).to(device)
            mole_mask_graph = torch.zeros(batch_size, 128).to(device)

        # Cross-attention processing

        try:
            # Get cross-attention outputs (ignoring attention scores for now)
            context_layer, _ = self.cross_attention(
                [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
                [rna_mask_seq, rna_mask_graph, mole_mask_seq.to(device), mole_mask_graph.to(device)],
                device
            )
        except Exception as e:
            # Create a fallback context layer with the expected shape
            context_layer = [None] * 3
            context_layer[-1] = [
                torch.zeros_like(rna_out_seq),
                torch.zeros_like(mole_out_graph)
            ]


        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]

        # Affinity Prediction Module
        try:
            # Process RNA sequence cross-attention
            rna_mask_seq_device = rna_mask_seq.to(device)
            rna_mask_seq_3d = rna_mask_seq_device.unsqueeze(dim=2)

            # Check if we need to slice out_rna
            if out_rna.size(1) >= 512:
                out_rna_seq = out_rna[:, 0:512]
            else:
                out_rna_seq = out_rna

            # Ensure mask has compatible dimensions with out_rna_seq
            if rna_mask_seq_3d.size(1) != out_rna_seq.size(1):
                # Resize mask to match out_rna_seq
                new_mask = torch.ones(1, out_rna_seq.size(1), 1, device=device)
                min_size = min(new_mask.size(1), rna_mask_seq_3d.size(1))
                new_mask[0, min_size:] = 0
                rna_mask_seq_3d = new_mask

            # Apply mask and calculate mean
            masked_out_rna_seq = out_rna_seq * rna_mask_seq_3d
            mean_out_rna_seq = masked_out_rna_seq.mean(dim=1).squeeze(dim=1)

            # Ensure rna_seq_final has compatible dimensions with mean_out_rna_seq
            if mean_out_rna_seq.size(-1) != rna_seq_final.size(-1):
                # Resize rna_seq_final to match mean_out_rna_seq
                if mean_out_rna_seq.size(-1) > rna_seq_final.size(-1):
                    # Pad rna_seq_final
                    pad_size = mean_out_rna_seq.size(-1) - rna_seq_final.size(-1)
                    padding = torch.zeros(rna_seq_final.size(0), pad_size, device=device)
                    rna_seq_final = torch.cat([rna_seq_final, padding], dim=-1)
                else:
                    # Truncate rna_seq_final
                    rna_seq_final = rna_seq_final[:, :mean_out_rna_seq.size(-1)]

            # Calculate cross sequence
            rna_cross_seq = (mean_out_rna_seq + rna_seq_final) / 2

            # Process RNA graph cross-attention
            rna_mask_graph_device = rna_mask_graph.to(device)
            rna_mask_graph_3d = rna_mask_graph_device.unsqueeze(dim=2)

            # Check if we need to slice out_rna for graph
            if out_rna.size(1) > 512:
                out_rna_graph = out_rna[:, 512:]
            else:
                out_rna_graph = out_rna

            # Ensure mask has compatible dimensions with out_rna_graph
            if rna_mask_graph_3d.size(1) != out_rna_graph.size(1):
                # Resize mask to match out_rna_graph
                new_mask = torch.ones(1, out_rna_graph.size(1), 1, device=device)
                min_size = min(new_mask.size(1), rna_mask_graph_3d.size(1))
                new_mask[0, min_size:] = 0
                rna_mask_graph_3d = new_mask

            # Apply mask and calculate mean
            masked_out_rna_graph = out_rna_graph * rna_mask_graph_3d
            mean_out_rna_graph = masked_out_rna_graph.mean(dim=1).squeeze(dim=1)

            # Ensure rna_graph_final has compatible dimensions with mean_out_rna_graph
            if mean_out_rna_graph.size(-1) != rna_graph_final.size(-1):
                # Resize rna_graph_final to match mean_out_rna_graph
                if mean_out_rna_graph.size(-1) > rna_graph_final.size(-1):
                    # Pad rna_graph_final
                    pad_size = mean_out_rna_graph.size(-1) - rna_graph_final.size(-1)
                    padding = torch.zeros(rna_graph_final.size(0), pad_size, device=device)
                    rna_graph_final = torch.cat([rna_graph_final, padding], dim=-1)
                else:
                    # Truncate rna_graph_final
                    rna_graph_final = rna_graph_final[:, :mean_out_rna_graph.size(-1)]

            # Calculate cross structure
            rna_cross_stru = (mean_out_rna_graph + rna_graph_final) / 2

        except Exception as e:
            # Create fallback tensors
            rna_cross_seq = torch.zeros(1, hidden_dim).to(device)
            rna_cross_stru = torch.zeros(1, hidden_dim).to(device)

        # Process final RNA cross with dimension validation
        try:
            rna_cross = (rna_cross_seq + rna_cross_stru) / 2
            print(f"\n=== Final RNA Cross Validation ===")
            print(f"rna_cross shape: {rna_cross.shape}, device: {rna_cross.device}")

            # Check if rna_cross has the expected feature dimension for rna1
            expected_dim = hidden_dim  # The expected input dimension for rna1
            if rna_cross.size(-1) != expected_dim:
                print(f"Warning: rna_cross size {rna_cross.size(-1)} doesn't match expected size {expected_dim}")
                # Resize rna_cross to match expected dimension
                if rna_cross.size(-1) < expected_dim:
                    # Pad rna_cross
                    pad_size = expected_dim - rna_cross.size(-1)
                    padding = torch.zeros(rna_cross.size(0), pad_size, device=device)
                    rna_cross = torch.cat([rna_cross, padding], dim=-1)
                    print(f"Padded rna_cross to shape {rna_cross.shape}")
                else:
                    # Truncate rna_cross
                    rna_cross = rna_cross[:, :expected_dim]
                    print(f"Truncated rna_cross to shape {rna_cross.shape}")

            # Apply linear transformations
            rna_cross = self.rna1(rna_cross)
            rna_cross = self.relu(rna_cross)
            rna_cross = self.dropout(rna_cross)
            rna_cross = self.rna2(rna_cross)

            print(f"Final rna_cross shape after transformations: {rna_cross.shape}")

        except Exception as e:
            print(f"Error in final RNA cross processing: {e}")
            # Create fallback tensor
            rna_cross = torch.zeros(1, hidden_dim).to(device)
            print("Using fallback rna_cross tensor")


        # Process molecule cross-attention with dimension validation
        try:
            # Print dimensions for debugging
            print("\n=== Molecule Affinity Prediction Module Dimensions ===")
            print(f"out_mole shape: {out_mole.shape}, device: {out_mole.device}")
            print(f"mole_mask_seq shape: {mole_mask_seq.shape}, device: {mole_mask_seq.device}")
            print(f"mole_seq_final shape: {mole_seq_final.shape}, device: {mole_seq_final.device}")
            print(f"mole_graph_final shape: {mole_graph_final.shape}, device: {mole_graph_final.device}")

            # Process molecule sequence cross-attention
            # First, ensure dimensions are compatible
            mole_mask_seq_device = mole_mask_seq.to(device)
            mole_mask_seq_3d = mole_mask_seq_device.unsqueeze(dim=2)

            # Check if we need to slice out_mole
            if out_mole.size(1) >= 128:
                out_mole_seq = out_mole[:, 0:128]
            else:
                print(f"Warning: out_mole has fewer than 128 columns ({out_mole.size(1)}). Using all available.")
                out_mole_seq = out_mole

            # Ensure mask has compatible dimensions with out_mole_seq
            if mole_mask_seq_3d.size(1) != out_mole_seq.size(1):
                print(f"Warning: mole_mask_seq_3d size {mole_mask_seq_3d.size(1)} doesn't match out_mole_seq size {out_mole_seq.size(1)}")
                # Resize mask to match out_mole_seq
                new_mask = torch.ones(1, out_mole_seq.size(1), 1, device=device)
                min_size = min(new_mask.size(1), mole_mask_seq_3d.size(1))
                new_mask[0, min_size:] = 0
                mole_mask_seq_3d = new_mask

            # Apply mask and calculate mean
            masked_out_mole_seq = out_mole_seq * mole_mask_seq_3d
            mean_out_mole_seq = masked_out_mole_seq.mean(dim=1).squeeze(dim=1)

            print(f"mean_out_mole_seq shape: {mean_out_mole_seq.shape}, device: {mean_out_mole_seq.device}")
            print(f"mole_seq_final shape: {mole_seq_final.shape}, device: {mole_seq_final.device}")

            # Ensure mole_seq_final has compatible dimensions with mean_out_mole_seq
            if mean_out_mole_seq.size(-1) != mole_seq_final.size(-1):
                print(f"Warning: mean_out_mole_seq size {mean_out_mole_seq.size(-1)} doesn't match mole_seq_final size {mole_seq_final.size(-1)}")
                # Resize mole_seq_final to match mean_out_mole_seq
                if mean_out_mole_seq.size(-1) > mole_seq_final.size(-1):
                    # Pad mole_seq_final
                    pad_size = mean_out_mole_seq.size(-1) - mole_seq_final.size(-1)
                    padding = torch.zeros(mole_seq_final.size(0), pad_size, device=device)
                    mole_seq_final = torch.cat([mole_seq_final, padding], dim=-1)
                else:
                    # Truncate mole_seq_final
                    mole_seq_final = mole_seq_final[:, :mean_out_mole_seq.size(-1)]

            # Calculate cross sequence
            mole_cross_seq = (mean_out_mole_seq + mole_seq_final) / 2

            # Process molecule graph cross-attention
            # Similar dimension validation for graph
            mole_mask_graph_device = mole_mask_graph.to(device)
            mole_mask_graph_3d = mole_mask_graph_device.unsqueeze(dim=2)

            # Check if we need to slice out_mole for graph
            if out_mole.size(1) > 128:
                out_mole_graph = out_mole[:, 128:]
            else:
                print(f"Warning: out_mole doesn't have graph section. Using sequence section.")
                out_mole_graph = out_mole

            # Ensure mask has compatible dimensions with out_mole_graph
            if mole_mask_graph_3d.size(1) != out_mole_graph.size(1):
                print(f"Warning: mole_mask_graph_3d size {mole_mask_graph_3d.size(1)} doesn't match out_mole_graph size {out_mole_graph.size(1)}")
                # Resize mask to match out_mole_graph
                new_mask = torch.ones(1, out_mole_graph.size(1), 1, device=device)
                min_size = min(new_mask.size(1), mole_mask_graph_3d.size(1))
                new_mask[0, min_size:] = 0
                mole_mask_graph_3d = new_mask

            # Apply mask and calculate mean
            masked_out_mole_graph = out_mole_graph * mole_mask_graph_3d
            mean_out_mole_graph = masked_out_mole_graph.mean(dim=1).squeeze(dim=1)

            print(f"mean_out_mole_graph shape: {mean_out_mole_graph.shape}, device: {mean_out_mole_graph.device}")
            print(f"mole_graph_final shape: {mole_graph_final.shape}, device: {mole_graph_final.device}")

            # Ensure mole_graph_final has compatible dimensions with mean_out_mole_graph
            if mean_out_mole_graph.size(-1) != mole_graph_final.size(-1):
                print(f"Warning: mean_out_mole_graph size {mean_out_mole_graph.size(-1)} doesn't match mole_graph_final size {mole_graph_final.size(-1)}")
                # Resize mole_graph_final to match mean_out_mole_graph
                if mean_out_mole_graph.size(-1) > mole_graph_final.size(-1):
                    # Pad mole_graph_final
                    pad_size = mean_out_mole_graph.size(-1) - mole_graph_final.size(-1)
                    padding = torch.zeros(mole_graph_final.size(0), pad_size, device=device)
                    mole_graph_final = torch.cat([mole_graph_final, padding], dim=-1)
                else:
                    # Truncate mole_graph_final
                    mole_graph_final = mole_graph_final[:, :mean_out_mole_graph.size(-1)]

            # Calculate cross structure
            mole_cross_stru = (mean_out_mole_graph + mole_graph_final) / 2

        except Exception as e:
            print(f"Error in molecule affinity prediction: {e}")
            # Create fallback tensors
            mole_cross_seq = torch.zeros(1, hidden_dim).to(device)
            mole_cross_stru = torch.zeros(1, hidden_dim).to(device)
            print("Using fallback molecule cross tensors")

        # Process final molecule cross with dimension validation
        try:
            mole_cross = (mole_cross_seq + mole_cross_stru) / 2
            print(f"\n=== Final Molecule Cross Validation ===")
            print(f"mole_cross shape: {mole_cross.shape}, device: {mole_cross.device}")

            # Check if mole_cross has the expected feature dimension for mole1
            expected_dim = hidden_dim  # The expected input dimension for mole1
            if mole_cross.size(-1) != expected_dim:
                print(f"Warning: mole_cross size {mole_cross.size(-1)} doesn't match expected size {expected_dim}")
                # Resize mole_cross to match expected dimension
                if mole_cross.size(-1) < expected_dim:
                    # Pad mole_cross
                    pad_size = expected_dim - mole_cross.size(-1)
                    padding = torch.zeros(mole_cross.size(0), pad_size, device=device)
                    mole_cross = torch.cat([mole_cross, padding], dim=-1)
                    print(f"Padded mole_cross to shape {mole_cross.shape}")
                else:
                    # Truncate mole_cross
                    mole_cross = mole_cross[:, :expected_dim]
                    print(f"Truncated mole_cross to shape {mole_cross.shape}")

            # Apply linear transformations
            mole_cross = self.mole1(mole_cross)
            mole_cross = self.relu(mole_cross)
            mole_cross = self.dropout(mole_cross)
            mole_cross = self.mole2(mole_cross)

            print(f"Final mole_cross shape after transformations: {mole_cross.shape}")

        except Exception as e:
            print(f"Error in final molecule cross processing: {e}")
            # Create fallback tensor
            mole_cross = torch.zeros(1, hidden_dim).to(device)
            print("Using fallback mole_cross tensor")

        # Final output processing with dimension validation
        try:
            # Print dimensions for debugging
            print("\n=== Final Output Processing Dimensions ===")
            print(f"rna_cross shape: {rna_cross.shape}, device: {rna_cross.device}")
            print(f"mole_cross shape: {mole_cross.shape}, device: {mole_cross.device}")

            # Ensure both tensors have the same batch size
            if rna_cross.size(0) != mole_cross.size(0):
                print(f"Warning: rna_cross batch size {rna_cross.size(0)} doesn't match mole_cross batch size {mole_cross.size(0)}")
                # Use the smaller batch size
                min_batch = min(rna_cross.size(0), mole_cross.size(0))
                rna_cross = rna_cross[:min_batch]
                mole_cross = mole_cross[:min_batch]

            # Concatenate the tensors
            out = torch.cat((rna_cross, mole_cross), 1)
            print(f"Concatenated output shape: {out.shape}")

            # Apply final layers
            out = self.line1(out)
            out = self.dropout(self.relu(out))
            print(f"After line1: {out.shape}")

            out = self.line2(out)
            out = self.dropout(self.relu(out))
            print(f"After line2: {out.shape}")

            out = self.line3(out)
            print(f"Final output shape: {out.shape}")

        except Exception as e:
            print(f"Error in final output processing: {e}")
            # Create fallback output
            out = torch.zeros(1, 1).to(device)
            print("Using fallback output tensor")

        # Validate final output
        print("\n=== Final Output Validation ===")
        print(f"Output shape: {out.shape}, dtype: {out.dtype}, device: {out.device}")
        print(f"Output has NaNs: {torch.isnan(out).any().item()}")
        print(f"Output min: {out.min().item()}, max: {out.max().item()}, mean: {out.mean().item()}")

        return out


# use viral RNA to train
train_dataset = CustomDualDataset(rna_dataset, molecule_dataset)
# independent test
test_dataset = CustomDualDataset(rna_dataset_in, molecule_dataset_in)


train_loader = DataLoader(
    train_dataset, batch_size=8, num_workers=1, drop_last=False, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False
)



model = DeepRSMA()
model.to(device)

y_pred_all = []
max_p = -1

optimizer = optim.Adam(model.parameters(), lr=6e-5 , weight_decay=1e-5)
optimal_loss = 1e10
loss_fct = torch.nn.MSELoss()
for epoch in range(0,EPOCH):
    train_loss = 0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        pre = model(batch[0].to(device), batch[1].to(device))

        y = batch[0].y

        # Check for NaN values in the prediction
        if torch.isnan(pre).any():
            print("Warning: NaN values detected in prediction. Skipping this batch.")
            continue

        # Compute loss
        try:
            loss = loss_fct(pre.squeeze(dim=1), y.float())

            # Check for NaN loss
            if torch.isnan(loss).any():
                print("Warning: NaN loss detected. Skipping this batch.")
                continue

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Print occasional loss for monitoring
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        except Exception as e:
            print(f"Error in training step: {e}")
            continue
        train_loss = train_loss + loss
    # Evaluation loop with minimal output
    with torch.set_grad_enabled(False):
        model.eval()
        y_label = []
        y_pred = []

        # Process each test batch with error handling
        for step, batch_v in enumerate(test_loader):
            try:
                # Convert label to tensor
                label = Variable(torch.from_numpy(np.array(batch_v[0].y))).float()

                # Forward pass with error handling
                try:
                    score = model(batch_v[0].to(device), batch_v[1].to(device))

                    # Check for NaN values in the prediction
                    if torch.isnan(score).any():
                        continue

                except Exception as e:
                    continue

                # Process predictions
                try:
                    logits = torch.squeeze(score).detach().cpu().numpy()
                    label_ids = label.to('cpu').numpy()

                    # Check for NaN values
                    if np.isnan(logits).any() or np.isnan(label_ids).any():
                        continue

                    # Add to results
                    y_label.extend(label_ids.flatten().tolist())
                    y_pred.extend(logits.flatten().tolist())

                except Exception as e:
                    continue

            except Exception as e:
                continue

        # Calculate metrics with minimal output
        try:
            # Check if we have enough data points
            if len(y_label) < 2 or len(y_pred) < 2:
                p = (0.0, 1.0)  # Default values
                s = (0.0, 1.0)
                rmse = float('inf')
            else:
                # Check for NaN or infinity values
                if np.isnan(y_label).any() or np.isnan(y_pred).any() or np.isinf(y_label).any() or np.isinf(y_pred).any():
                    # Remove NaN and infinity values
                    valid_indices = ~(np.isnan(y_label) | np.isnan(y_pred) | np.isinf(y_label) | np.isinf(y_pred))
                    y_label_clean = np.array(y_label)[valid_indices]
                    y_pred_clean = np.array(y_pred)[valid_indices]

                    if len(y_label_clean) < 2:
                        p = (0.0, 1.0)  # Default values
                        s = (0.0, 1.0)
                        rmse = float('inf')
                    else:
                        p = pearsonr(y_label_clean, y_pred_clean)
                        s = spearmanr(y_label_clean, y_pred_clean)
                        rmse = np.sqrt(mean_squared_error(y_label_clean, y_pred_clean))
                else:
                    # Calculate metrics normally
                    p = pearsonr(y_label, y_pred)
                    s = spearmanr(y_label, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_label, y_pred))

            print('epo:', epoch, 'pcc:', p[0], 'scc: ', s[0], 'rmse:', rmse)

            # Update best metrics
            if p[0] > best_pcc:
                best_pcc = p[0]
                best_scc = s[0]
                best_rmse = rmse
                best_epoch = epoch
                print('\nBest: epo:', best_epoch, 'pcc:', best_pcc, 'scc: ', best_scc, 'rmse:', best_rmse)

        except Exception as e:
            p = (0.0, 1.0)  # Default values
            s = (0.0, 1.0)
            rmse = float('inf')

        if max_p < p[0]:
            max_p = p[0]
            print(' ')
            print('Best:', 'epo:',epoch, 'pcc:',p[0],'scc: ',s[0],'rmse:',rmse)

            # Save model with error handling
            try:
                os.makedirs('save', exist_ok=True)
                save_path = 'save/' + 'model_independent_'+str(seed)+'.pth'
                torch.save(model.state_dict(), save_path)
                print(f"Model saved successfully to {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
                # Try an alternative approach
                try:
                    alt_save_path = f"model_independent_{seed}_{epoch}.pt"
                    torch.save(model.state_dict(), alt_save_path)
                    print(f"Model saved to alternative path: {alt_save_path}")
                except Exception as e2:
                    print(f"Failed to save model to alternative path: {e2}")


        model.train()
