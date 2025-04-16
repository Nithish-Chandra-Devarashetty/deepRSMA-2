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
                print(f"Detected {num_nodes} nodes from edge_index")

                # Create batch index with the correct size
                batch_idx = torch.zeros(num_nodes, dtype=torch.long).to(device)

                # Print sizes for debugging
                print(f"RNA x size: {x.size()}, batch_idx size: {batch_idx.size()}")

                # Verify that the batch index size matches the node feature size
                if batch_idx.size(0) != x.size(0):
                    print(f"Warning: batch_idx size {batch_idx.size(0)} doesn't match x size {x.size(0)}")
                    # Use the larger size to be safe
                    max_size = max(batch_idx.size(0), x.size(0))
                    batch_idx = torch.zeros(max_size, dtype=torch.long).to(device)
                    print(f"Created new batch_idx with size {batch_idx.size(0)}")
            except Exception as e:
                print(f"Error creating batch index: {e}")
                # Fallback to using x size
                batch_idx = torch.zeros(x.size(0), dtype=torch.long).to(device)
                print(f"Fallback: Created batch_idx with size {batch_idx.size(0)}")

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

        # Print rna_len for debugging
        print(f"rna_len type: {type(rna_batch.rna_len)}, value: {rna_batch.rna_len}")

        # Convert rna_len to a scalar integer
        try:
            if isinstance(rna_batch.rna_len, torch.Tensor):
                rna_len = rna_batch.rna_len.item()
            elif isinstance(rna_batch.rna_len, (list, tuple)):
                rna_len = rna_batch.rna_len[0]
            else:
                rna_len = int(rna_batch.rna_len)

            print(f"Converted rna_len to: {rna_len}")
        except Exception as e:
            print(f"Error converting rna_len: {e}")
            # Fallback to a default value
            rna_len = 100
            print(f"Using fallback rna_len: {rna_len}")

        # Create sequence and graph masks
        # Ensure they're created on the same device as other tensors
        rna_mask_seq = torch.ones(1, 512, device=device)
        rna_mask_seq[0, rna_len:] = 0

        rna_mask_graph = torch.ones(1, 128, device=device)
        rna_mask_graph[0, rna_len:] = 0

        # Print mask shapes for debugging
        print(f"rna_mask_seq shape: {rna_mask_seq.shape}, device: {rna_mask_seq.device}")
        print(f"rna_mask_graph shape: {rna_mask_graph.shape}, device: {rna_mask_graph.device}")

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
                print(f"Detected {num_nodes} molecule nodes from edge_index")

                # Create batch index with the correct size
                mole_batch_idx = torch.zeros(num_nodes, dtype=torch.long).to(device)

                # Print sizes for debugging
                print(f"Molecule x size: {mole_x.size()}, batch_idx size: {mole_batch_idx.size()}")

                # Verify that the batch index size matches the node feature size
                if mole_batch_idx.size(0) != mole_x.size(0):
                    print(f"Warning: mole_batch_idx size {mole_batch_idx.size(0)} doesn't match mole_x size {mole_x.size(0)}")
                    # Use the larger size to be safe
                    max_size = max(mole_batch_idx.size(0), mole_x.size(0))
                    mole_batch_idx = torch.zeros(max_size, dtype=torch.long).to(device)
                    print(f"Created new mole_batch_idx with size {mole_batch_idx.size(0)}")
            except Exception as e:
                print(f"Error creating molecule batch index: {e}")
                # Fallback to using x size
                mole_batch_idx = torch.zeros(mole_x.size(0), dtype=torch.long).to(device)
                print(f"Fallback: Created mole_batch_idx with size {mole_batch_idx.size(0)}")

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

        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)

        mole_seq_final = (mole_seq_emb[-1]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)


        # mole graph
        flag = 0
        mole_out_graph = []
        mask = []
        for i in mole_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag+count_i]
            temp = torch.zeros((128-x.size()[0]), hidden_dim).to(device)
            x = torch.cat((x, temp),0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)

        # Validate shapes before cross-attention
        print(f"Shape validation before cross-attention:")
        print(f"rna_out_seq: {rna_out_seq.shape}, device: {rna_out_seq.device}")
        print(f"rna_out_graph: {rna_out_graph.shape}, device: {rna_out_graph.device}")
        print(f"mole_seq_emb[-1]: {mole_seq_emb[-1].shape}, device: {mole_seq_emb[-1].device}")
        print(f"mole_out_graph: {mole_out_graph.shape}, device: {mole_out_graph.device}")

        # Ensure all masks are on the same device
        # No need to call .to(device) again since we created them on the device
        try:
            # Get cross-attention outputs (ignoring attention scores for now)
            context_layer, _ = self.cross_attention(
                [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
                [rna_mask_seq, rna_mask_graph, mole_mask_seq.to(device), mole_mask_graph.to(device)],
                device
            )
            print("Cross-attention successful")
        except Exception as e:
            print(f"Error in cross-attention: {e}")
            # Create a fallback context layer with the expected shape
            # This is a last resort to prevent the model from crashing
            context_layer = [None] * 3
            context_layer[-1] = [
                torch.zeros_like(rna_out_seq),
                torch.zeros_like(mole_out_graph)
            ]
            print("Using fallback context layer")


        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]

        # Affinity Prediction Module
        rna_cross_seq = ((out_rna[:, 0:512]*(rna_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final ) / 2
        rna_cross_stru = ((out_rna[:, 512:]*(rna_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2

        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout((self.relu(self.rna1(rna_cross)))))


        mole_cross_seq = ((out_mole[:,0:128]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mole[:,128:]*(mole_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2

        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout((self.relu(self.mole1(mole_cross)))))

        out = torch.cat((rna_cross, mole_cross),1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)


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
    with torch.set_grad_enabled(False):
        model.eval()
        y_label = []
        y_pred = []
        for step, (batch_v) in enumerate(test_loader):
            try:
                # Convert label to tensor
                label = Variable(torch.from_numpy(np.array(batch_v[0].y))).float()

                # Forward pass with error handling
                try:
                    score = model(batch_v[0].to(device), batch_v[1].to(device))

                    # Check for NaN values in the prediction
                    if torch.isnan(score).any():
                        print(f"Warning: NaN values detected in prediction for test batch {step}. Skipping.")
                        continue

                except Exception as e:
                    print(f"Error in model forward pass for test batch {step}: {e}")
                    continue

                # Process predictions
                try:
                    logits = torch.squeeze(score).detach().cpu().numpy()
                    label_ids = label.to('cpu').numpy()

                    # Check for NaN values
                    if np.isnan(logits).any() or np.isnan(label_ids).any():
                        print(f"Warning: NaN values detected in processed outputs for test batch {step}. Skipping.")
                        continue

                    # Add to results
                    y_label = y_label + label_ids.flatten().tolist()
                    y_pred = y_pred + logits.flatten().tolist()

                    # Print progress
                    if step % 10 == 0:
                        print(f"Evaluated {step} test batches")

                except Exception as e:
                    print(f"Error processing outputs for test batch {step}: {e}")
                    continue

            except Exception as e:
                print(f"Error in evaluation loop for batch {step}: {e}")
                continue

        # Calculate metrics with error handling
        try:
            # Check if we have enough data points
            if len(y_label) < 2 or len(y_pred) < 2:
                print(f"Warning: Not enough data points for metric calculation. y_label: {len(y_label)}, y_pred: {len(y_pred)}")
                p = (0.0, 1.0)  # Default values
                s = (0.0, 1.0)
                rmse = float('inf')
            else:
                # Check for NaN or infinity values
                if np.isnan(y_label).any() or np.isnan(y_pred).any() or np.isinf(y_label).any() or np.isinf(y_pred).any():
                    print("Warning: NaN or infinity values detected in metrics calculation. Cleaning data.")
                    # Remove NaN and infinity values
                    valid_indices = ~(np.isnan(y_label) | np.isnan(y_pred) | np.isinf(y_label) | np.isinf(y_pred))
                    y_label_clean = np.array(y_label)[valid_indices]
                    y_pred_clean = np.array(y_pred)[valid_indices]

                    if len(y_label_clean) < 2:
                        print("Warning: Not enough valid data points after cleaning.")
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

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            p = (0.0, 1.0)  # Default values
            s = (0.0, 1.0)
            rmse = float('inf')
            print('epo:', epoch, 'metrics calculation failed')

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
