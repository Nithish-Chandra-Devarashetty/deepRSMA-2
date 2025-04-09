import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from data import RNA_dataset, Molecule_dataset
import numpy as np
import os
import random
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.cuda.amp import autocast, GradScaler
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
EPOCH = 1500
hidden_dim = 128
seed = 42
LR = 1e-4
cold_type = 'rna'
RNA_type = 'All_sf'
seed_dataset = 2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CustomDualDataset(Dataset):
    def __init__(self, rna_ds, mol_ds):
        self.rna_ds = rna_ds
        self.mol_ds = mol_ds
        assert len(self.rna_ds) == len(self.mol_ds)
    def __getitem__(self, idx):
        return self.rna_ds[idx], self.mol_ds[idx]
    def __len__(self):
        return len(self.rna_ds)

class DeepRSMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna_graph_model = RNA_feature_extraction(hidden_dim)
        self.mole_graph_model = GNN_molecule(hidden_dim)
        self.mole_seq_model = mole_seq_model(hidden_dim)
        self.cross_attention = cross_attention(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        self.rna_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.ReLU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.mol_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.ReLU(), nn.Linear(hidden_dim*4, hidden_dim))

    def forward(self, rna_batch, mol_batch):
        rna_seq_out, rna_graph_out, rna_seq_mask, rna_graph_mask, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
        mol_graph_emb, mol_graph_final = self.mole_graph_model(mol_batch)
        mol_seq_emb, _, mol_seq_mask = self.mole_seq_model(mol_batch, device)
        mol_seq_final = (mol_seq_emb[-1] * mol_seq_mask.unsqueeze(-1)).mean(dim=1)

        rna_seq_mask = rna_seq_mask.to(device)
        rna_graph_mask = rna_graph_mask.to(device)
        rna_seq_final = rna_seq_final.to(device)
        rna_graph_final = rna_graph_final.to(device)
        mol_seq_mask = mol_seq_mask.to(device)
        mol_seq_final = mol_seq_final.to(device)
        mol_graph_final = mol_graph_final.to(device)

        max_graph_len = 128
        padded_graphs = torch.zeros(len(mol_batch), max_graph_len, hidden_dim, device=device)
        masks = torch.zeros(len(mol_batch), max_graph_len, device=device)

        flag = 0
        graph_lens = mol_batch.graph_len.to(device) if isinstance(mol_batch.graph_len, torch.Tensor) else mol_batch.graph_len
        for i, l in enumerate(graph_lens):
            padded_graphs[i, :l] = mol_graph_emb[flag:flag + l].to(device)
            masks[i, :l] = 1
            flag += l

        context_layer, _ = self.cross_attention(
            [rna_seq_out, rna_graph_out, mol_seq_emb[-1], padded_graphs],
            [rna_seq_mask, rna_graph_mask, mol_seq_mask, masks],
            device
        )

        out_rna, out_mol = context_layer[-1]
        rna_cross = (out_rna[:, :512] * rna_seq_mask.unsqueeze(-1)).mean(1) + rna_seq_final
        rna_cross = (rna_cross + (out_rna[:, 512:] * rna_graph_mask.unsqueeze(-1)).mean(1) + rna_graph_final) / 2
        rna_cross = self.rna_proj(rna_cross)

        mol_cross = (out_mol[:, :128] * mol_seq_mask.unsqueeze(-1)).mean(1) + mol_seq_final
        mol_cross = (mol_cross + (out_mol[:, 128:] * masks.unsqueeze(-1)).mean(1) + mol_graph_final) / 2
        mol_cross = self.mol_proj(mol_cross)

        return self.fc(torch.cat([rna_cross, mol_cross], dim=1))

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Resumed from epoch {ckpt['epoch']+1}")
        return ckpt['epoch'] + 1
    return 0

def evaluate(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for rna_batch, mol_batch in loader:
            pred = model(rna_batch.to(device), mol_batch.to(device)).squeeze().detach().cpu()
            label = rna_batch.y.cpu().float()
            preds.extend(pred.numpy())
            labels.extend(label.numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    rmse = mean_squared_error(labels, preds, squared=False)
    pcc = pearsonr(labels, preds)[0]
    scc = spearmanr(labels, preds)[0]
    return rmse, pcc, scc

# Main training logic
if __name__ == "__main__":
    set_seed(seed)
    rna_dataset = RNA_dataset(RNA_type)
    mol_dataset = Molecule_dataset(RNA_type)
    all_df = pd.read_csv(f'data/RSM_data/{RNA_type}_dataset_v1.csv', delimiter='\t')
    folds = [pd.read_csv(f'data/blind_test/cold_{cold_type}{i+1}.csv') for i in range(5)]

    scaler = GradScaler()

    for fold_idx, test_df in enumerate(folds):
        print(f"\n🚀 Starting Fold {fold_idx+1}")
        test_ids = all_df[all_df['Entry_ID'].isin(test_df['Entry_ID'])].index.tolist()
        train_ids = all_df[~all_df.index.isin(test_ids)].index.tolist()

        train_ds = CustomDualDataset(rna_dataset[train_ids], mol_dataset[train_ids])
        test_ds = CustomDualDataset(rna_dataset[test_ids], mol_dataset[test_ids])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = DeepRSMA().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        checkpoint_path = f'checkpoints/blind_fold{fold_idx+1}.pth'
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

        for epoch in range(start_epoch, EPOCH):
            start_time = time()
            model.train()
            total_loss = 0

            for rna_batch, mol_batch in train_loader:
                optimizer.zero_grad()
                with autocast():
                    preds = model(rna_batch.to(device), mol_batch.to(device)).squeeze()
                    labels = rna_batch.y.float().to(device)
                    loss = loss_fn(preds, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            rmse, pcc, scc = evaluate(model, test_loader)
            print(f"[Epoch {epoch}] Loss: {total_loss/len(train_loader):.4f} | RMSE: {rmse:.4f} | PCC: {pcc:.4f} | SCC: {scc:.4f} | Time: {time() - start_time:.2f}s")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
