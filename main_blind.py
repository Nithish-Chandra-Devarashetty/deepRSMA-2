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
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (aligned with Code B)
BATCH_SIZE = 16
EPOCH = 1500
hidden_dim = 128
seed = 2
LR = 5e-4
cold_type = 'rna'
RNA_type = 'All_sf'
seed_dataset = 2

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)

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

        self.line1 = nn.Linear(hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.rna1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.rna2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim*4, hidden_dim)
        
        self.relu = nn.ReLU()

    def forward(self, rna_batch, mol_batch):
        rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
        mole_graph_emb, mole_graph_final = self.mole_graph_model(mol_batch)
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mol_batch, device)
        mole_seq_final = (mole_seq_emb[-1] * (mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)

        flag = 0
        mole_out_graph = []
        mask = []
        for i in mol_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag + count_i]
            temp = torch.zeros((128 - x.size()[0]), hidden_dim).to(device)
            x = torch.cat((x, temp), 0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float).to(device)

        context_layer, _ = self.cross_attention(
            [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
            [rna_mask_seq.to(device), rna_mask_graph.to(device), mole_mask_seq.to(device), mole_mask_graph],
            device
        )

        out_rna = context_layer[-1][0]
        out_mol = context_layer[-1][1]

        rna_cross_seq = ((out_rna[:, 0:512] * (rna_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final) / 2
        rna_cross_stru = ((out_rna[:, 512:] * (rna_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2
        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout(self.relu(self.rna1(rna_cross))))

        mole_cross_seq = ((out_mol[:, 0:128] * (mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mol[:, 128:] * (mole_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout(self.relu(self.mole1(mole_cross))))

        out = torch.cat([rna_cross, mole_cross], dim=1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        return out

def evaluate(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for rna_batch, mol_batch in loader:
            pred = model(rna_batch.to(device), mol_batch.to(device)).squeeze().detach().cpu().numpy()
            label = rna_batch.y.cpu().float().numpy()
            preds.extend(pred.flatten().tolist())
            labels.extend(label.flatten().tolist())
    preds = np.array(preds)
    labels = np.array(labels)
    rmse = mean_squared_error(labels, preds) ** 0.5
    pcc = pearsonr(labels, preds)[0]
    scc = spearmanr(labels, preds)[0]
    return rmse, pcc, scc

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

# Main training logic
if __name__ == "__main__":
    set_seed(seed)
    rna_dataset = RNA_dataset(RNA_type)
    mol_dataset = Molecule_dataset(RNA_type)
    all_df = pd.read_csv(f'data/RSM_data/{RNA_type}_dataset_v1.csv', delimiter='\t')
    folds = [pd.read_csv(f'data/blind_test/cold_{cold_type}{i+1}.csv') for i in range(5)]

    p_list = []
    s_list = []
    r_list = []

    for fold_idx, test_df in enumerate(folds):
        print(f"\n🚀 Starting Fold {fold_idx+1}")
        test_ids = all_df[all_df['Entry赴积累了 Entry_ID'].isin(test_df['Entry_ID']).index.tolist()
        
        train_ids = []
        for j, other_df in enumerate(folds):
            if j != fold_idx:
                train_ids.extend(other_df['Entry_ID'].tolist())
        train_ids = all_df[all_df['Entry_ID'].isin(train_ids)].index.tolist()

        print(len(test_ids), len(train_ids))

        train_ds = CustomDualDataset(rna_dataset[train_ids], mol_dataset[train_ids])
        test_ds = CustomDualDataset(rna_dataset[test_ids], mol_dataset[test_ids])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)

        model = DeepRSMA().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
        loss_fn = nn.MSELoss()

        # Load periodic checkpoint if exists
        checkpoint_path = f'checkpoints/blind_fold{fold_idx+1}.pth'
        best_path = f'save/model_blind_{cold_type}{seed_dataset}_{fold_idx+1}_{seed}.pth'
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path) if os.path.exists(checkpoint_path) else 0

        max_p = -1
        max_s = 0
        max_rmse = 0

        for epoch in range(start_epoch, EPOCH):
            start_time = time()
            model.train()
            total_loss = 0

            for rna_batch, mol_batch in train_loader:
                optimizer.zero_grad()
                preds = model(rna_batch.to(device), mol_batch.to(device))
                labels = rna_batch.y.float().to(device)
                loss = loss_fn(preds.squeeze(dim=1), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            rmse, pcc, scc = evaluate(model, test_loader)
            save_checkpoint(model, optimizer, epoch, checkpoint_path)  # Save every epoch
            if pcc > max_p:
                max_p = pcc
                max_s = scc
                max_rmse = rmse
                torch.save(model.state_dict(), best_path)  # Save best PCC model
                print(f"epo: {epoch} pcc: {pcc:.4f} scc: {scc:.4f} rmse: {rmse:.4f}")

        p_list.append(max_p)
        s_list.append(max_s)
        r_list.append(max_rmse)

    print(f"p: {np.mean(p_list):.4f}")
    print(f"s: {np.mean(s_list):.4f}")
    print(f"rmse: {np.mean(r_list):.4f}")