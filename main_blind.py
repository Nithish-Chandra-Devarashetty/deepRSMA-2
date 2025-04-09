import multiprocessing
multiprocessing.freeze_support()

import torch
from torch.utils.data import Dataset
from data import RNA_dataset, Molecule_dataset
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import random
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16
EPOCH = 1500
hidden_dim = 128
seed_dataset = 2
cold_type = 'rna'
seed = 2
LR = 5e-4

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
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)
    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]
    def __len__(self):
        return len(self.dataset1)

class DeepRSMA(nn.Module):
    def __init__(self):
        super(DeepRSMA, self).__init__()
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

    def forward(self, rna_batch, mole_batch):
        rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)
        mole_seq_final = (mole_seq_emb[-1]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)

        flag = 0
        mole_out_graph = []
        mask = []
        for i in mole_batch.graph_len:
            x = mole_graph_emb[flag:flag+i]
            temp = torch.zeros((128 - x.size()[0]), hidden_dim).to(device)
            x = torch.cat((x, temp), 0)
            mole_out_graph.append(x)
            mask.append([1]*i + [0]*(128 - i))
            flag += i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)

        context_layer, attention_score = self.cross_attention(
            [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
            [rna_mask_seq.to(device), rna_mask_graph.to(device), mole_mask_seq.to(device), mole_mask_graph.to(device)],
            device)

        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]

        rna_cross_seq = ((out_rna[:, 0:512]*(rna_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final) / 2
        rna_cross_stru = ((out_rna[:, 512:]*(rna_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2
        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout(self.relu(self.rna1(rna_cross))))

        mole_cross_seq = ((out_mole[:, 0:128]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mole[:, 128:]*(mole_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout(self.relu(self.mole1(mole_cross))))

        out = torch.cat((rna_cross, mole_cross), 1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        return out

# Checkpointing
def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming training from epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1
    return 0

if __name__ == '__main__':
    set_seed(seed)

    RNA_type = 'All_sf'
    rna_dataset = RNA_dataset(RNA_type)
    molecule_dataset = Molecule_dataset(RNA_type)

    all_df = pd.read_csv(f'data/RSM_data/{RNA_type}_dataset_v1.csv', delimiter='\t')
    df = [pd.read_csv(f'data/blind_test/cold_{cold_type}{i+1}.csv') for i in range(5)]
    df_cold_all = pd.concat(df, axis=0).reset_index()

    fold = 0
    p_list, s_list, r_list = [], [], []

    for i, df_f in enumerate(df):
        test_id = all_df[all_df['Entry_ID'].isin(df_f['Entry_ID'].tolist())].index.tolist()
        train_id = [id for j, d in enumerate(df) if j != i for id in d['Entry_ID'].tolist()]
        train_id = all_df[all_df['Entry_ID'].isin(train_id)].index.tolist()

        fold += 1
        print(f"Fold {fold} - Train size: {len(train_id)} | Test size: {len(test_id)}")

        train_dataset = CustomDualDataset(rna_dataset[train_id], molecule_dataset[train_id])
        test_dataset = CustomDualDataset(rna_dataset[test_id], molecule_dataset[test_id])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, shuffle=False)

        model = DeepRSMA().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
        loss_fct = nn.MSELoss()

        max_p, max_s, max_rmse = -1, 0, 0
        checkpoint_path = f'checkpoints/model_blind_{cold_type}{seed_dataset}_{fold}_{seed}.pth'
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

        for epo in range(start_epoch, EPOCH):
            model.train()
            train_loss = 0
            for step, (batch_rna, batch_mole) in enumerate(train_loader):
                optimizer.zero_grad()
                pre = model(batch_rna.to(device), batch_mole.to(device))
                labels = batch_rna.y.to(device).float()
                loss = loss_fct(pre.squeeze(dim=1), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Evaluation
            model.eval()
            y_label, y_pred = [], []
            with torch.no_grad():
                for step, (batch_rna_test, batch_mole_test) in enumerate(test_loader):
                    label = Variable(torch.from_numpy(np.array(batch_rna_test.y))).float()
                    score = model(batch_rna_test.to(device), batch_mole_test.to(device))
                    logits = torch.squeeze(score).detach().cpu().numpy()
                    label_ids = label.to('cpu').numpy()
                    y_label += label_ids.flatten().tolist()
                    y_pred += logits.flatten().tolist()

            p = pearsonr(y_label, y_pred)
            s = spearmanr(y_label, y_pred)
            rmse = np.sqrt(mean_squared_error(y_label, y_pred))

            if max_p < p[0]:
                print(f'[Epoch {epo}] PCC: {p[0]:.4f}, SCC: {s[0]:.4f}, RMSE: {rmse:.4f}')
                max_p, max_s, max_rmse = p[0], s[0], rmse
                os.makedirs("save", exist_ok=True)
                torch.save(model.state_dict(), f'save/model_blind_{cold_type}{seed_dataset}_{fold}_{seed}.pth')

            save_checkpoint(model, optimizer, epo, checkpoint_path)

        p_list.append(max_p)
        s_list.append(max_s)
        r_list.append(max_rmse)

        print(f'[Fold {fold}] Mean PCC: {np.mean(p_list):.4f}, SCC: {np.mean(s_list):.4f}, RMSE: {np.mean(r_list):.4f}')
