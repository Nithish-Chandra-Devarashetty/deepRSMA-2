import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

class RNA_dataset(InMemoryDataset):
    def __init__(self,
                 RNA_type,
                 root="dataset/rna",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.RNA_type = RNA_type
        self.root = os.path.join(root, RNA_type)
        self.csv_path = f"data/RSM_data/{RNA_type}_dataset_v1.csv"
        self.df = pd.read_csv(self.csv_path, delimiter='\t')

        # Paths for contact maps
        self.contact_map_dirs = {
            "Aptamers": "data/RNA_contact/Aptamers_contact",
            "miRNA": "data/RNA_contact/miRNA_contact",
            "Repeats": "data/RNA_contact/Repeats_contact",
            "Ribosomal": "data/RNA_contact/Ribosomal_contact",
            "Riboswitch": "data/RNA_contact/Riboswitch_contact",
            "Viral_RNA": "data/RNA_contact/Viral_RNA_contact"
        }

        # Embedding path
        self.emb_folder_path = 'data/representations_cv'

        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "data_rna.pt"

    def process(self):
        data_list = []

        for _, row in self.df.iterrows():
            entry_id = row['Entry_ID']
            target_id = row["Target_RNA_ID"]
            sequence = row['Target_RNA_sequence'][:512]  # Cap at 512 tokens
            y_value = row['pKd']

            # Load the first available contact map
            file_path = self._find_contact_map(entry_id)
            if file_path is None:
                print(f"[Warning] Contact map not found for {entry_id}")
                continue

            try:
                contact_map = np.loadtxt(file_path)
                contact_map = (contact_map >= 0.5).astype(np.float32)
                edges = np.argwhere(contact_map == 1)
            except Exception as e:
                print(f"[Error] Failed to load contact map for {entry_id}: {e}")
                continue

            try:
                emb_path = os.path.join(self.emb_folder_path, f"{target_id}.npy")
                rna_emb = torch.tensor(np.load(emb_path), dtype=torch.float32)
            except FileNotFoundError:
                print(f"[Missing] RNA embedding not found for {target_id}")
                continue

            one_hot_seq = [char_to_one_hot(c) for c in sequence]
            x = torch.tensor(one_hot_seq, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            rna_len = x.shape[0]

            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([y_value], dtype=torch.float32),
                t_id=target_id,
                e_id=entry_id,
                emb=rna_emb,
                rna_len=rna_len
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        print(f"[Info] Saving {len(data_list)} RNA samples to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])

    def _find_contact_map(self, entry_id):
        for path in self.contact_map_dirs.values():
            file_path = os.path.join(path, f"{entry_id}.prob_single")
            if os.path.exists(file_path):
                return file_path
        return None

# Nucleotide to one-hot (A, U, G, C, T, X, Y)
def char_to_one_hot(char):
    mapping = {'A': [1, 0, 0, 0, 0, 0, 0],
               'U': [0, 1, 0, 0, 0, 0, 0],
               'G': [0, 0, 1, 0, 0, 0, 0],
               'C': [0, 0, 0, 1, 0, 0, 0],
               'T': [0, 1, 0, 0, 0, 0, 0],
               'X': [0, 0, 0, 0, 1, 0, 0],
               'Y': [0, 0, 0, 0, 0, 1, 0]}
    return mapping.get(char, [0, 0, 0, 0, 0, 0, 1])  # Unknown chars → [0 0 0 0 0 0 1]
