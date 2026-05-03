import torch
import pandas as pd
import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Dataset
from collections import defaultdict
from tqdm import tqdm

# Konfiguracja logowania - wypisuje czas i wiadomość
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def scaffold_split_three_way(df, smiles_col="canonical_smiles", val_size=0.1, test_size=0.1):
    logging.info(f"Rozpoczynam Scaffold Split dla {len(df)} cząsteczek...")
    
    scaffolds = defaultdict(list)
    # Dodajemy pasek postępu dla analizy scaffoldów
    for idx, smi in enumerate(tqdm(df[smiles_col], desc="Analiza scaffoldów", leave=False)):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
        scaffolds[scaffold].append(idx)

    logging.info(f"Znaleziono {len(scaffolds)} unikalnych scaffoldów.")

    scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)
    train_idx, val_idx, test_idx = [], [], []
    val_cutoff = int(len(df) * val_size)
    test_cutoff = int(len(df) * test_size)
    
    for group in scaffold_sets:
        if len(test_idx) + len(group) <= test_cutoff:
            test_idx.extend(group)
        elif len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            train_idx.extend(group)
            
    logging.info(f"Podział zakończony: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]

def mol_to_graph(smiles, y):
    # Ta funkcja jest wywoływana wewnątrz pętli tqdm w Dataset
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # Cechy atomów: Liczba atomowa, stopień podstawienia, ładunek formalny
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(), 
            atom.GetDegree(), 
            atom.GetValence(Chem.ValenceType.IMPLICIT),
            int(atom.GetIsAromatic())
        ])
    x = torch.tensor(node_feats, dtype=torch.float)
    
    # Krawędzie
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    y_tensor = torch.as_tensor(y, dtype=torch.float)

    if y_tensor.dim() == 1:
        y_tensor = y_tensor.unsqueeze(0)

    return Data(x=x, edge_index=edge_index, y=y_tensor)

class MoleculeDataset(Dataset):
    def __init__(self, df, label_cols):
        super().__init__()
        self.data_list = []
        
        desc_name = f"Konwersja cząsteczek ({label_cols[0]}...)" if len(label_cols) > 1 else f"Konwersja {label_cols[0]}"
        
        logging.info(f"Przygotowywanie grafów dla zbioru danych...")
        
        # tqdm pokazuje pasek postępu podczas iteracji po DataFrame
        error_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc_name):
            y = row[label_cols].values.astype(float)
            graph = mol_to_graph(row['canonical_smiles'], y)
            if graph: 
                self.data_list.append(graph)
            else:
                error_count += 1
        
        if error_count > 0:
            logging.warning(f"Pominięto {error_count} cząsteczek z powodu błędów SMILES.")
        
        logging.info(f"Dataset gotowy. Liczba poprawnych grafów: {len(self.data_list)}")
            
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]