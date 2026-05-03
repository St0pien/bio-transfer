import torch
from torch_geometric.loader import DataLoader
from data_utils import scaffold_split_three_way, MoleculeDataset
from models.gnn_model import GNNMoleculeModel
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import matplotlib
import argparse  # Dodano bibliotekę do obsługi parametrów

# Ustawienie backendu dla serwerów bez X11 (headless)
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_loss(history, mode, target):
    """Generuje i zapisuje wykres straty."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Train Loss', color='#3498db', linewidth=2)
    plt.plot(history['val'], label='Val Loss', color='#e74c3c', linewidth=2)
    
    plt.title(f"Loss Curve: {mode.upper()} - {target}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogits)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/loss_{mode}_{target}.png", dpi=150)
    plt.close()

def evaluate(model, loader, criterion, is_multitask=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y.view(out.shape)
            
            if is_multitask:
                mask = ~torch.isnan(y)
                if mask.sum() == 0: continue
                loss = criterion(out[mask], y[mask])
            else:
                loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_one_epoch(model, loader, optimizer, criterion, is_multitask=False):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        y = batch.y.view(out.shape)
        
        if is_multitask:
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            loss = criterion(out[mask], y[mask])
        else:
            loss = criterion(out, y)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Zaktualizowana sygnatura funkcji - dodano parametr epochs
def run_training(mode="downstream", target="BACE1", epochs=30):
    print(f"\n🚀 Start: {mode.upper()} | Target: {target} | Epochs: {epochs} | Device: {device}")
    
    # --- Przygotowanie danych ---
    if mode == "upstream":
        df = pd.read_csv("./data/upstream/upstream_filtered.csv")
        # Używamy pivot_table z aggfunc='mean', aby uśrednić pChEMBL dla duplikatów
        df_labels = df.pivot_table(
            index="molecule_chembl_id", 
            columns="target_name", 
            values="pchembl_value", 
            aggfunc='mean'
        )
        df_labels = (df_labels >= 6.0).astype(float)
        smiles_map = df.drop_duplicates("molecule_chembl_id").set_index("molecule_chembl_id")["canonical_smiles"]
        final_df = df_labels.join(smiles_map).reset_index()
        label_cols = df_labels.columns.tolist()
    else:
        final_df = pd.read_csv(f"./data/downstream/{target}_labeled.csv")
        label_cols = ["active"]

    train_df, val_df, test_df = scaffold_split_three_way(final_df)
    
    train_loader = DataLoader(MoleculeDataset(train_df, label_cols), batch_size=32, shuffle=True)
    val_loader = DataLoader(MoleculeDataset(val_df, label_cols), batch_size=32)
    test_loader = DataLoader(MoleculeDataset(test_df, label_cols), batch_size=32)

    model = GNNMoleculeModel(num_node_features=4, num_targets=len(label_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {'train': [], 'val': []}
    best_val_loss = float('inf')

    # Wykorzystanie parametru epochs przekazanego do funkcji
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, is_multitask=(mode=="upstream"))
        val_loss = evaluate(model, val_loader, criterion, is_multitask=(mode=="upstream"))
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/best_gnn_{mode}_{target}.pt"
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print(f"  ✨ Nowy rekord walidacji! Model zapisany.")

    plot_loss(history, mode, target)
    print(f"📈 Wykres straty zapisany w ./plots/loss_{mode}_{target}.png")

    # Test końcowy
    checkpoint = torch.load(f"checkpoints/best_gnn_{mode}_{target}.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = evaluate(model, test_loader, criterion, is_multitask=(mode=="upstream"))
    print(f"\n✅ Finalny Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    # Konfiguracja parsera argumentów
    parser = argparse.ArgumentParser(description="Trenowanie modelu GNN dla molekuł.")
    
    parser.add_argument("--mode", type=str, default="downstream", 
                        choices=["upstream", "downstream"], 
                        help="Tryb uczenia (domyślnie: downstream)")
    
    parser.add_argument("--target", type=str, default="BACE1", 
                        help="Nazwa celu/targetu (domyślnie: BACE1)")
    
    parser.add_argument("--epochs", type=int, default=30, 
                        help="Liczba epok treningowych (domyślnie: 30)")

    args = parser.parse_args()

    # Uruchomienie treningu z parametrami z konsoli
    run_training(mode=args.mode, target=args.target, epochs=args.epochs)