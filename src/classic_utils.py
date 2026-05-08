import pandas as pd
import numpy as np
import logging
from data_utils import scaffold_split_three_way # Importujemy Twoją funkcję z GNN

def load_classic_data(target_name, mode="downstream"):
    """
    Synchronizuje fingerprinty (.npy) z podziałem Scaffold Split (.csv).
    Gwarantuje, że liczba wierszy w CSV i FPS jest identyczna.
    """
    if mode == "upstream":
        logging.info("Ładowanie danych Upstream...")
        df_long = pd.read_csv("./data/upstream/upstream_filtered.csv")
        fps = np.load("./data/upstream/fps_upstream.npy")
        
        # 1. Agregujemy etykiety do formatu szerokiego (unikalne cząsteczki)
        df_labels = df_long.pivot_table(
            index="molecule_chembl_id",
            columns="target_name",
            values="pchembl_value",
            aggfunc="mean"
        )
        y = (df_labels >= 6.0).astype(float).fillna(-1).values
        label_cols = df_labels.columns.tolist()
        
        # 2. Tworzymy czysty DataFrame z SMILES potrzebny do Scaffold Split
        # Musimy mieć pewność, że kolejność SMILES odpowiada kolejności w df_labels i FPS
        smiles_map = df_long.drop_duplicates("molecule_chembl_id").set_index("molecule_chembl_id")["canonical_smiles"]
        
        # Budujemy nowy DF w 100% zsynchronizowany z df_labels
        df_sync = pd.DataFrame(index=df_labels.index)
        df_sync["canonical_smiles"] = smiles_map
        
        # KRYTYCZNY MOMENT: Resetujemy indeks, aby mieć liczby 0, 1, 2... N-1
        # To sprawi, że indeksy z podziału będą odpowiadać wierszom w macierzy numpy
        df_for_split = df_sync.reset_index(drop=True)
        
        logging.info(f"Rozmiar po agregacji (unikalne): {len(df_for_split)}, Rozmiar FPS: {fps.shape[0]}")

        # Jeśli nadal jest różnica (np. o 1-2 rekordy przez błędy RDKit przy generowaniu FPS)
        if len(df_for_split) != fps.shape[0]:
            min_size = min(len(df_for_split), fps.shape[0])
            logging.warning(f"⚠️ Niezgodność rozmiarów! Przycinam do {min_size}")
            df_for_split = df_for_split.iloc[:min_size]
            fps = fps[:min_size]
            y = y[:min_size]
            
    else:
        # Downstream zazwyczaj nie wymaga pivot_table, ale reset_index jest kluczowy
        df_for_split = pd.read_csv(f"./data/downstream/{target_name}_labeled.csv")
        fps = np.load(f"./data/downstream/fps_{target_name}.npy")
        y = df_for_split["active"].values
        label_cols = ["active"]
        df_for_split = df_for_split.reset_index(drop=True)

    # 3. Scaffold Split na zsynchronizowanym zbiorze
    logging.info("Rozpoczynam Scaffold Split...")
    train_df, val_df, test_df = scaffold_split_three_way(df_for_split)
    
    # Wyciągamy indeksy pozycji (iloc)
    train_idx = train_df.index.tolist()
    val_idx = val_df.index.tolist()
    test_idx = test_df.index.tolist()
    
    logging.info(f"Finałowe rozmiary: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Zwracamy wycinki macierzy numpy
    return (fps[train_idx], y[train_idx]), (fps[val_idx], y[val_idx]), (fps[test_idx], y[test_idx]), label_cols