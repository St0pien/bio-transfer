import subprocess, sys
import os, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# Konfiguracja
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

os.makedirs("./data/downstream", exist_ok=True)
os.makedirs("./data/upstream", exist_ok=True)

activity_client = new_client.activity

# ── Cele downstream – WYKLUCZONE z upstreamu ────────────────────────────────
DOWNSTREAM_TARGETS = {
    "BACE1": "CHEMBL4822",
    "TYK2":  "CHEMBL3553",
    "A2a":   "CHEMBL251",
}

# ── Cele upstream – farmakologicznie istotne, różnorodne ────────────────────
# Wybrane tak żeby pokrywały różne klasy białek (kinazy, GPCRy, proteazy, etc.)
# i żeby dawały dobry transfer do BACE1/TYK2/A2a
UPSTREAM_TARGETS = {
    # Kinazy (podobne do TYK2)
    "JAK1":     "CHEMBL2835",
    "JAK2":     "CHEMBL2971",
    "JAK3":     "CHEMBL5250",
    "EGFR":     "CHEMBL203",
    "CDK2":     "CHEMBL301",
    "p38alpha":  "CHEMBL260",
    # Proteazy aspartylowe (podobne do BACE1)
    "BACE2":    "CHEMBL3717",
    "Cathepsin_D": "CHEMBL1865",
    "Renin":    "CHEMBL286",
    "Pepsin":   "CHEMBL3880",
    # Receptory adenozynowe i GPCRy (podobne do A2a)
    "A1":       "CHEMBL226",
    "A2b":      "CHEMBL254",
    "A3":       "CHEMBL252",
    "DRD2":     "CHEMBL217",   # receptor dopaminowy D2
    "5HT2A":    "CHEMBL224",   # receptor serotoninowy
    # Inne
    "HDAC1":    "CHEMBL325",
    "PARP1":    "CHEMBL3105",
    "HSP90":    "CHEMBL3880757",
}

# --- Funkcje pomocnicze ---
def fetch_bioactivity(target_chembl_id: str, max_records: int = None) -> pd.DataFrame:
    standard_types = ["IC50", "Ki", "Kd", "EC50"]
    records = []
    # Definiujemy wymagane kolumny na wypadek pustego wyniku
    required_cols = ["molecule_chembl_id", "canonical_smiles", "standard_value", "pchembl_value"]
    
    for stype in standard_types:
        try:
            res = activity_client.filter(
                target_chembl_id=target_chembl_id,
                standard_type=stype,
                standard_relation="=",
            ).only(
            "molecule_chembl_id", "canonical_smiles",
            "standard_value", "standard_units",
            "standard_type", "pchembl_value",
            "assay_chembl_id",
        )
            batch = list(res)
            if max_records:
                batch = batch[:max_records]
            records.extend(batch)
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=required_cols)
    return pd.DataFrame.from_records(records)

def is_valid_smiles(smi) -> bool:
    try:
        if pd.isna(smi): return False
        return Chem.MolFromSmiles(str(smi)) is not None
    except:
        return False

def clean_bioactivity_df(df: pd.DataFrame,
                          smiles_col="canonical_smiles",
                          value_col="standard_value") -> pd.DataFrame:
    if df.empty:
        return df
    
    # Sprawdzenie czy kolumny istnieją (zabezpieczenie przed specyficznymi błędami API)
    if smiles_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=df.columns)

    df = df.copy()
    # 1. Usuń NaN
    df = df.dropna(subset=[smiles_col, value_col])

    # 2. Konwersja na float

    if df.empty: 
        return df

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if "pchembl_value" in df.columns:
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=[value_col])

    # 3. Walidacja SMILES
    mask = df[smiles_col].apply(is_valid_smiles)
    df = df[mask]

    # 4. Deduplikacja
    return df.drop_duplicates(subset=[smiles_col, value_col]).reset_index(drop=True)

def smiles_to_fp(smi: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- 1. Pobieranie i czyszczenie danych ---
upstream_parts = []
for name, chembl_id in tqdm(UPSTREAM_TARGETS.items(), desc="Pobieranie upstream"):
    df_raw = fetch_bioactivity(chembl_id)
    df_cleaned = clean_bioactivity_df(df_raw)
    
    if not df_cleaned.empty:
        df_cleaned["target_name"], df_cleaned["target_chembl_id"] = name, chembl_id
        upstream_parts.append(df_cleaned)
    else:
        print(f" ⚠️  Brak danych dla: {name} ({chembl_id}) - pomijam.")

if not upstream_parts:
    raise ValueError("Nie pobrano żadnych danych upstream! Sprawdź połączenie z ChEMBL API.")

df_upstream = pd.concat(upstream_parts, ignore_index=True)

downstream_clean = {}
for name, chembl_id in DOWNSTREAM_TARGETS.items():
    df_raw = fetch_bioactivity(chembl_id)
    df_cleaned = clean_bioactivity_df(df_raw)
    if not df_cleaned.empty:
        df_cleaned["target_name"], df_cleaned["target_chembl_id"] = name, chembl_id
        downstream_clean[name] = df_cleaned
    else:
        print(f" ⚠️  Brak danych dla celu downstream: {name}!")


# --- 2. Filtrowanie Leakage (Usuwanie związków obecnych w Downstream z Upstream) ---
downstream_smiles = set().union(*(df["canonical_smiles"] for df in downstream_clean.values()))
downstream_ids = set().union(*(df["molecule_chembl_id"] for df in downstream_clean.values()))
ds_target_ids = set(DOWNSTREAM_TARGETS.values())

mask = (~df_upstream["canonical_smiles"].isin(downstream_smiles)) & \
       (~df_upstream["molecule_chembl_id"].isin(downstream_ids)) & \
       (~df_upstream["target_chembl_id"].isin(ds_target_ids))

df_upstream_filtered = df_upstream[mask].reset_index(drop=True)

# --- 3. Obliczanie Fingerprintów (ECFP4) ---
def process_and_save_fps(df, path_npy, smiles_col="canonical_smiles"):
    fps = []
    valid_idx = []
    for i, smi in enumerate(df[smiles_col]):
        fp = smiles_to_fp(str(smi))
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    fps_arr = np.array(fps)
    np.save(path_npy, fps_arr)
    return df.iloc[valid_idx].reset_index(drop=True), fps_arr

df_upstream_filtered, _ = process_and_save_fps(df_upstream_filtered, "./data/upstream/fps_upstream.npy")
df_upstream_filtered.to_csv("./data/upstream/upstream_filtered.csv", index=False)

# --- 4. Labeling Downstream (pChEMBL >= 6.0 jako aktywne) ---
PCHEMBL_THRESHOLD = 6.0 # pIC50 ≥ 6 → aktywny (IC50 ≤ 1 µM)
for name, df in downstream_clean.items():
    if "pchembl_value" in df.columns:
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.dropna(subset=["pchembl_value"])
        df["active"] = (df["pchembl_value"] >= PCHEMBL_THRESHOLD).astype(int)
    else:
        df["active"] = (df["standard_value"] <= 1000).astype(int)
    
    df, _ = process_and_save_fps(df, f"./data/downstream/fps_{name}.npy")
    df.to_csv(f"./data/downstream/{name}_labeled.csv", index=False)

print("✅ Preprocessing zakończony. Dane zapisane w ./data/")