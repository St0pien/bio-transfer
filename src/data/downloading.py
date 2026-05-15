import json
from pathlib import Path

import requests
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

DEFAULT_STANDARD_TYPES = ["IC50", "Ki", "Kd", "EC50"]

activity_client = new_client.activity


def fetch_data_from_chembl(
    target_chembl_id: str,
    standard_types=DEFAULT_STANDARD_TYPES,
    data_dir="data/chembl_raw",
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / f"{target_chembl_id}.json"

    if save_path.exists():
        with open(save_path) as f:
            data = json.load(f)

        return data

    records = []

    for stype in standard_types:
        res = activity_client.filter(
            target_chembl_id=target_chembl_id,
            standard_type=stype,
            standard_relation="=",
        ).only(
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "standard_type",
            "pchembl_value",
            "assay_chembl_id",
        )

        batch = list(res)
        records.extend(batch)

    with open(save_path, "w") as f:
        json.dump(records, f)

    return records


def fetch_uniprot_from_chembl(chembl_id: str):
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
        response = requests.get(url)
        data = response.json()
    except:
        print(chembl_id, response.text[:100])

    return data["target_components"][0]["accession"]


def download_alphafold_pdb(uniprot_id: str, filename: str):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "w") as f:
            f.write(response.text)
        return filename
    else:
        raise Exception(f"Alphafold structure not found! uniprot_id: {uniprot_id}")


def fetch_protein_seqeuence(uniprot_id: str):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url)
    r.raise_for_status()
    txt = r.text

    return "".join(txt.splitlines()[1:])


def fetch_targets_sequences(target_ids):
    target_sequences = {}
    for chembl_id in tqdm(target_ids, desc="Downloading target protein sequences"):
        target_sequences[chembl_id] = fetch_protein_seqeuence(
            fetch_uniprot_from_chembl(chembl_id)
        )

    return target_sequences
