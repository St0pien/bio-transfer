import json
from pathlib import Path

from tqdm import tqdm

from data.config import UPSTREAM_TARGETS
from model.esm_target_embedder import ESMTargetEmbedder
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from data.gnn_dataset import GNNDataset
from data.downloading import fetch_uniprot_from_chembl, fetch_protein_seqeuence
from torch_geometric.loader.dataloader import DataLoader
from model.multi_target_gnn import MultiTargetGINE

# seq_dir = Path("data", "sequences")
# with open(seq_dir / "upstream.json") as f:
#     targets = json.load(f)


# embedder = ESMTargetEmbedder()


# outputs = embedder.get_target_embeddings(targets)
# mean = torch.stack(list(outputs.values())).mean(dim=0)
# print(mean)

# for k, t in outputs.items():
#     embedding = t - mean
#     embedding = embedding / embedding.norm()


# smiles = "CCCN(CCC)C(=O)c1cccc(C(=O)N[C@@H](CC(C)C)[C@H](N)C[C@@H](C)C(=O)N[C@H](C(=O)NCc2ccccc2)C(C)C)c1"

# mol = Chem.MolFromSmiles(smiles)

# for a in mol.GetAtoms():
#     print(a.GetSymbol())


target_embedder = ESMTargetEmbedder()

dataset = GNNDataset.from_csv(
    "data/splits/upstream/42/full/None/upstream_full_raw_train.csv",
    target_embedder,
    target_dict_json="data/sequences/upstream.json",
)

loader = DataLoader(dataset, batch_size=512, num_workers=0, shuffle=True)

model = MultiTargetGINE(13, 4).cuda()

params = 0
for p in model.parameters():
    params += p.numel()

print(params)

for batch in tqdm(loader):
    # target_embeddings = dataset.target_embeddings[x.target_id[x.batch]]

    target_embeddings = dataset.target_embeddings[batch.target_id]

    output = model(
        batch.x.cuda(),
        batch.edge_index.cuda(),
        batch.edge_attr.cuda(),
        batch.batch.cuda(),
        target_embeddings,
    )

    # print(output.mean())
