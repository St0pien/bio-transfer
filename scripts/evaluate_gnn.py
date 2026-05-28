import argparse

from data.gnn_dataset import GNNDataset
from model.esm_target_embedder import ESMTargetEmbedder
from model.multi_target_gnn import MultiTargetGINE
from upstream.eval import eval_upstream_gnn


def main(checkpoint_path: str, dataset_path: str, batch_size=64, device: str = "cuda"):
    gnn = MultiTargetGINE.from_pretrained(checkpoint_path).to(device)
    gnn.eval()
    esm_embedder = ESMTargetEmbedder(device=device)
    dataset = GNNDataset.from_csv(dataset_path, embedder=esm_embedder)

    results = eval_upstream_gnn(gnn, dataset, batch_size=batch_size, device=device)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the pretrained GNN checkpoint file.",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset CSV file.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during evaluation.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (e.g. 'cuda' or 'cpu').",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(**vars(args))
