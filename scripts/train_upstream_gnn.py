import sys
from pathlib import Path

# Ensure src/ is importable when running scripts from the project root.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from upstream.train import (
    train_upstream_gnn,
    GNNTrainConfig,
)

from omegaconf import OmegaConf
import argparse


def load_config(config_file: str = None):
    cfg = OmegaConf.structured(GNNTrainConfig)
    if config_file is not None:
        yaml_cfg = OmegaConf.load(config_file)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    return OmegaConf.to_object(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)

    train_upstream_gnn(config)
