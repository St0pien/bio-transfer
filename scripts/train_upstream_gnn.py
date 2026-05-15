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
    print(config)

    train_upstream_gnn(config)
