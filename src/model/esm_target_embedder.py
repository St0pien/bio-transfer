import torch
from transformers import AutoTokenizer
from transformers.models.esm import EsmModel


class ESMTargetEmbedder:
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embedding_dim = self.model.config.hidden_size

    @torch.no_grad()
    def get_target_embeddings(self, targets: dict[str, str]):
        sequences = list(targets.values())
        tokens = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        )

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        token_embeddings = self.model(**tokens).last_hidden_state
        mean_pooled = token_embeddings.mean(dim=1)

        return {k: t for k, t in zip(targets.keys(), mean_pooled)}
