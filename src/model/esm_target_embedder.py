import torch
from transformers import AutoTokenizer
from transformers.models.esm import EsmModel


class ESMTargetEmbedder:
    def __init__(
        self, model_name="facebook/esm2_t30_150M_UR50D", device="cuda", batch_size=32
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embedding_dim = self.model.config.hidden_size

        self.batch_size = batch_size

    @torch.no_grad
    def get_target_embeddings(self, targets: dict[str, str]):
        sequences = list(targets.values())
        tokens = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True
        )

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        token_embeddings = self.model(**tokens).last_hidden_state
        mean_pooled = token_embeddings.mean(dim=1)
        mean_center = mean_pooled.mean(dim=0)

        mean_shifted = mean_pooled - mean_center
        normalized_embeddings = torch.nn.functional.normalize(mean_shifted)

        return {k: t for k, t in zip(targets.keys(), normalized_embeddings)}
