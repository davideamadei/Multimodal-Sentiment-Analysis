from transformers import PretrainedConfig
import torch
class MMSAConfig(PretrainedConfig):
    model_type = "MMSA"

    def __init__(
        self,
        feature_extractor="jinaai/jina-clip-v1",
        **kwargs):
      if feature_extractor not in ["jinaai/jina-clip-v1", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]:
        raise ValueError("Only the following models are accepted:\n" + "\n".join(["jinaai/jina-clip-v1", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]))
      self.feature_extractor_name = feature_extractor
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      super().__init__(**kwargs)