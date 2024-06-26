from transformers import PretrainedConfig
import torch


class TweetMSAConfig(PretrainedConfig):
    model_type = "multimodal-sentiment-analysis"

    def __init__(
        self,
        feature_extractor: str ="jinaai/jina-clip-v1",
        feature_extractor_config: PretrainedConfig = None,
        device: str = None,
        **kwargs) -> None:
      
      if feature_extractor not in ["jinaai/jina-clip-v1", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]:
        raise ValueError("Only the following models are accepted:\n" + "\n".join(["jinaai/jina-clip-v1", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]))
      
      self.feature_extractor_name = feature_extractor
      self.feature_extractor_config = feature_extractor_config
      
      if device is None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      elif device == "cpu":
        self.device = torch.device("cpu")
      elif device == "gpu":
        if torch.cuda.is_available():
          self.device = torch.device("cuda:0")
        else:
          print("A cuda device is not available, the model will be initialized on CPU")
          self.device = torch.device("cpu")
      else:
        raise ValueError("cpu and gpu are the only values accepted")
      super().__init__(**kwargs)