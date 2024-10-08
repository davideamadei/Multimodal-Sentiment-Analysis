from transformers import PretrainedConfig
import torch


class TweetMSAConfig(PretrainedConfig):
    model_type = "multimodal-sentiment-analysis"

    def __init__(
        self,
        feature_extractor: str ="jina",
        feature_extractor_config: PretrainedConfig = None,
        # device: str = None,
        dropout_p: float = 0.2,
        layers: tuple[int] = (512,512),
        weight_initialization:str = "xavier_normal",
        label_threshold:float = None,
        **kwargs) -> None:
      
      # TODO: clip_large is currently not working
      if feature_extractor not in ["jina", "clip_base", "clip_large"]:
        raise ValueError("Only the following models are accepted for feature extraction:\n" + "\n".join(["jina", "clip_base", "clip_large"]))
      if len(layers) <= 0:
        raise ValueError("The number of layers must be a positive integer")
      
      # if device is None:
      #   self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      # elif device == "cpu":
      #   self.device = torch.device("cpu")
      # elif device == "gpu":
      #   if torch.cuda.is_available():
      #     self.device = torch.device("cuda:0")
      #   else:
      #     print("A cuda device is not available, the model will be initialized on CPU")
      #     self.device = torch.device("cpu")
      # else:
      #   raise ValueError("cpu and gpu are the only values accepted")
      
      if dropout_p > 1 or dropout_p < 0:
        raise ValueError("Dropout rate must be between 0 and 1")
      
      if label_threshold is not None and (label_threshold > 1 or label_threshold < 0):
        raise ValueError("Threshold for labels must be between 0 and 1")


      self.feature_extractor_name = TweetMSAConfig.get_feature_extractor_name_map()[feature_extractor]
      self.feature_extractor_config = feature_extractor_config

      self.dropout_p = dropout_p

      self.layers = layers

      self.weight_initialization = weight_initialization

      self.label_threshold = label_threshold

      super().__init__(**kwargs)

    @staticmethod
    def get_feature_extractor_name_map():
      return {"jina" : "jinaai/jina-clip-v1", "clip_base" : "openai/clip-vit-base-patch32", "clip_large" : "openai/clip-vit-large-patch14"}