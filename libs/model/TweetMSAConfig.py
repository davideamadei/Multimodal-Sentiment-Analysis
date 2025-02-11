from transformers import PretrainedConfig
import torch


class TweetMSAConfig(PretrainedConfig):
    model_type = "multimodal-sentiment-analysis"

    def __init__(
        self,
        feature_extractor: str ="jina",
        feature_extractor_config: PretrainedConfig = None,
        dropout_p: float = 0.2,
        n_layers: int = 2,
        n_units: int = 512,
        weight_initialization: str = "xavier_normal",
        use_focal_loss: bool = False,
        text_only: bool = False,
        **kwargs) -> None:
      
      # TODO: clip_large is currently not working
      if feature_extractor not in ["jina", "base", "large", "siglip", "blip2"]:
        raise ValueError("Only the following models are accepted for feature extraction:\n" + "\n".join(["jina", "base", "large", "siglip", "blip2"]))
      if n_layers <= 0:
        raise ValueError("The number of layers must be a positive integer")
      if n_units <= 0:
        raise ValueError("The number of units must be a positive integer")
      
      if dropout_p > 1 or dropout_p < 0:
        raise ValueError("Dropout rate must be between 0 and 1")

      self.feature_extractor_name_simple = feature_extractor
      self.feature_extractor_name = TweetMSAConfig.get_feature_extractor_name(feature_extractor)
      self.feature_extractor_config = feature_extractor_config

      self.dropout_p = dropout_p

      self.n_layers = n_layers
      self.n_units = n_units

      self.weight_initialization = weight_initialization

      self.use_focal_loss = use_focal_loss

      self.text_only = text_only

      super().__init__(**kwargs)

    @staticmethod
    def get_feature_extractor_name(feature_extractor):
      return {"jina" : "jinaai/jina-clip-v1", 
              "base" : "openai/clip-vit-base-patch32", 
              "large" : "openai/clip-vit-large-patch14", 
              "siglip" : "google/siglip-so400m-patch16-256-i18n", 
              "blip2": "Salesforce/blip2-itm-vit-g-coco"}[feature_extractor]