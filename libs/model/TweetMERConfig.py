from transformers import PretrainedConfig


class TweetMERConfig(PretrainedConfig):
    "Class storing the configuration of TweetMERModel"
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
        n_classes: int = 9,
        **kwargs) -> None:
      """_summary_

      Parameters
      ----------
      feature_extractor : str, optional
          which feature extractor to use, by default "jina"
      feature_extractor_config : PretrainedConfig, optional
          the configuration class of the feature extractor, by default None
      dropout_p : float, optional
          parameter controlling the dropout of hidden layers, by default 0.2
      n_layers : int, optional
          number of hidden layers, by default 2
      n_units : int, optional
          number of units in each hidden layer, by default 512
      weight_initialization : str, optional
          weight initialization method to use, by default "xavier_normal"
      use_focal_loss : bool, optional
          flag controlling the usage of the focal loss, by default False
      text_only : bool, optional
          flag controlling wether to use only text or both image and text, by default False
      n_classes : int, optional
          number of classes that the model has to distinguish between, by default 9

      Raises
      ------
      ValueError
          when invalid inputs are given
      """
      
      if feature_extractor not in ["jina", "base", "large", "siglip", "blip2"]:
        raise ValueError("Only the following models are accepted for feature extraction:\n" + "\n".join(["jina", "base", "large", "siglip", "blip2"]))
      if n_layers <= 0:
        raise ValueError("The number of layers must be a positive integer")
      if n_units <= 0:
        raise ValueError("The number of units must be a positive integer")
      
      if dropout_p > 1 or dropout_p < 0:
        raise ValueError("Dropout rate must be between 0 and 1")

      self.feature_extractor_name_simple = feature_extractor
      self.feature_extractor_name = TweetMERConfig.get_feature_extractor_name(feature_extractor)
      self.feature_extractor_config = feature_extractor_config

      self.dropout_p = dropout_p

      self.n_layers = n_layers
      self.n_units = n_units

      self.weight_initialization = weight_initialization

      self.use_focal_loss = use_focal_loss

      self.text_only = text_only
      self.n_classes = n_classes
      super().__init__(**kwargs)

    @staticmethod
    def get_feature_extractor_name(feature_extractor):
      return {"jina" : "jinaai/jina-clip-v1", 
              "base" : "openai/clip-vit-base-patch32", 
              "large" : "openai/clip-vit-large-patch14", 
              "siglip" : "google/siglip-so400m-patch16-256-i18n", 
              "blip2": "Salesforce/blip2-itm-vit-g-coco"}[feature_extractor]