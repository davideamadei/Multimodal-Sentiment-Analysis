from .TweetMSAConfig import TweetMSAConfig
from transformers import PreTrainedModel, AutoModel, AutoProcessor, PretrainedConfig
from torch import nn, concatenate
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO


# TODO weight initialization

class TweetMSA(PreTrainedModel):
    def __init__(self, config: TweetMSAConfig) -> None:
        super().__init__(config)

        # Load model directly
        self.processor = AutoProcessor.from_pretrained(config.feature_extractor_name, trust_remote_code=True)
        
        self.feature_extractor = AutoModel.from_pretrained(config.feature_extractor_name, trust_remote_code=True)

        # TODO: polish structure of classifier
        self.fc_layers = nn.ModuleList()

        input_layer = nn.Sequential()
        input_layer.append(nn.Linear(self.feature_extractor.config.projection_dim*2, 512))
        input_layer.append(nn.Dropout(config.dropout_p))
        input_layer.append(nn.LeakyReLU())
        self.fc_layers.append(input_layer)
        
        for i in range(config.n_layers):
            layer = nn.Sequential()
            layer.append(nn.Linear(512, 512))
            layer.append(nn.Dropout(config.dropout_p))
            layer.append(nn.LeakyReLU())
            self.fc_layers.append(layer)        

        output_layer = nn.Sequential()
        output_layer.append(nn.Linear(512, 10))
        output_layer.append(nn.Sigmoid())
        self.fc_layers.append(output_layer)

        self.criterion = nn.BCELoss()
        
        self.to(self.device)
    
    def preprocess(self, text_inputs, image_inputs):
        processed_images= []
        for img in image_inputs:
            if isinstance(img, str):
                if img.startswith('http'):
                    response = requests.get(img)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    image = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                image = img.convert('RGB')
            else:
                raise ValueError("Unsupported image format")

            processed_images.append(image)


        processed_inputs = self.processor(text = text_inputs, images = processed_images, padding=True)
        return processed_inputs

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_embedding = self.feature_extractor.get_text_features(input_ids=input_ids, attention_mask=attention_mask).to(self.device)
        image_embedding = self.feature_extractor.get_image_features(pixel_values=pixel_values).to(self.device)

        logits = concatenate((text_embedding, image_embedding), axis=-1)

        for layer in self.fc_layers:
            logits = layer(logits)

        if labels is not None :
            loss = self.criterion(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return logits