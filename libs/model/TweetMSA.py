from .TweetMSAConfig import TweetMSAConfig
from transformers import PreTrainedModel, AutoModel, AutoProcessor, Blip2ForImageTextRetrieval
from torch import nn, concatenate, where
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import torch

# TODO weight initialization

class TweetMSA(PreTrainedModel):
    config_class=TweetMSAConfig
    def __init__(self, config: TweetMSAConfig) -> None:
        super().__init__(config)
        self.config=config
        # the processor of the feature extractor is loaded but is not actually used, it is only present for external access
        self.processor = AutoProcessor.from_pretrained(config.feature_extractor_name, trust_remote_code=True)
        if self.config.feature_extractor_name_simple != "blip2":
            self.feature_extractor = AutoModel.from_pretrained(config.feature_extractor_name, trust_remote_code=True)
        else:
            self.feature_extractor = Blip2ForImageTextRetrieval.from_pretrained(
                TweetMSAConfig.get_feature_extractor_name(self.config.feature_extractor_name_simple))

        self.fc_layers = nn.ModuleList()

        input_layer = nn.Sequential()
        if self.config.feature_extractor_name_simple != "blip2":
            input_layer.append(nn.Linear(self.feature_extractor.config.projection_dim*2, config.n_units))
        else:
            input_layer.append(nn.Linear(256*2, config.n_units))
        input_layer.append(nn.Dropout(config.dropout_p))
        input_layer.append(nn.LeakyReLU())
        self.fc_layers.append(input_layer)
        
        for i in range(config.n_layers-1):
            layer = nn.Sequential()
            layer.append(nn.Linear(config.n_units, config.n_units))
            layer.append(nn.Dropout(config.dropout_p))
            layer.append(nn.LeakyReLU())
            self.fc_layers.append(layer)        

        output_layer = nn.Sequential()
        output_layer.append(nn.Linear(config.n_units, 9))
        self.fc_layers.append(output_layer)

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.sigmoid = nn.Sigmoid()

        self.fc_layers.apply(self._init_weights)    
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        if self.config.feature_extractor_name_simple != "blip2":
            text_embedding = self.feature_extractor.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            image_embedding = self.feature_extractor.get_image_features(pixel_values=pixel_values)
        else:
            itm_out = self.feature_extractor(input_ids=input_ids, attention_mask=attention_mask, 
                                             pixel_values=pixel_values, use_image_text_matching_head=False, return_dict=True)
            text_embedding = itm_out.text_embeds
            image_embedding = itm_out.image_embeds[:,0,:]
        logits = concatenate((text_embedding, image_embedding), axis=-1)

        for layer in self.fc_layers:
            logits = layer(logits)

        outputs = self.sigmoid(logits)

        if labels is not None :
            loss = self.criterion(logits, labels)
            return {"loss": loss, "output": outputs}
        
        return outputs

    @staticmethod
    def preprocess_dataset(dataset, model="jina", text_column="tweet", image_column="img_path", label_column=None):
        if model not in ["jina", "base", "large", "blip2"]:
            raise ValueError("Only the following models are accepted:\n" + "\n".join(["jina", "base", "large", "blip2"]))
        
        processor = AutoProcessor.from_pretrained(TweetMSAConfig.get_feature_extractor_name(model), trust_remote_code=True)

        processed_images= []
        for img in dataset[image_column]:
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

        processed_inputs = processor(
                                        text = list(dataset[text_column]), 
                                        images = processed_images, 
                                        padding=True, truncation=True, 
                                        return_tensors="np"
                                        )

        processed_dataset = pd.DataFrame({
            "input_ids": processed_inputs["input_ids"].tolist(),
            "attention_mask": processed_inputs["attention_mask"].tolist(), 
            "pixel_values": processed_inputs["pixel_values"].tolist()
        })
        if label_column is not None:
            processed_dataset[label_column] = dataset[label_column].to_list()
        return processed_dataset