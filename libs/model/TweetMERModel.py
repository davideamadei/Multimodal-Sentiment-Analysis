from .TweetMERConfig import TweetMERConfig
from transformers import PreTrainedModel, AutoModel, AutoProcessor, Blip2ForImageTextRetrieval
from torch import nn, concatenate, Tensor
import pandas as pd
from PIL import Image
import requests
from io import BytesIO


class TweetMERModel(PreTrainedModel):
    "Class implementing a model for Tweet Multimodal Emotion Recognition"
    config_class=TweetMERConfig
    def __init__(self, config: TweetMERConfig) -> None:
        """init function

        Parameters
        ----------
        config : TweetMSAConfig
            config class for values of different parameters of the model
        """
        super().__init__(config)
        self.config=config
        # the processor of the feature extractor is loaded but is not actually used, it is only present for external access
        self.processor = AutoProcessor.from_pretrained(config.feature_extractor_name, trust_remote_code=True)
        if self.config.feature_extractor_name_simple != "blip2":
            self.feature_extractor = AutoModel.from_pretrained(config.feature_extractor_name, trust_remote_code=True)
        else:
            self.feature_extractor = Blip2ForImageTextRetrieval.from_pretrained(
                TweetMERConfig.get_feature_extractor_name(self.config.feature_extractor_name_simple))

        self.fc_layers = nn.ModuleList()

        input_layer = nn.Sequential()

        n_inputs = 2
        if self.config.text_only:
            n_inputs = 1

        # initialize the input layer of the FC classification head depending on the used feature extractor
        if self.config.feature_extractor_name_simple == "blip2":
            input_layer.append(nn.Linear(256*n_inputs, config.n_units))
        elif self.config.feature_extractor_name_simple == "siglip":
            input_layer.append(nn.Linear(1152*n_inputs, config.n_units))
        else:
            input_layer.append(nn.Linear(self.feature_extractor.config.projection_dim*n_inputs, config.n_units))

        input_layer.append(nn.Dropout(config.dropout_p))
        input_layer.append(nn.LeakyReLU())
        self.fc_layers.append(input_layer)
        
        # add the hidden layers to the model
        for i in range(config.n_layers-1):
            layer = nn.Sequential()
            layer.append(nn.Linear(config.n_units, config.n_units))
            layer.append(nn.Dropout(config.dropout_p))
            layer.append(nn.LeakyReLU())
            self.fc_layers.append(layer)        

        # add the output layer to the model
        output_layer = nn.Sequential()
        output_layer.append(nn.Linear(config.n_units, self.config.n_classes))
        self.fc_layers.append(output_layer)

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        self.sigmoid = nn.Sigmoid()

        self.fc_layers.apply(self._init_weights)    
    
    def _init_weights(self, m:nn.Linear):
        """Function to initialize the weights of a layer

        Parameters
        ----------
        m : nn.Linear
            layer to initialize the weights of
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input_ids:Tensor, pixel_values:Tensor, attention_mask:Tensor=None, labels:Tensor=None)->Tensor|dict:
        """Forward call of the model

        Parameters
        ----------
        input_ids : Tensor
            embeddings of the input text
        pixel_values : Tensor
            embeddings of the input image
        attention_mask : Tensor, optional
            attention mask of the input text, by default None
        labels : Tensor, optional
            labels of the input data, by default None

        Returns
        -------
        Tensor|dict
            The outputs of the model. If the data in input has labels the loss is also returned in a dict with the outputs
        """
        # encode inputs with feature extractor
        if self.config.feature_extractor_name_simple != "blip2":
            text_embedding = self.feature_extractor.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            if not self.config.text_only:
                image_embedding = self.feature_extractor.get_image_features(pixel_values=pixel_values)
        else:
            itm_out = self.feature_extractor(input_ids=input_ids, attention_mask=attention_mask, 
                                             pixel_values=pixel_values, use_image_text_matching_head=False, return_dict=True)
            text_embedding = itm_out.text_embeds
            image_embedding = itm_out.image_embeds.mean(dim=1)

        if self.config.text_only:
            logits = text_embedding
        else:
            logits = concatenate((text_embedding, image_embedding), axis=-1)

        # give features to FC network
        for layer in self.fc_layers:
            logits = layer(logits)

        # apply sigmoid for final outputs
        outputs = self.sigmoid(logits)

        # compute loss if labels are given in input
        if labels is not None :
            loss = self.criterion(logits, labels)
            if self.config.use_focal_loss:
                p_t = outputs * labels + (1-outputs) * (1-labels)
                loss = loss*((1-p_t) ** 2)
            loss = loss.mean()
            return {"loss": loss, "output": outputs}
        
        return outputs

    @staticmethod
    def preprocess_dataset(dataset:pd.DataFrame, model="jina", text_column="tweet", image_column="img_path", label_column:str=None)->pd.DataFrame:
        """Static function to preprocess dataset so they can be given in input to to the model

        Parameters
        ----------
        dataset : pd.DataFrame
            dataset to preprocess
        model : str, optional
            the model to use the Processor of, by default "jina"
        text_column : str, optional
            the column containing the text to use for the processed dataset, by default "tweet"
        image_column : str, optional
            the column containing the paths of the images, by default "img_path"
        label_column : str, optional
            the column containing the labels, by default None

        Returns
        -------
        pd.DataFrame
            the processed dataset usable as input of the model

        Raises
        ------
        ValueError
            when invalid inputs are given
        """
        if model not in ["jina", "base", "large", "siglip", "blip2"]:
            raise ValueError("Only the following models are accepted:\n" + "\n".join(["jina", "base", "large", "blip2"]))
        
        processor = AutoProcessor.from_pretrained(TweetMERConfig.get_feature_extractor_name(model), trust_remote_code=True)

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

        processed_dataset = pd.DataFrame({key: processed_inputs[key].tolist() for key in processed_inputs.keys()})

#        processed_dataset = pd.DataFrame({
#            "input_ids": processed_inputs["input_ids"].tolist(),
#            "attention_mask": processed_inputs["attention_mask"].tolist(), 
#            "pixel_values": processed_inputs["pixel_values"].tolist()
#        })
        if label_column is not None:
            processed_dataset[label_column] = dataset[label_column].to_list()
        return processed_dataset
