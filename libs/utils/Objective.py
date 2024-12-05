from datasets import Dataset
from libs.dataset_loader import MulTweEmoDataset
from libs.utils.ModelWrappers import TweetMSAWrapper, BertWrapper, VitWrapper
import sklearn.metrics as skm
from libs.model import TweetMSA
from transformers import AutoTokenizer, AutoImageProcessor
import requests
from PIL import Image
from io import BytesIO


class TweetMSAObjective(object):
    def __init__(self, clip_version="jina", append_captions:bool=False, mode="M", freeze_weights:bool=False, seed:int=123):
        self.train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode=mode, drop_something_else=True, test_split=None, seed=seed)
        self.val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", mode=mode, drop_something_else=True, test_split=None, seed=seed)


        if append_captions:
            self.train["tweet"] = self.train.apply(lambda x: x["tweet"] + " " + x["caption"], axis=1)
        #    self.val["tweet"] = self.val.apply(lambda x: x["tweet"] + " "  + x["caption"], axis=1)

        self.train = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=self.train, model=clip_version, text_column="tweet", label_column="labels"))
        self.val = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=self.val, model=clip_version, text_column="tweet", label_column="labels"))
        self.clip_version = clip_version
        self.freeze_weights = freeze_weights

    def __call__(self, trial):
        n_epochs = trial.suggest_int("n_epochs", 2, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=10)
        if self.clip_version == "siglip" or self.clip_version == "blip2":
            batch_size = 8
        else:
            batch_size = trial.suggest_categorical("batch_size", [8,16,32])
        n_layers = trial.suggest_int("n_layers", 1, 10)
        n_units = trial.suggest_int("n_units", 32, 1024, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 1.0)
        model = TweetMSAWrapper(n_epochs=n_epochs, warmup_steps=warmup_steps, learning_rate=learning_rate, 
                                 batch_size=batch_size, n_layers=n_layers, n_units=n_units,
                                 dropout=dropout, clip_version=self.clip_version, freeze_weights=self.freeze_weights)
        model.fit(self.train, self.train["labels"])
        predictions, results =  model.score(self.val, self.val["labels"])
        label_names = MulTweEmoDataset.get_labels()
        metrics = skm.classification_report(self.val["labels"], predictions, output_dict=True, zero_division=0, target_names=label_names)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        count = 0
        for sample in predictions:
            if 1 not in sample:
                count+=1
        trial.set_user_attr("no_prediction_samples", count)
        del model
        return results["loss"], results["f1_score"], results["exact_match"]
    
class BertObjective(object):
    
    def __init__(self, bert_version="bert-large-uncased", append_captions:bool=False, mode="M", seed=123):
        self.bert_version = bert_version
        self.train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode=mode, drop_something_else=True, test_split=None, seed=seed)
        self.val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", mode=mode, drop_something_else=True, test_split=None, seed=seed)
        
        if append_captions:
            self.train["tweet"] = self.train.apply(lambda x: x["tweet"] + " " + x["caption"], axis=1)
            # self.val["tweet"] = self.val.apply(lambda x: x["tweet"] + " "  + x["caption"], axis=1)
        else:
            self.train = self.train.drop_duplicates(subset=["id"])
            self.val = self.val[~self.val.id.isin(self.train.id.values)]


        self.train = Dataset.from_pandas(self.train)
        self.val = Dataset.from_pandas(self.val)

        self.train = self.train.map(self._preprocess_data, batched=True, remove_columns=[col for col in self.train.column_names if col != "labels"])
        
        self.val = self.val.map(self._preprocess_data, batched=True, remove_columns=[col for col in self.val.column_names if col != "labels"])


    def __call__(self, trial):
        n_epochs = trial.suggest_int("n_epochs", 2, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=10)
        batch_size = trial.suggest_categorical("batch_size", [8,16,32])

        model = BertWrapper(n_epochs=n_epochs, warmup_steps=warmup_steps, learning_rate=learning_rate, batch_size=batch_size)
        
        model.fit(self.train, self.train["labels"])
        predictions, results =  model.score(self.val, self.val["labels"])
        label_names = MulTweEmoDataset.get_labels()
        metrics = skm.classification_report(self.val["labels"], predictions, output_dict=True, zero_division=0, target_names=label_names)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        count = 0
        for sample in predictions:
            if 1 not in sample:
                count+=1
        trial.set_user_attr("no_prediction_samples", count)
        del model
        return results["loss"], results["f1_score"], results["exact_match"]
    
    def _preprocess_data(self, examples):
        tokenizer = AutoTokenizer.from_pretrained(self.bert_version)
        text = examples["tweet"]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)        
        return encoding
    


class VitObjective(object):
    
    def __init__(self, vit_version="google/vit-large-patch16-224-in21k", mode="M", seed=123):
        self.vit_version = vit_version
        self.train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode=mode, drop_something_else=True, test_split=None, seed=seed)
        self.val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", mode=mode, drop_something_else=True, test_split=None, seed=seed)
        
        self.train = Dataset.from_pandas(self.train)
        self.val = Dataset.from_pandas(self.val)

        self.train = self.train.map(self._preprocess_data, batched=True, remove_columns=[col for col in self.train.column_names if col != "labels"])
        
        self.val = self.val.map(self._preprocess_data, batched=True, remove_columns=[col for col in self.val.column_names if col != "labels"])


    def __call__(self, trial):
        n_epochs = trial.suggest_int("n_epochs", 2, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=10)
        batch_size = trial.suggest_categorical("batch_size", [8,16,32])

        model = VitWrapper(n_epochs=n_epochs, warmup_steps=warmup_steps, learning_rate=learning_rate, batch_size=batch_size)
        
        model.fit(self.train, self.train["labels"])
        predictions, results =  model.score(self.val, self.val["labels"])
        label_names = MulTweEmoDataset.get_labels()
        metrics = skm.classification_report(self.val["labels"], predictions, output_dict=True, zero_division=0, target_names=label_names)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        count = 0
        for sample in predictions:
            if 1 not in sample:
                count+=1
        trial.set_user_attr("no_prediction_samples", count)
        del model
        return results["loss"], results["f1_score"], results["exact_match"]
    
    def _preprocess_data(self, examples):
        processor = AutoImageProcessor.from_pretrained(self.vit_version)
        images = examples["img_path"]
        processed_images= []
        for img in images:
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
        encoding = processor(processed_images)        
        return encoding
    

# class CVObjective(object):
#     def __init__(self, clip_version="jina", append_captions:bool=False, freeze_weights:bool=False, cv_folds:int=4, seed:int=123):
#         self.cv_folds = KFold(cv_folds, shuffle=True, random_state=seed)
#         self.dataset, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None)
#         if append_captions:
#             self.dataset["tweet"] = self.dataset.apply(lambda x: x["tweet"] + " " +  x["caption"], axis=1)

#         self.dataset = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=self.dataset, model=clip_version, text_column="tweet", label_column="labels"))
#         self.clip_version = clip_version
#         self.freeze_weights = freeze_weights

#     def __call__(self, trial):
#         n_epochs = trial.suggest_int("n_epochs", 2, 10)
#         learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
#         warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=10)
#         batch_size = trial.suggest_categorical("batch_size", [8,16,32])
#         n_layers = trial.suggest_int("n_layers", 1, 10)
#         n_units = trial.suggest_int("n_units", 32, 1024, log=True)
#         dropout = trial.suggest_float("dropout", 0.0, 1.0)
#         metrics_keys = ["loss", "f1_score", "exact_match"]
#         results = dict.fromkeys(metrics_keys, 0)
#         for i, (train_index, val_index) in enumerate(self.cv_folds.split(self.dataset)):
#             train = self.dataset.select(train_index)
#             val = self.dataset.select(val_index)
#             model = TweetMSAWrapper(n_epochs=n_epochs, warmup_steps=warmup_steps, learning_rate=learning_rate, 
#                                  batch_size=batch_size, n_layers=n_layers, n_units=n_units,
#                                  dropout=dropout, clip_version=self.clip_version, freeze_weights=self.freeze_weights)
#             model.fit(train, train["labels"])
#             _, fold_results =  model.score(val, val["labels"])
#             for key in metrics_keys:
#                 results[key] = results[key] + fold_results[key]
#             del model
#             results = {key: (value)/self.cv_folds.get_n_splits() for key, value in results.items()}
#         return tuple(results[key] for key in metrics_keys)
