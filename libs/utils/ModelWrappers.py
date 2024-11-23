from libs.model import TweetMSA, TweetMSAConfig
from torch import manual_seed
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification, AutoModelForSequenceClassification
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import sklearn.metrics as skm
from datasets import Dataset
import torch

#TODO: parameter checks
#TODO: documentation
class TweetMSAWrapper(BaseEstimator):

    def __init__(self, learning_rate:float=1e-5, batch_size:int=16, n_epochs:int=10,
                 n_layers:int=2, n_units:int=512, dropout:float=0.2, 
                 warmup_steps:int=100, clip_version:str="jina", freeze_weights=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.clip_version = clip_version
        self.freeze_weights = freeze_weights

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X, y):
        self._create_model(X)
        self._trainer.train()
        self._is_fitted = True
        return self
    
    def score(self, X, y):
        check_is_fitted(self)
        preds=self.predict(X)
    
        y_pred = preds.predictions > 0.5
        metrics_dict = {}

        metrics_dict["loss"] = preds.metrics["test_loss"]
        metrics_dict["exact_match"] = skm.accuracy_score(y, y_pred, normalize=True, sample_weight=None)
        metrics_dict["recall"] = skm.recall_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["precision"] = skm.precision_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["f1_score"] = skm.f1_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        return y_pred, metrics_dict

    def predict(self, X):
        check_is_fitted(self)
        return self._trainer.predict(X)

#TODO: add interface to load checkpoint
    def _create_model(self, X):
        config = TweetMSAConfig(feature_extractor=self.clip_version, layers=self.n_layers, n_units=self.n_units, dropout_p=self.dropout)
        model = TweetMSA(config).cuda()
        if self.freeze_weights:
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        args = TrainingArguments(
                report_to="none",
                save_strategy="no", 
                eval_strategy="no", 
                logging_strategy="no",

                output_dir="./ckp/",

                # necessary to save models
                save_safetensors=False,

                bf16=True,

                learning_rate=self.learning_rate,
                num_train_epochs=self.n_epochs,
                warmup_steps=self.warmup_steps,
                per_device_train_batch_size=self.batch_size, 

                push_to_hub=False,
                disable_tqdm=True,
                bf16_full_eval=True
                )

        trainer = Trainer(
            model=model,
            train_dataset=X,
            args=args,
            tokenizer=None,
            )
        self._trainer = trainer



class BertWrapper(BaseEstimator):

    def __init__(self, bert_version="bert-base-uncased", learning_rate:float=1e-5, batch_size:int=16, n_epochs:int=10, warmup_steps:int=100):
        super().__init__()
        self.bert_version = bert_version
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X, y):
        self._create_model(X)
        self._trainer.train()
        self._is_fitted  = True
        return self
    
    def score(self, X, y):
        check_is_fitted(self)
        metrics_dict = {}
        preds=self.predict(X)
        metrics_dict["loss"] = preds.metrics["test_loss"]
        preds = torch.nn.functional.sigmoid(torch.Tensor(preds.predictions))
        y_pred = preds > 0.5

        metrics_dict["exact_match"] = skm.accuracy_score(y, y_pred, normalize=True, sample_weight=None)
        metrics_dict["recall"] = skm.recall_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["precision"] = skm.precision_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["f1_score"] = skm.f1_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        return y_pred, metrics_dict

    def predict(self, X):
        check_is_fitted(self)
        predictions = self._trainer.predict(X)
        return predictions

    def _create_model(self, X):
        model = AutoModelForSequenceClassification.from_pretrained(self.bert_version, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(X["labels"][0])
                                                           )
        args = TrainingArguments(
                report_to="none",
                save_strategy="no", 
                eval_strategy="no", 
                logging_strategy="no",

                output_dir="./ckp/",

                # necessary to save models
                save_safetensors=False,

                bf16=True,

                learning_rate=self.learning_rate,
                num_train_epochs=self.n_epochs,
                warmup_steps=self.warmup_steps,
                per_device_train_batch_size=self.batch_size, 

                push_to_hub=False,
                disable_tqdm=True,
                )

        trainer = Trainer(
            model=model,
            train_dataset=X,
            args=args,
            tokenizer=None,
            )
        self._trainer = trainer



class VitWrapper(BaseEstimator):

    def __init__(self, vit_version="google/vit-base-patch16-224", learning_rate:float=1e-5, batch_size:int=16, n_epochs:int=10, warmup_steps:int=100):
        super().__init__()
        self.vit_version = vit_version
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X, y):
        self._create_model(X)
        self._trainer.train()
        self._is_fitted  = True
        return self
    
    def score(self, X, y):
        check_is_fitted(self)
        metrics_dict = {}
        preds=self.predict(X)
        metrics_dict["loss"] = preds.metrics["test_loss"]
        preds = torch.nn.functional.sigmoid(torch.Tensor(preds.predictions))
        y_pred = preds > 0.5

        metrics_dict["exact_match"] = skm.accuracy_score(y, y_pred, normalize=True, sample_weight=None)
        metrics_dict["recall"] = skm.recall_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["precision"] = skm.precision_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["f1_score"] = skm.f1_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        return y_pred, metrics_dict

    def predict(self, X):
        check_is_fitted(self)
        predictions = self._trainer.predict(X)
        return predictions

    def _create_model(self, X):
        model = AutoModelForImageClassification.from_pretrained(self.vit_version, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(X["labels"][0]),
                                                           ignore_mismatched_sizes=True
                                                           )
        args = TrainingArguments(
                report_to="none",
                save_strategy="no", 
                eval_strategy="no", 
                logging_strategy="no",

                output_dir="./ckp/",

                # necessary to save models
                save_safetensors=False,

                bf16=True,

                learning_rate=self.learning_rate,
                num_train_epochs=self.n_epochs,
                warmup_steps=self.warmup_steps,
                per_device_train_batch_size=self.batch_size, 

                push_to_hub=False,
                disable_tqdm=True,
                )

        trainer = Trainer(
            model=model,
            train_dataset=X,
            args=args,
            tokenizer=None,
            )
        self._trainer = trainer
