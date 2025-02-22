from abc import ABC, abstractmethod
from libs.model import TweetMSA, TweetMSAConfig
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification, AutoModelForSequenceClassification
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import sklearn.metrics as skm
import math
import numpy as np

class CustomWrapper(ABC, BaseEstimator):
            
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def from_pretrained(self, ckp_dir):
        self._create_model(train=None, val=None, ckp_dir=ckp_dir)

        self._is_fitted = True
        return self

    def fit(self, train, labels, val=None, ckp_dir=None, compute_metrics=None):
        self._create_model(train, val, ckp_dir=ckp_dir, compute_metrics=compute_metrics)
        self._trainer.train(resume_from_checkpoint=ckp_dir)
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
    
    def print_model(self):
        check_is_fitted(self)
        return str(self._trainer.model)

    @abstractmethod
    def _create_model(self):
        pass

#TODO: parameter checks
#TODO: documentation
class TweetMSAWrapper(CustomWrapper):

    def __init__(self, learning_rate:float=1e-5, batch_size:int=16, n_epochs:int=10,
                n_layers:int=2, n_units:int=512, dropout:float=0.2, 
                warmup_steps:int=100, clip_version:str="jina", freeze_weights=False, text_only=False,
                seed=123, output_dir:str=None, run_name:str=None):
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
        self.text_only = text_only
        self.output_dir = output_dir
        self.run_name = run_name
        self.seed = seed

    def _create_model(self, train, val=None, ckp_dir=None, compute_metrics=None):
        config = TweetMSAConfig(
            feature_extractor=self.clip_version,
            layers=self.n_layers,
            n_units=self.n_units,
            dropout_p=self.dropout,
            text_only=self.text_only)
        
        if ckp_dir:
            model = TweetMSA.from_pretrained(ckp_dir).cuda()
        else:
            model = TweetMSA(config).cuda()
        if self.freeze_weights:
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

        output_dir = ".ckp/"
        if self.output_dir:
            output_dir += self.output_dir

        args = TrainingArguments(
            report_to="none" if not self.run_name else "wandb",
            run_name=self.run_name,
            resume_from_checkpoint=False if not ckp_dir else self.output_dir,
            save_strategy="no" if not self.output_dir else "epoch", 
            eval_strategy="no" if not val else "epoch", 
            logging_strategy="no" if not self.output_dir else "steps",
            
            save_total_limit=10,
            logging_steps=1,

            output_dir=output_dir,

            # necessary to save models
            save_safetensors=True,

            bf16=True,

            learning_rate=self.learning_rate,
            num_train_epochs=self.n_epochs,
            warmup_steps=self.warmup_steps,
            per_device_train_batch_size=self.batch_size, 

            push_to_hub=False,
            disable_tqdm=True if not self.output_dir else False,
            bf16_full_eval=True,
            load_best_model_at_end=False,
            seed=self.seed
            )

        trainer = Trainer(
            model=model,
            train_dataset=train,
            eval_dataset=val,
            args=args,
            tokenizer=None,
            compute_metrics=compute_metrics
            )
        self._trainer = trainer



class BertWrapper(CustomWrapper):

    def __init__(self, bert_version="bert-large-uncased", learning_rate:float=1e-5,
                batch_size:int=16, n_epochs:int=10, warmup_steps:int=100,
                seed=123, output_dir:str=None, run_name:str=None):
        super().__init__()
        self.bert_version = bert_version
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.run_name = run_name
        self.seed = seed

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def score(self, X, y):
        check_is_fitted(self)
        preds=self.predict(X)
        y_pred = np.vectorize(self._sigmoid)(preds.predictions) > 0.5
        metrics_dict = {}

        metrics_dict["loss"] = preds.metrics["test_loss"]
        metrics_dict["exact_match"] = skm.accuracy_score(y, y_pred, normalize=True, sample_weight=None)
        metrics_dict["recall"] = skm.recall_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["precision"] = skm.precision_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["f1_score"] = skm.f1_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        return y_pred, metrics_dict
    
    def _create_model(self, train, val=None, ckp_dir=None, compute_metrics=None):
        if ckp_dir:
            model = AutoModelForSequenceClassification.from_pretrained(ckp_dir).cuda()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.bert_version, 
                                                        problem_type="multi_label_classification", 
                                                        num_labels=len(train["labels"][0]))
        
        output_dir = ".ckp/"
        if self.output_dir:
            output_dir += self.output_dir
        
        args = TrainingArguments(
            report_to="none" if not self.run_name else "wandb",
            run_name=self.run_name,
            resume_from_checkpoint=False if not ckp_dir else self.output_dir,
            save_strategy="no" if not self.output_dir else "epoch", 
            eval_strategy="no" if not val else "epoch", 
            logging_strategy="no" if not self.output_dir else "steps",
            
            save_total_limit=10,
            logging_steps=1,

            output_dir=output_dir,

            # necessary to save models
            save_safetensors=True,

            bf16=True,

            learning_rate=self.learning_rate,
            num_train_epochs=self.n_epochs,
            warmup_steps=self.warmup_steps,
            per_device_train_batch_size=self.batch_size, 

            push_to_hub=False,
            disable_tqdm=True if not self.output_dir else False,
            bf16_full_eval=True,
            load_best_model_at_end=False,
            seed=self.seed
            )

        trainer = Trainer(
            model=model,
            train_dataset=train,
            eval_dataset=val,
            args=args,
            tokenizer=None,
            compute_metrics=compute_metrics
            )
        self._trainer = trainer



class VitWrapper(CustomWrapper):

    def __init__(self, vit_version="google/vit-large-patch16-224-in21k", learning_rate:float=1e-5,
                batch_size:int=16, n_epochs:int=10, warmup_steps:int=100,
                seed=123, output_dir:str=None, run_name:str=None):
        super().__init__()
        self.vit_version = vit_version
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.run_name = run_name
        self.seed = seed

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def score(self, X, y):
        check_is_fitted(self)
        preds=self.predict(X)
        y_pred = np.vectorize(self._sigmoid)(preds.predictions) > 0.5
        metrics_dict = {}

        metrics_dict["loss"] = preds.metrics["test_loss"]
        metrics_dict["exact_match"] = skm.accuracy_score(y, y_pred, normalize=True, sample_weight=None)
        metrics_dict["recall"] = skm.recall_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["precision"] = skm.precision_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        metrics_dict["f1_score"] = skm.f1_score(y_true=y, y_pred=y_pred, average='samples', zero_division=0)
        return y_pred, metrics_dict
    
    def _create_model(self, train, val=None, ckp_dir=None, compute_metrics=None):
        if ckp_dir:
            model = AutoModelForImageClassification.from_pretrained(ckp_dir).cuda()
        else:
            model = AutoModelForImageClassification.from_pretrained(self.vit_version, 
                                                        problem_type="multi_label_classification", 
                                                        num_labels=len(train["labels"][0]),
                                                        ignore_mismatched_sizes=True
                                                        )
    
        output_dir = ".ckp/"
        if self.output_dir:
            output_dir += self.output_dir
        
        args = TrainingArguments(
            report_to="none" if not self.run_name else "wandb",
            run_name=self.run_name,
            resume_from_checkpoint=False if not ckp_dir else self.output_dir,
            save_strategy="no" if not self.output_dir else "epoch", 
            eval_strategy="no" if not val else "epoch", 
            logging_strategy="no" if not self.output_dir else "steps",
            
            save_total_limit=10,
            logging_steps=1,

            output_dir=output_dir,

            # necessary to save models
            save_safetensors=True,

            bf16=True,

            learning_rate=self.learning_rate,
            num_train_epochs=self.n_epochs,
            warmup_steps=self.warmup_steps,
            per_device_train_batch_size=self.batch_size, 

            push_to_hub=False,
            disable_tqdm=True if not self.output_dir else False,
            bf16_full_eval=True,
            load_best_model_at_end=False,
            seed=self.seed
            )

        trainer = Trainer(
            model=model,
            train_dataset=train,
            eval_dataset=val,
            args=args,
            tokenizer=None,
            compute_metrics=compute_metrics
            )
        self._trainer = trainer