from libs.model import TweetMSA, TweetMSAConfig
from torch import manual_seed
from transformers import Trainer, TrainingArguments
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import sklearn.metrics as skm
from datasets import Dataset


#TODO: parameter checks
#TODO: documentation
class TweetMSA_Wrapper(BaseEstimator):

    def __init__(self, learning_rate:float=1e-5, batch_size:int=16, n_epochs:int=10, layers:tuple[int]=(512, 512), 
                 warmup_steps:int=100, clip_version:str="jina", freeze_weights=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.layers = layers
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
        self._is_fitted  = True
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
        return metrics_dict

    def predict(self, X):
        check_is_fitted(self)
        return self._trainer.predict(Dataset.from_pandas(X))

    def _create_model(self, X):
        config = TweetMSAConfig(feature_extractor=self.clip_version, layers=self.layers)
        model = TweetMSA(config).cuda()

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
                disable_tqdm=True
                )

        trainer = Trainer(
            model=model,
            train_dataset=Dataset.from_pandas(X),
            args=args,
            tokenizer=None,
            # compute_metrics=compute_metrics,
        
            )
        self._trainer = trainer
