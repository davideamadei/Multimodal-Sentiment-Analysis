from torch import manual_seed
import argparse
from libs.dataset_loader import MulTweEmoDataset
from libs.utils import *
from libs.model import TweetMSA
from datasets import Dataset
from transformers import AutoTokenizer
import os
import sklearn.metrics as skm
import pandas as pd

def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    y_pred = y_pred > 0.5
    metrics_dict = {}

    metrics_dict["accuracy"] = skm.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    metrics_dict["recall"] = skm.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["precision"] = skm.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["f1_score"] = skm.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    return metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
    parser.add_argument('-m', '--model', choices=["bert", "base", "base_captions", "base_augment"], type=str, default="base", help="the model to train")
    
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    
    model_type = args.model
    seed = args.seed
    manual_seed(seed)

    train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None, seed=123)

    val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None, seed=123)

    # train = train.head(20)
    # val = val.head(10)
    
    os.environ["WANDB_PROJECT"] = "final_models"

    if model_type == "bert":
        def _preprocess_data(examples):
            tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
            text = examples["tweet"]
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)        
            return encoding


        train = train.drop_duplicates(subset=["id"])
        val = val.drop_duplicates(subset=["id"])
        val = val[~val.id.isin(train.id.values)]

        train = Dataset.from_pandas(train)
        val = Dataset.from_pandas(val)

        train = train.map(_preprocess_data, batched=True, remove_columns=[col for col in train.column_names if col != "labels"])
        val = val.map(_preprocess_data, batched=True, remove_columns=[col for col in val.column_names if col != "labels"])

        model = BertWrapper(n_epochs=11,
                            batch_size=16,
                            warmup_steps=20,
                            learning_rate=4.259470947149478e-05,
                            output_dir=model_type,
                            run_name=model_type,
                            seed=seed
                            )

        
    elif model_type == "base":
        train = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model = TweetMSAWrapper(clip_version="base",
                                n_epochs=14,
                                batch_size=16,
                                warmup_steps=150,
                                learning_rate=1.6856413214253974e-05,
                                n_layers=1,
                                n_units=66,
                                dropout=0.3240428275567533,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )    
            
    elif model_type == "base_captions":
        train = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))
       
        tweet_caption_data = train.apply(lambda x: x["tweet"] + " " + x["caption"], axis=1)
        train["tweet"] = tweet_caption_data 
        
        model = TweetMSAWrapper(clip_version="base",
                                n_epochs=14,
                                batch_size=16,
                                warmup_steps=150,
                                learning_rate=1.6856413214253974e-05,
                                n_layers=1,
                                n_units=66,
                                dropout=0.2,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )
        
    else:
        silver_train = MulTweEmoDataset.load_silver_dataset(silver_label_mode="threshold",
                                                            seed_threshold=0.82,
                                                            top_seeds={
                                                                "trust":40,
                                                                "fear":40,
                                                                "surprise":30,
                                                                },
                                                            csv_path="./dataset/silver_MulTweEmo.csv",
                                                            test_split=None,
                                                            mode="M"
                                                            )
        train = pd.concat([train, silver_train])
        
        train = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model = TweetMSAWrapper(clip_version="base",
                                n_epochs=6,
                                batch_size=8,
                                warmup_steps=50,
                                learning_rate=2.8626493397033086e-05,
                                n_layers=1,
                                n_units=199,
                                dropout=0.2443714062077184,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )
        
    model.fit(train, train["labels"], val, compute_metrics=compute_metrics)
    

        