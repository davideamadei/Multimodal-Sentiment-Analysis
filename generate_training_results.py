import argparse
from libs.dataset_loader import MulTweEmoDataset
from libs.utils import *
from libs.model import TweetMERModel
from datasets import Dataset
from transformers import AutoTokenizer
import os
import json
from pathlib import Path

def _preprocess_data(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    text = examples["tweet"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)        
    return encoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
    parser.add_argument('-m', '--model', choices=["bert", "base", "base_captions", "base_augment", "text_only"], type=str, default="base", help="the model to train")

    args = parser.parse_args()
    
    model_type = args.model

    train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", drop_something_else=True, test_split=None)
    val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", drop_something_else=True, test_split=None)

    if model_type=="bert":
        
        train = train.drop_duplicates(subset=["id"])
        val = val.drop_duplicates(subset=["id"])
        
        train = Dataset.from_pandas(train)
        val = Dataset.from_pandas(val)
        train = train.map(_preprocess_data, batched=True, remove_columns=[col for col in train.column_names if col != "labels"])
        val = val.map(_preprocess_data, batched=True, remove_columns=[col for col in val.column_names if col != "labels"])

        model_class = BertWrapper

    elif model_type=="base":
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="base_augment":
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="base_captions":
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="high_support":
        train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", drop_something_else=True, drop_low_support=True, test_split=None)
        val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", drop_something_else=True, drop_low_support=True, test_split=None)
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="text_only":        
        train = train.drop_duplicates(subset=["id"])
        val = val.drop_duplicates(subset=["id"])
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    results_dict = {data: {x: None for x in os.listdir(f".ckp/{model_type}/")} for data in ["train", "val"]}
    for ckp in os.listdir(f".ckp/{model_type}/"):
        model = model_class()
        model.from_pretrained(f".ckp/{model_type}/{ckp}")

        train_predictions, train_scores = model.score(train, train["labels"])
        results_dict["train"][ckp] = train_scores

        val_predictions, val_scores = model.score(val, val["labels"])
        results_dict["val"][ckp] = val_scores

    save_dir = Path("checkpoint_metrics/")
    save_dir.mkdir(exist_ok=True)

    with open(f"{save_dir}/{model_type}.json", 'w') as fp:
        json.dump(results_dict, fp)