from torch import manual_seed
import argparse
from libs.dataset_loader import MulTweEmoDataset
from libs.utils import *
from libs.model import TweetMERModel
from datasets import Dataset
from transformers import AutoTokenizer
import os
import sklearn.metrics as skm
import pandas as pd
import optuna
import numpy as np
import math
from pathlib import Path

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def compute_metrics_bert(eval_pred):
    y_pred, y_true = eval_pred
    
    y_pred = np.vectorize(sigmoid)(y_pred) > 0.5
    y_pred = y_pred > 0.5
    metrics_dict = {}

    metrics_dict["accuracy"] = skm.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    metrics_dict["recall"] = skm.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["precision"] = skm.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["f1_score"] = skm.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    return metrics_dict

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
    parser.add_argument('-m', '--model', choices=["bert", "base", "base_captions", "base_augment", "text_only", "high_support"], type=str, default="base", help="the model to train")
    parser.add_argument("-t", "--trial", type=int)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    
    model_type = args.model
    seed = args.seed
    trial = args.trial
    storage_name = "sqlite:///final_study_2.db"
    study = optuna.create_study(study_name=model_type+"_final_study", storage=storage_name, load_if_exists=True, directions=["minimize", "maximize", "maximize"])
    trials = study.get_trials()

    label_names = MulTweEmoDataset.get_labels()

    train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None, seed=123)

    val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None, seed=123)
    test, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None, seed=123)

    os.environ["WANDB_PROJECT"] = "final_models"

    # trial 268
    if model_type == "bert":
        def _preprocess_data(examples):
            tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
            text = examples["tweet"]
            encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)        
            return encoding


        train = train.drop_duplicates(subset=["id"])
        val = val.drop_duplicates(subset=["id"])
        test = test.drop_duplicates(subset=["id"])

        train = Dataset.from_pandas(train)
        val = Dataset.from_pandas(val)
        test = Dataset.from_pandas(test)

        train = train.map(_preprocess_data, batched=True, remove_columns=[col for col in train.column_names if col != "labels"])
        val = val.map(_preprocess_data, batched=True, remove_columns=[col for col in val.column_names if col != "labels"])
        test = test.map(_preprocess_data, batched=True, remove_columns=[col for col in test.column_names if col != "labels"])
        manual_seed(seed)

        model = BertWrapper(**(trials[trial].params),
                            output_dir=model_type,
                            run_name=model_type,
                            seed=seed
                            )
        
        metrics_function = compute_metrics_bert

    # trial 167
    elif model_type == "base":
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        manual_seed(seed)
        model = TweetMERWrapper(clip_version="base",
                                **(trials[trial].params),
                                batch_size=16,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )    
            
        metrics_function = compute_metrics
    
    # trial 287
    elif model_type == "base_captions":
        tweet_caption_data = train.apply(lambda x: x["tweet"] + " " + x["caption"], axis=1)
        train["tweet"] = tweet_caption_data

        train = TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels")
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        train = Dataset.from_pandas(train)

        manual_seed(seed)
        model = TweetMERWrapper(clip_version="base",
                                **(trials[trial].params),
                                batch_size=16,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )
        
        metrics_function = compute_metrics

    # trial 214
    elif model_type == "base_augment":
        print(model_type)
        silver_train, _ = MulTweEmoDataset.load_silver_dataset(silver_label_mode="threshold",
                                                            seed_threshold=0.84,
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
        
        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        manual_seed(seed)
        model = TweetMERWrapper(clip_version="base",
                                **(trials[trial].params),
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )
        
        metrics_function = compute_metrics
    
    # trial 170
    elif model_type == "high_support":
        train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True,
                                          drop_low_support=True, test_split=None, seed=123)
        val, _ = MulTweEmoDataset.load(csv_path="./dataset/val_MulTweEmo.csv", mode="M", drop_something_else=True,
                                          drop_low_support=True, test_split=None, seed=123)
        test, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", mode="M", drop_something_else=True,
                                          drop_low_support=True, test_split=None, seed=123)
        label_names = MulTweEmoDataset.get_labels(drop_low_support=True)

        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        manual_seed(seed)
        model = TweetMERWrapper(clip_version="base",
                                **(trials[trial].params),
                                batch_size=16,
                                n_classes=6,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )

        metrics_function = compute_metrics
    
    # trial 148
    elif model_type == "text_only":
        
        train = train.drop_duplicates(subset=["id"])
        val = val.drop_duplicates(subset=["id"])
        test = test.drop_duplicates(subset=["id"])

        train = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=train, model="base", text_column="tweet", label_column="labels"))
        val = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=val, model="base", text_column="tweet", label_column="labels"))
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        manual_seed(seed)
        model = TweetMERWrapper(clip_version="base",
                                **(trials[trial].params),
                                batch_size=16,
                                text_only=True,
                                output_dir=model_type,
                                run_name=model_type,
                                seed=seed
                                )
        
        metrics_function = compute_metrics
    
    model.fit(train, train["labels"], val, compute_metrics=metrics_function)
    
    _, results = model.score(val, val["labels"])
    val_predictions = model.predict(val).predictions
    test_predictions = model.predict(test).predictions
    train_predictions = model.predict(train).predictions

    print("\n\n\n")
    print(results)
    print(skm.classification_report(val["labels"], val_predictions>0.5, output_dict=False, zero_division=0, target_names=label_names))

    save_dir = Path(f"./multimodal_results/{model_type}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(f"./multimodal_results/{model_type}/val_predictions.np", "wb") as f:
        np.save(f, val_predictions)

    with open(f"./multimodal_results/{model_type}/test_predictions.np", "wb") as f:
        np.save(f, test_predictions)
        
    with open(f"./multimodal_results/{model_type}/train_predictions.np", "wb") as f:
        np.save(f, train_predictions)

    

        