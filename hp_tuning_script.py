from libs.utils import TweetMSA_Wrapper
from libs.dataset_loader import MulTweEmoDataset
from libs.model import TweetMSA
from datasets import Dataset
from torch import manual_seed
import optuna
import argparse
from sklearn.model_selection import KFold
import sklearn.metrics as skm    

class CVObjective(object):
    def __init__(self, clip_version="jina", append_captions:bool=False, freeze_weights:bool=False, cv_folds:int=4, seed:int=123):
        self.cv_folds = KFold(cv_folds, shuffle=True, random_state=seed)
        self.dataset, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=None)
        if append_captions:
            self.dataset["tweet"] = self.dataset.apply(lambda x: x["tweet"] + " " +  x["caption"], axis=1)

        self.dataset = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=self.dataset, model=clip_version, text_column="tweet", label_column="labels"))
        self.clip_version = clip_version
        self.freeze_weights = freeze_weights

    def __call__(self, trial):
        n_epochs = trial.suggest_int("n_epochs", 2, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=10)
        batch_size = trial.suggest_categorical("batch_size", [8,16,32])
        n_layers = trial.suggest_int("n_layers", 1, 10)
        n_units = trial.suggest_int("n_units", 32, 1024, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 1.0)
        metrics_keys = ["loss", "f1_score", "exact_match"]
        results = dict.fromkeys(metrics_keys, 0)
        for i, (train_index, val_index) in enumerate(self.cv_folds.split(self.dataset)):
            train = self.dataset.select(train_index)
            val = self.dataset.select(val_index)
            model = TweetMSA_Wrapper(n_epochs=n_epochs, warmup_steps=warmup_steps, learning_rate=learning_rate, 
                                 batch_size=batch_size, n_layers=n_layers, n_units=n_units,
                                 dropout=dropout, clip_version=self.clip_version, freeze_weights=self.freeze_weights)
            model.fit(train, train["labels"])
            _, fold_results =  model.score(val, val["labels"])
            for key in metrics_keys:
                results[key] = results[key] + fold_results[key]
            del model
            results = {key: (value)/self.cv_folds.get_n_splits() for key, value in results.items()}
        return tuple(results[key] for key in metrics_keys)
    

class NormalObjective(object):
    def __init__(self, clip_version="jina", append_captions:bool=False, freeze_weights:bool=False, seed:int=123):
        self.train, self.val = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True, test_split=0.25, seed=seed)

        if append_captions:
            self.train["tweet"] = self.train.apply(lambda x: x["tweet"] + " " + x["caption"], axis=1)
            self.val["tweet"] = self.val.apply(lambda x: x["tweet"] + " "  + x["caption"], axis=1)

        self.train = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=self.train, model=clip_version, text_column="tweet", label_column="labels"))
        self.val = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=self.val, model=clip_version, text_column="tweet", label_column="labels"))
        self.clip_version = clip_version
        self.freeze_weights = freeze_weights

    def __call__(self, trial):
        n_epochs = trial.suggest_int("n_epochs", 2, 15, logs=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=10)
        batch_size = trial.suggest_categorical("batch_size", [8,16,32])
        n_layers = trial.suggest_int("n_layers", 1, 10)
        n_units = trial.suggest_int("n_units", 32, 1024, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 1.0)
        model = TweetMSA_Wrapper(n_epochs=n_epochs, warmup_steps=warmup_steps, learning_rate=learning_rate, 
                                 batch_size=batch_size, n_layers=n_layers, n_units=n_units,
                                 dropout=dropout, clip_version=self.clip_version, freeze_weights=self.freeze_weights)
        model.fit(self.train, self.train["labels"])
        predictions, results =  model.score(self.val, self.val["labels"])
        label_names = MulTweEmoDataset.get_labels()
        label_names.remove("something else")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
    parser.add_argument('-c', '--clip-version', choices=["base", "large", "jina"], type=str, help='clip version for feature extraction')
    parser.add_argument("--freeze_weights", action="store_true", help="freezes weights of feature extractor")
    parser.add_argument("--append_captions", action="store_true", help="append auto-generated captions to tweet")
    parser.add_argument("-t", "--trials", type=int, default=None, help="number of trials to run, by default continues until killed")
    parser.add_argument("-T", "--timeout", type=int, default=None, help="how long to continue hpyerparameter searching in seconds, by default continues until killed")
    args = parser.parse_args()

    clip = args.clip_version
    
    if clip == "base":
        clip_model = "clip_base"
    elif clip == "large":
        clip_model = "clip_large"
    elif clip == "jina":	
        clip_model = "jina"
    else:
        raise ValueError("clip model is invalid, use help to see suported versions")
    manual_seed(123)
    objective = NormalObjective(clip_version=clip_model, append_captions=args.append_captions, freeze_weights=args.freeze_weights, seed=123)

    study_name = f"{clip_model}"  # Unique identifier of the study.
    if args.append_captions: study_name += "_append-captions"
    if args.freeze_weights: study_name += "_freeze-weights"
    study_name += "_study"
    storage_name = "sqlite:///MulTweEmo_study.db"
    
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name, 
                                load_if_exists=True,
                                directions=["minimize", "maximize", "maximize"])
    study.set_metric_names(["loss", "f1_score", "exact_match"])

    study.optimize(objective,n_trials=args.trials, timeout=args.timeout, n_jobs=1)

    print(study_name)

        