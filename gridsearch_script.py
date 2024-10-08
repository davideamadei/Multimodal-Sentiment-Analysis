from libs.model import TweetMSA, TweetMSAConfig
from libs.dataset_loader import MulTweEmoDataset
from libs.utils import TweetMSA_Wrapper
from torch import manual_seed
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import argparse
import pickle



def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    y_pred = y_pred > 0.5
    metrics_dict = {}

    metrics_dict["exact_match"] = skm.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    metrics_dict["recall"] = skm.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["precision"] = skm.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["f1_score"] = skm.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    return metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
    parser.add_argument('-c', '--clip-version', choices=["base", "large", "jina"], type=str, help='clip version for feature extraction')
    parser.add_argument("--freeze_weights", action="store_true", help="freezes weights of feature extractor")
    parser.add_argument("--append_captions", action="store_true", help="append auto-generated captions to tweet")
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
    estimator = TweetMSA_Wrapper(clip_version=clip_model)

    #TODO: sample dataset properly
    dataset, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode="M", drop_something_else=True)

    if args.append_captions:
        dataset ["input"] = dataset.apply(lambda x: x["tweet"] + x["caption"], axis=1)

    dataset = TweetMSA.preprocess_dataset(dataset=dataset, model=clip_model, text_column="tweet" if not args.append_captions else "input", label_column="labels")

    param_grid = {"n_epochs": [2,3,4,5,6], 
                  "learning_rate": [1e-4, 1e-5, 5e-5],
                  "batch_size": [8, 16, 32],
                  "layers": [(512, ), (512,512), (512,512,512)],
                  "warmup_steps": [0, 50, 100]
                  }
    folds = KFold(5, shuffle=False)

    grid_search = GridSearchCV(estimator, param_grid, refit=False, cv=folds, return_train_score=True, verbose=True, n_jobs=1)

    results = grid_search.fit(dataset, dataset["labels"].tolist(),)

    print(results.cv_results_)
    with open("./gridsearch_results.pkl", "wb") as f:
        pickle.dump(results.cv_results_, f)

        