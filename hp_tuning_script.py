from libs.utils import *
from torch import manual_seed
import optuna
import argparse
from sklearn.model_selection import KFold
import sklearn.metrics as skm    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
    parser.add_argument('-m', '--model', choices=["text", "image", "multimodal"], type=str, default="clip", help="type of model on which hp optimization will be performed")
    parser.add_argument('-c', '--clip-version', choices=["base", "large", "jina", "blip2"], type=str, default="jina", help='clip version for feature extraction, multimodal only')
    parser.add_argument("--freeze_weights", action="store_true", help="freezes weights of feature extractor, multimodal only")
    parser.add_argument("--append_captions", action="store_true", help="append auto-generated captions to tweet, multimodal only")
    parser.add_argument("-t", "--trials", type=int, default=None, help="number of trials to run, by default continues until killed")
    parser.add_argument("-T", "--timeout", type=int, default=None, help="how long to continue hpyerparameter searching in seconds, by default continues until killed")
    parser.add_argument("-s", "--study_name", type=str, default="MulTweEmo_study", help="name of the file where studies will be saved, cannot be a path")
    parser.add_argument('-M', '--mode', choices=["M", "T"], type=str, default="M", help="which version of the labels to use, Multimodal or Text only")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    clip = args.clip_version
    model = args.model
    mode = args.mode
    seed = args.seed

    if model != "multimodal" and args.freeze_weights:
        raise ValueError("--freeze_weights cannot be passed for non multimodal model")
    
    if model == "image" and args.append_captions: 
        raise ValueError("--append_captions cannot be passed for image only model")
    
    if model not in ["text", "image", "multimodal"]:
        raise ValueError("type of model to optimize is not valid, use help to see suported models")
    
    if model == "multimodal" and clip not in ["base", "large", "jina", "blip2"]:
        raise ValueError("clip model is not valid, use help to see suported versions")

    if mode not in ["M", "T"]:
        raise ValueError("mode can only be M or T")
    
    manual_seed(seed)
    
    if model == "multimodal":
        objective = TweetMSAObjective(clip_version=clip, append_captions=args.append_captions, freeze_weights=args.freeze_weights, seed=seed, mode=mode)
        study_name = f"{clip}"  # Unique identifier of the study.
    elif model == "text":
        objective = BertObjective(append_captions=args.append_captions, mode=mode, seed=seed)
        study_name = "bert"  # Unique identifier of the study.
    else:
        objective = VitObjective(mode=mode, seed=seed)
        study_name = "vit"  # Unique identifier of the study.
    
    if model != "image" and args.append_captions: study_name += "_append-captions"
    if model == "multimodal" and args.freeze_weights: study_name += "_freeze-weights"
    study_name += "_study"
    storage_name = f"sqlite:///{args.study_name}.db"
    
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name, 
                                load_if_exists=True,
                                directions=["minimize", "maximize", "maximize"])
    study.set_metric_names(["loss", "f1_score", "exact_match"])

    study.optimize(objective,n_trials=args.trials, timeout=args.timeout, n_jobs=1)

    print(study_name)

        