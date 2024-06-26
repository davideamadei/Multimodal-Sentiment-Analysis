import pandas as pd
import gdown
from os.path import exists
from pathlib import Path, PurePath
from datasets import load_dataset
from zipfile import ZipFile


labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'something else', 'surprise', 'trust']

# TODO could add a config for the paths
# TODO documentation
 
def _download_raw(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", image_zip_path="dataset/raw/images.zip")->None:
    image_id = "1gfce4Ko3GsE4eJ2ILtr5swQoRXMRyFpS"
    text_id = "1x5zOBcS2ktknP_lDTdLfMYMzMAqc_xMU"

    image_url = f"https://drive.google.com/uc?id={image_id}"
    text_url = f"https://drive.google.com/uc?id={text_id}"
    

    gdown.download(image_url, image_zip_path, quiet=False, resume=True)
    gdown.download(text_url, raw_dataset_path, quiet=False, resume=True)


def _extract_images(image_zip_path="dataset/raw/images.zip", image_path="./dataset/images") -> None:
    if not exists(image_zip_path):
        _download_raw(image_zip_path=image_zip_path)

    with ZipFile(image_zip_path) as zfile:
        for info in zfile.infolist():
            if not info.is_dir():
                info.filename=PurePath(info.filename).name
                if not exists(f"{image_path}/{info.filename}"):
                    zfile.extract(info, image_path)
    

def _prepare_dataset(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl") -> pd.DataFrame:
    if not exists(raw_dataset_path):
        _download_raw(raw_dataset_path=raw_dataset_path)

    with open(raw_dataset_path, 'rb') as file:
        raw_dataset = pd.compat.pickle_compat.load(file)

    raw_dataset = raw_dataset.drop(columns = ["M_gold_multi_label", "T_gold_multi_label"])
    raw_dataset["img_count"] = raw_dataset["path_photos"].apply(len)
    
    labels = raw_dataset.columns[raw_dataset.columns.str.startswith("M_") | raw_dataset.columns.str.startswith("T_")].to_list()
    columns = ["id", "tweet", "img_count"] + labels

    raw_dataset = raw_dataset[columns]

    raw_dataset = raw_dataset[raw_dataset["M_Anger"].notnull()].copy().reset_index(drop=True)
    return raw_dataset
    

def _create_csv(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", csv_path="./dataset/MulTweEmo.csv", image_path="./dataset/images"):              
    dataset = _prepare_dataset(raw_dataset_path)

    labels = dataset.columns[dataset.columns.str.startswith("M_") | dataset.columns.str.startswith("T_")].to_list()
    for label in labels:
        dataset[label] = dataset[label].apply(lambda x: 1 if x>=2 else 0)

    # TODO decide how to handle tweets with multiple images
    dataset["img_path"] = dataset["id"].apply(lambda x : f"{image_path}/{x}_0.jpg")
    columns = ["id", "tweet", "img_path"] + labels
    dataset.to_csv(csv_path, columns=columns, index=False)


def load(mode="M", raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", csv_path="./dataset/MulTweEmo.csv", image_path="./dataset/images", image_zip_path="dataset/raw/images.zip"):
    if not exists(csv_path):
        _create_csv(raw_dataset_path=raw_dataset_path, csv_path=csv_path, image_path=image_path)

    dataset = load_dataset("csv", data_files=csv_path, split="train")
    
    _extract_images(image_zip_path=image_zip_path, image_path=image_path)

    features = dataset.features
    if mode=="M":
        dropped_features = [x for x in features if x.startswith("T_")]
    elif mode=="T":
        dropped_features = [x for x in features if x.startswith("M_")]
    else:
        raise ValueError("The only modes accepted are M for multimodal labels and T for text only labels.")
    
    dataset = dataset.remove_columns(dropped_features)

    rename_map = {}
    features_to_rename = [x for x in features if x.startswith(f"{mode}_")]
    for feature in features_to_rename:
        rename_map[feature] = feature[2:].lower()

    dataset = dataset.rename_columns(rename_map)
    return dataset

def build_label_matrix(dataset):
    global labels
    features = dataset.features
    label_matrix = []
    for elem in dataset:
        label_row = []
        for label in labels:
            label_row.append(elem[label])
        label_matrix.append(label_row)
    return label_matrix

