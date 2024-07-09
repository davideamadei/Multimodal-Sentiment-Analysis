import pandas as pd
import gdown
from os.path import exists
from pathlib import PurePath
from datasets import load_dataset, Dataset
from zipfile import ZipFile
import re

# a list of the possible labels for the dataset
labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'something else', 'surprise', 'trust']

# TODO could add a config for the paths
 
def _download_raw(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", image_zip_path="dataset/raw/images.zip")->None:
    """utility function to download the raw dataset. If the files are already present they will not be downloaded again

    Parameters
    ----------
    raw_dataset_path : str, optional
        path where the raw dataset will be saved, by default "./dataset/raw/MulTweEmo_raw.pkl"
    image_zip_path : str, optional
        path where the image zip archive will be saved, by default "dataset/raw/images.zip"
    """
    image_id = "1gfce4Ko3GsE4eJ2ILtr5swQoRXMRyFpS"
    text_id = "1x5zOBcS2ktknP_lDTdLfMYMzMAqc_xMU"

    image_url = f"https://drive.google.com/uc?id={image_id}"
    text_url = f"https://drive.google.com/uc?id={text_id}"
    

    gdown.download(image_url, image_zip_path, quiet=False, resume=True)
    gdown.download(text_url, raw_dataset_path, quiet=False, resume=True)


def _extract_images(image_zip_path="dataset/raw/images.zip", image_path="./dataset/images") -> None:
    """utility function to extract images from the zip archive

    Parameters
    ----------
    image_zip_path : str, optional
        path where the zip archive is saved, by default "dataset/raw/images.zip"
    image_path : str, optional
        path where the images will be extracted, by default "./dataset/images"
    """
    if not exists(image_zip_path):
        _download_raw(image_zip_path=image_zip_path)

    with ZipFile(image_zip_path) as zfile:
        for info in zfile.infolist():
            if not info.is_dir():
                info.filename=PurePath(info.filename).name
                if not exists(f"{image_path}/{info.filename}"):
                    zfile.extract(info, image_path)
    

def _prepare_dataset(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl") -> pd.DataFrame:
    """utility function to process the raw dataset

    Parameters
    ----------
    raw_dataset_path : str, optional
        path where the raw dataset is located, by default "./dataset/raw/MulTweEmo_raw.pkl"

    Returns
    -------
    pd.DataFrame
        the processed dataset
    """
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
    

def _create_csv(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", csv_path="./dataset/MulTweEmo.csv", image_path="./dataset/images")->None:
    """utility function to save the csv file containing the processed dataset

    Parameters
    ----------
    raw_dataset_path : str, optional
        path where the raw dataset is located, by default "./dataset/raw/MulTweEmo_raw.pkl"
    csv_path : str, optional
        path where the csv will be saved, by default "./dataset/MulTweEmo.csv"
    image_path : str, optional
        path where the images are, by default "./dataset/images"
    force_override : bool, optional
        flag to force setup of dataset from the start, by default False
    """
    dataset = _prepare_dataset(raw_dataset_path)

    labels = dataset.columns[dataset.columns.str.startswith("M_") | dataset.columns.str.startswith("T_")].to_list()
    for label in labels:
        dataset[label] = dataset[label].apply(lambda x: 1 if x>=2 else 0)

    
    dataset["img_path"] = dataset["img_count"].apply(lambda x : range(x))
    dataset = dataset.explode("img_path")
    dataset["img_path"] = dataset.apply(lambda x : f"{image_path}/{x['id']}_{x['img_path']}.jpg", axis=1)
    columns = ["id", "tweet", "img_path"] + labels
    dataset.to_csv(csv_path, columns=columns, index=False)


# TODO add options for loading i.e. preprocess tweets, build label matrix etc
def load(mode="M", raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", csv_path="./dataset/MulTweEmo.csv", 
        image_path="./dataset/images", image_zip_path="dataset/raw/images.zip",
        force_override=False, preprocess_tweets=True, build_label_matrix=True, automated_captions=False)->Dataset:
        
    """function to load the MulTweEmo dataset, downloads dataset if not cached. The processed dataset for further uses is also saved as a csv

    Parameters
    ----------
    mode : str, optional
        which labels to load, "M" for multimodal ones, "T" for text-only, by default "M"
    raw_dataset_path : str, optional
        where to download the raw dataset, by default "./dataset/raw/MulTweEmo_raw.pkl"
    csv_path : str, optional
        where to save the processed dataset, by default "./dataset/MulTweEmo.csv"
    image_path : str, optional
        where to extract the images of the dataset, by default "./dataset/images"
    image_zip_path : str, optional
        where to save the downloaded zip atchive containing the images, by default "dataset/raw/images.zip"
    force_override : bool, optional
        flag to force setup of dataset from the start, by default False
    preprocess_tweets : bool, optional
        flag to decide application of tweet preprocessing, by default True
    build_label_matrix : bool, optional
        flag to also add labels as a list of lists, by default True

    Returns
    -------
    Dataset
        the dataset with the selected labels loaded as a huggingface dataset

    Raises
    ------
    ValueError
        if mode has a value which is neither "M" or "T"
    """

    # if the csv with the processed dataset does not exist yet, create it
    if not exists(csv_path) or force_override:
        _create_csv(raw_dataset_path=raw_dataset_path, csv_path=csv_path, image_path=image_path)

    dataset = load_dataset("csv", data_files=csv_path, split="train")
    
    # extract images from the zip file
    # NOTE: currently it extracts every image, even those of the samples without gold labels
    _extract_images(image_zip_path=image_zip_path, image_path=image_path)

    # drop the labels for the mode which was not selected    
    features = dataset.features
    if mode=="M":
        dropped_features = [x for x in features if x.startswith("T_")]
    elif mode=="T":
        dropped_features = [x for x in features if x.startswith("M_")]
    else:
        raise ValueError("The only modes accepted are M for multimodal labels and T for text only labels.")
    
    dataset = dataset.remove_columns(dropped_features)

    # rename labels
    rename_map = {}
    features_to_rename = [x for x in features if x.startswith(f"{mode}_")]
    for feature in features_to_rename:
        rename_map[feature] = feature[2:].lower()
    dataset = dataset.rename_columns(rename_map)

    # preprocess tweets if necessary
    if preprocess_tweets:
        dataset = dataset.map(_preprocess_tweet)

    if build_label_matrix:
        dataset = dataset.add_column("labels", _build_label_matrix(dataset))

    # remove rows without a label
    id_list = []
    global labels

    for i, elem in enumerate(dataset):
    # iterate on labels, if one with a non zero value is found the row is kept
        for emotion in labels:
            if elem[emotion] != 0:
                id_list.append(i)
                break
    dataset = dataset.select(id_list)

    return dataset

# TODO potentially handle emoji, urls, mentions with substitution instead of removal
def _preprocess_tweet(input: dict):
    tweet = input["tweet"]

    # remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    tweet = re.sub(emoji_pattern,'',tweet)

    # remove urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+?')
    tweet = re.sub(url_pattern,'', tweet)

    # remove @ mentions and hashes
    hash_pattern = re.compile("#")
    tweet = re.sub(hash_pattern,"",tweet)

    mention_pattern = re.compile("@[A-Za-z0–9_]+")
    tweet = re.sub(mention_pattern,"",tweet)
    
    # remove occurrences of &amp;
    and_pattern = re.compile("&amp;")
    tweet = re.sub(and_pattern,"&",tweet)

    input["tweet"] = tweet
    return input

def _build_label_matrix(dataset):
    global labels
    label_matrix = []
    for elem in dataset:
        label_row = []
        for label in labels:
            label_row.append(float(elem[label]))
        label_matrix.append(label_row)
    return label_matrix

