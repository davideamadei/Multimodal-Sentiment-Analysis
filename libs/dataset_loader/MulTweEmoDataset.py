import pandas as pd
import gdown
from os.path import exists
from pathlib import PurePath
from zipfile import ZipFile
import re
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import html
import pathlib
from sklearn.model_selection import train_test_split
import emoji 

# a list of the possible labels for the dataset

def get_labels(drop_something_else=True):
    labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'something else', 'surprise', 'trust']
    if drop_something_else:
        labels.remove("something else")
    return labels

def get_id2label(drop_something_else=True):
    id2label = {}
    for i, label in enumerate(get_labels(drop_something_else)):
        id2label[i] = label
    return id2label

def get_label2id(drop_something_else=True):
    label2id = {}
    for i, label in enumerate(get_labels(drop_something_else)):
        label2id[label] = i
    return label2id
    

# TODO could add a config for the paths

def _download_dataset(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", image_zip_path="./dataset/raw/images.zip",
                   csv_path="./dataset/MulTweEmo.csv", create_dataset=False)->None:
    """utility function to download the raw dataset. If the files are already present they will not be downloaded again

    Parameters
    ----------
    raw_dataset_path : str, optional
        path where the raw dataset will be saved, by default "./dataset/raw/MulTweEmo_raw.pkl"
    image_zip_path : str, optional
        path where the image zip archive will be saved, by default "dataset/raw/images.zip"
    csv_path : str, optional
        path where the dataset csv will be saved, by default "./dataset/MulTweEmo.csv"
    create_dataset : bool, optional
        if True the dataset is created from scratch using the raw data, controls wether the raw dataset or the processed dataset is downloaded, by default False
    """
    # download images archive
    image_id = "1gfce4Ko3GsE4eJ2ILtr5swQoRXMRyFpS"
    image_url = f"https://drive.google.com/uc?id={image_id}"
    # create path if it does not exist
    image_path = pathlib.Path(image_zip_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    gdown.download(image_url, image_zip_path, quiet=False, resume=True)
    
    # decide which files to download, depending on if the dataset is being built from scratch or not
    if create_dataset:

        # download raw dataset pkl
        text_id = "1x5zOBcS2ktknP_lDTdLfMYMzMAqc_xMU"
        text_url = f"https://drive.google.com/uc?id={text_id}"
        # create path if it does not exist
        text_path = pathlib.Path(raw_dataset_path)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(text_url, raw_dataset_path, quiet=False, resume=True)
    else:

        # download processed dataset csv
        dataset_id = "10Yc4pFlVVPGGFHNblJe9ApA5n1sk8j14"
        dataset_url = f"https://drive.google.com/uc?id={dataset_id}"
        # create path if it does not exist
        dataset_path = pathlib.Path(csv_path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(dataset_url, csv_path, quiet=False, resume=True)


def _extract_images(files_to_extract: list[str], image_zip_path="./dataset/raw/images.zip", image_path="./dataset/images") -> None:
    """utility function to extract images from the zip archive

    Parameters
    ----------
    files_to_extract : list[str]
        list of files to extract
    image_zip_path : str, optional
        path where the zip archive is saved, by default "dataset/raw/images.zip"
    image_path : str, optional
        path where the images will be extracted, by default "./dataset/images"
    """
    with ZipFile(image_zip_path) as zfile:
        for info in zfile.infolist():
            if not info.is_dir():
                info.filename=PurePath(info.filename).name
                if str(info.filename) in files_to_extract and not exists(f"{image_path}/{info.filename}"):
                    zfile.extract(info, image_path)
      

def _create_csv(raw_dataset_path="./dataset/raw/MulTweEmo_raw.pkl", csv_path="./dataset/MulTweEmo.csv", 
                image_path="./dataset/images")->None:
    """utility function to save the csv file containing the processed dataset

    Parameters
    ----------
    raw_dataset_path : str, optional
        path where the raw dataset is located, by default "./dataset/raw/MulTweEmo_raw.pkl"
    csv_path : str, optional
        path where the csv will be saved, by default "./dataset/MulTweEmo.csv"
    image_path : str, optional
        path where the images are, by default "./dataset/images"
    image_zip_path : str, optional
        where to save the downloaded zip atchive containing the images, by default "dataset/raw/images.zip"
    """
    with open(raw_dataset_path, 'rb') as file:
        dataset = pd.compat.pickle_compat.load(file)

    dataset = dataset.drop(columns = ["M_gold_multi_label", "T_gold_multi_label"])
    dataset["img_count"] = dataset["path_photos"].apply(len)
    
    labels = dataset.columns[dataset.columns.str.startswith("M_") | dataset.columns.str.startswith("T_")].to_list()
    columns = ["id", "tweet", "img_count"] + labels

    dataset = dataset[columns]

    dataset = dataset[dataset["M_Anger"].notnull()].copy().reset_index(drop=True)

    labels = dataset.columns[dataset.columns.str.startswith("M_") | dataset.columns.str.startswith("T_")].to_list()
    for label in labels:
        dataset[label] = dataset[label].apply(lambda x: 1 if x>=2 else 0)

    dataset["img_name"] = dataset["img_count"].apply(lambda x : range(x))
    dataset = dataset.explode("img_name")
    dataset["img_name"] = dataset.apply(lambda x : f"{x['id']}_{x['img_name']}.jpg", axis=1)
    columns = ["id", "tweet", "img_name"] + labels

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)

    images = [Image.open(f"{image_path}/{x}").convert("RGB") for x in dataset["img_name"]]
    processed_images = processor(images, return_tensors="pt").to(device, torch.float16)

    out = model.generate(**processed_images)
    captions = processor.batch_decode(out, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]

    dataset["caption"] = captions
    columns = columns + ["caption"]
    del processor
    del model

    dataset.to_csv(csv_path, columns=columns, index=False, mode="w+")


def load(mode:str="M", raw_dataset_path:str="./dataset/raw/MulTweEmo_raw.pkl", csv_path:str="./dataset/MulTweEmo.csv", 
        image_path:str="./dataset/images", image_zip_path:str="./dataset/raw/images.zip",
        force_override=False, preprocess_tweets=True, emoji_decoding=False, build_label_matrix=True, drop_something_else=True,
        create_dataset=False, test_split:float=None, seed:int=None)->tuple[pd.DataFrame]:
        
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
        path where the images are located, and where they will be extracted if extract_images is set to True, by default "./dataset/images"
    image_zip_path : str, optional
        where to save the downloaded zip atchive containing the images, by default "dataset/raw/images.zip"
    force_override : bool, optional
        flag to overwrite the dataset, by default False
    preprocess_tweets : bool, optional
        flag to decide application of tweet preprocessing, by default True
    emoji_decoding : bool, optional
        flag to decide application if emojis will be replaced with their textual representation, by default False
    build_label_matrix : bool, optional
        flag to also add labels as a list of lists, by default True
    drop_something_else : bool, optional
        flag to drop the label 'something else', and data points which only have no other labels, by default True
    create_dataset : bool, optional
        if True the dataset is created from scratch using the raw data, by default False
    test_split : float, optional
        size of test split, training set will be the remaining percentage. If set to None test size will be empty by default 0.2
    seed : int, optional
        seed for splitting the dataset in train and test set, by default None

    Returns
    -------
    tuple[pd.DataFrame]
        tuple containig two pandas dataframes, one with training set and one with test set. If test_split is None the test set will be None

    Raises
    ------
    ValueError
        if mode has a value which is neither "M" or "T"
    """
    
    if test_split is not None and (test_split < 0 or test_split > 1):
        raise ValueError("train_test_split must be between 0 and 1")

    # download archive with images in dataset
    if not exists(image_zip_path):
        _download_dataset(raw_dataset_path=raw_dataset_path, image_zip_path=image_zip_path, csv_path=csv_path, create_dataset=create_dataset)

    # create dataset from raw data or download already processed dataset
    if create_dataset:
        if not exists(raw_dataset_path) or force_override:
            _download_dataset(raw_dataset_path=raw_dataset_path, image_zip_path=image_zip_path, csv_path=csv_path, create_dataset=create_dataset)
            _create_csv(raw_dataset_path=raw_dataset_path, csv_path=csv_path, image_path=image_path)
    else:
        if not exists(csv_path):
            _download_dataset(raw_dataset_path=raw_dataset_path, image_zip_path=image_zip_path, csv_path=csv_path, create_dataset=create_dataset)

    dataset = pd.read_csv(csv_path)

    # extract images from the zip file
    # skips already existing files
    _extract_images(image_zip_path=image_zip_path, image_path=image_path, files_to_extract=dataset["img_name"].to_list())

    # drop the labels for the mode which was not selected    
    features = dataset.columns.to_list()
    if mode=="M":
        dropped_features = [x for x in features if x.startswith("T_")]
    elif mode=="T":
        dropped_features = [x for x in features if x.startswith("M_")]
    else:
        raise ValueError("The only modes accepted are M for multimodal labels and T for text only labels.")
    dataset = dataset.drop(columns=dropped_features)

    # rename labels
    rename_map = {}
    features_to_rename = [x for x in features if x.startswith(f"{mode}_")]
    for feature in features_to_rename:
        rename_map[feature] = feature[2:].lower()
        
    dataset = dataset.rename(columns=rename_map)

    # preprocess tweets if necessary
    if preprocess_tweets:
        dataset = dataset.apply(_preprocess_tweet, axis=1, emoji_decoding=emoji_decoding)

    # remove rows without a label
    id_list = []
    labels = get_labels(drop_something_else)

    if drop_something_else:
        dataset = dataset.drop(columns=["something else"])

    if build_label_matrix:
        dataset.insert(2, "labels", _build_label_matrix(dataset, labels))

    for i, row in dataset.iterrows():
    # iterate on labels, if one with a non zero value is found the row is kept
        for emotion in labels:
            if row[emotion] != 0:
                id_list.append(i)
                break

    dataset = dataset.loc[id_list].reset_index(drop=True)

    
    dataset["img_name"] = dataset["img_name"].apply(lambda x: f"{image_path}/{x}")
    dataset = dataset.rename(columns={"img_name":"img_path"})

    if test_split is not None:
        return train_test_split(dataset, test_size=test_split, random_state=seed)

    return (dataset, None)
    # return dataset.train_test_split(test_size=test_split, seed=seed, shuffle=True)


def load_silver_dataset(raw_dataset_path="./dataset/MulTweEmo_raw.pkl",
                        csv_path="./dataset/silver_MulTweEmo.csv",  
                        silver_label_mode="label",
                        label_name="multi_label",
                        seed_threshold=0.81,
                        top_seeds:(int|dict)=None,
                        **kwargs):
        
    if silver_label_mode != "threshold" and silver_label_mode != "label":
        raise ValueError("mode must be chosen between \"top\", \"threhsold\" or \"label\"")

    with open(raw_dataset_path, 'rb') as file:
        dataset = pd.compat.pickle_compat.load(file)

    dataset = dataset[dataset["M_Anger"].isnull()].copy().reset_index(drop=True)
    
    dataset = dataset.drop(columns = ["M_gold_multi_label", "T_gold_multi_label"])
    dataset["img_count"] = dataset["path_photos"].apply(len)

    labels = get_labels(drop_something_else=False)

    emotions_m = {emotion: "M_"+emotion.capitalize() for emotion in labels}
    emotions_t = {emotion: "T_"+emotion.capitalize() for emotion in labels}
    
    label_columns = list(emotions_m.values()) + list(emotions_t.values())
    columns = ["id", "tweet"] + label_columns

    dataset[label_columns] = 0

    if silver_label_mode=="label":
        columns
        def set_labels(row):
            if label_name == "multi_label":
                for label in row[label_name]:
                    row[emotions_m[label]] = 1
                    row[emotions_t[label]] = 1
            elif label_name == "uni_label":
                label = row[label_name]
                row[emotions_m[label]] = 1
                row[emotions_t[label]] = 1
            else:
                raise ValueError()
            return row
    else:
        def set_labels(row):
            for e, d in row["seeds"].items():
                avg = sum(d.values())/len(d.values())
                if avg > seed_threshold:
                    row[emotions_m[e]] = 1
                    row[emotions_t[e]] = 1
            return row
        
    def seeds_avg(row):
        avgs = {}
        for e, d in row["seeds"].items():
            avgs[e] = sum(d.values())/len(d.values())
        row["avg_seeds"] = avgs
        return row
    
    dataset = dataset.apply(set_labels, axis=1)

    if top_seeds != None:
        dataset = dataset.apply(seeds_avg, axis=1)
        labels.remove("neutral")
        labels.remove("something else")
        indices = []
        if type(top_seeds) == int:
            for label in labels:
                indices += pd.DataFrame(dataset["avg_seeds"].to_list()).sort_values(by=label, ascending=False).head(top_seeds).index.to_list()
        else:
            for label, top_n in top_seeds.items():
                indices += pd.DataFrame(dataset["avg_seeds"].to_list()).sort_values(by=label, ascending=False).head(top_n).index.to_list()
        dataset = dataset.iloc[indices].sort_index().drop_duplicates(subset="id")
    dataset = dataset[columns]
    
    dataset["img_name"] = dataset["img_count"].apply(lambda x : range(x))
    dataset = dataset.explode("img_name")
    dataset["img_name"] = dataset.apply(lambda x : f"{x['id']}_{x['img_name']}.jpg", axis=1)
    dataset.to_csv(csv_path, index=False, mode="w+")
    
    return load(csv_path=csv_path, **kwargs)


def _preprocess_tweet(input: dict, emoji_decoding:bool):
    tweet = input["tweet"]

    if emoji_decoding:
        tweet = emoji.demojize(tweet, delimiters=(" ", " "))

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
    
    # converts html character references to actual character, e.g. &amp; to &
    tweet = html.unescape(tweet)
    
    if emoji_decoding:
        tweet=tweet.replace("_", " ")

    input["tweet"] = tweet
    return input

def _build_label_matrix(dataset, labels):
    label_matrix = []
    for i, row in dataset.iterrows():
        label_row = []
        for label in labels:
            label_row.append(float(row[label]))
        label_matrix.append(label_row)
    return label_matrix

