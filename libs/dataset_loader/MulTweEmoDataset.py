import pandas as pd
import os.path
from datasets import load_dataset
class MulTweEmoDataset():
    
    @staticmethod
    def _download_raw():
        pass

    @staticmethod
    def create_csv(dataset_path="./dataset/gold_label_dataset.pkl", images_path="./dataset/gold_images/twint_images3/"):
        with open(dataset_path, 'rb') as file:
            dataset = pd.compat.pickle_compat.load(file)

        labels = dataset.columns[dataset.columns.str.startswith("M_") | dataset.columns.str.startswith("T_")].to_list()
        for label in labels:
            dataset[label] = dataset[label].apply(lambda x: 1 if x>=2 else 0)

        dataset["img_path"] = dataset["id"].apply(lambda x : f"{images_path}{x}_0.jpg")
        columns = ["id", "tweet", "img_path"] + labels
        dataset.to_csv("./dataset/gold_dataset.csv", columns=columns, index=False)

    @staticmethod
    def load(dataset_path="./dataset/gold_label_dataset.pkl", mode="multi", images_path="./dataset/gold_images/twint_images3/", force_override=False):
        if not os.path.exists("./gold_dataset.csv") or force_override:
            MulTweEmoDataset.create_csv()

        dataset = load_dataset("csv", data_files="./dataset/gold_dataset.csv")

        features = dataset["train"].features
        if mode=="multi":
            dropped_features = [x for x in features if x.startswith("T_")]
        else:
            dropped_features = [x for x in features if x.startswith("M_")]


        dataset = dataset.remove_columns(dropped_features)
        dataset = MulTweEmoDataset._create_label_matrix(dataset)
        return dataset


    @staticmethod
    def _create_label_matrix(dataset):
        for split in dataset:
            features = dataset[split].features
            labels = [x for x in features if (x.startswith("T_") or x.startswith("M_"))]
            label_matrix = []
            for elem in dataset[split]:
                label_row = []
                for label in labels:
                    label_row.append(elem[label])
                label_matrix.append(label_row)
            dataset[split] = dataset[split].add_column("labels", label_matrix)
        return dataset

