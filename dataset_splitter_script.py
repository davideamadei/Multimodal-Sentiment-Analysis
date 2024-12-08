from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from libs.dataset_loader import MulTweEmoDataset
import numpy as np

def count_labels(dataset):
    labels = MulTweEmoDataset.get_labels()
    labels
    count = {}
    for i in labels:
        count[i] = 0

    for i, row in dataset.iterrows():
        for label in labels:
            count[label] += 1 if row[label] else 0
    return count

def compute_percent(dataset):
    count = count_labels(dataset)
    for key in count.keys():
        count[key] /= len(dataset)
        count[key] = "{:.3f}".format(count[key])
    return count

def split_indices(dataset, split=0.2):
    choice = np.random.choice(range((dataset.id.unique().shape[0])), size=int(dataset.id.unique().shape[0]*split), replace=False)    
    ind = np.zeros(dataset.id.unique().shape[0], dtype=bool)
    ind[choice] = True
    rest = ~ind
    return rest, ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='dataset splitter',
        description='Split dataset in training and test set and saves them to disk as separate files',
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-d", "--dataset_path", type=str, default="./dataset/MulTweEmo.csv")
    parser.add_argument("-f", "--full_train_path", type=str, default="./dataset/full_train_MulTweEmo.csv")
    parser.add_argument("-t", "--train_path", type=str, default="./dataset/train_MulTweEmo.csv")
    parser.add_argument("-v", "--val_path", type=str, default="./dataset/val_MulTweEmo.csv")
    parser.add_argument("-s", "--test_path", type=str, default="./dataset/test_MulTweEmo.csv")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_path)
    
    full_train_ind, test_ind = split_indices(dataset, split=args.test_size)

    full_train = dataset[dataset["id"].isin(dataset.id.unique()[full_train_ind])]
    test = dataset[dataset["id"].isin(dataset.id.unique()[test_ind])]

    train_ind, val_ind = split_indices(full_train, split=args.test_size/(1-args.test_size))

    train = dataset[dataset["id"].isin(full_train.id.unique()[train_ind])]
    val = dataset[dataset["id"].isin(full_train.id.unique()[val_ind])]

    full_train.reset_index(names="original_index").to_csv(args.full_train_path, index=False)
    train.reset_index(names="original_index").to_csv(args.train_path, index=False)
    val.reset_index(names="original_index").to_csv(args.val_path, index=False)
    test.reset_index(names="original_index").to_csv(args.test_path, index=False)

    mode="M"
    full_train, _ = MulTweEmoDataset.load(csv_path=args.full_train_path, mode=mode, drop_something_else=True, force_override=True, test_split=None)
    train, _ = MulTweEmoDataset.load(csv_path=args.train_path,mode=mode, drop_something_else=True, force_override=True, test_split=None)
    val, _ = MulTweEmoDataset.load(csv_path=args.val_path,mode=mode, drop_something_else=True, force_override=True, test_split=None)
    test, _ = MulTweEmoDataset.load(csv_path=args.test_path,mode=mode, drop_something_else=True, force_override=True, test_split=None)
    dataset, _ = MulTweEmoDataset.load(csv_path=args.dataset_path, mode=mode, drop_something_else=True, force_override=True, test_split=None, seed=0)

    count_fun = count_labels
    print("Total:\n\ndataset: ", count_fun(dataset), "\ntrain: ", count_fun(train), "\ntest: ", count_fun(test), "\nval: ", count_fun(val))

    count_fun = compute_percent
    print("\n\n\n\n\nPercent:\n\ndataset: ", count_fun(dataset), "\ntrain: ", count_fun(train), "\ntest: ", count_fun(test), "\nval: ", count_fun(val))