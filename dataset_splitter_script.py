from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from libs.dataset_loader import MulTweEmoDataset

def count_labels(dataset):
    labels = MulTweEmoDataset.get_labels()
    labels.remove("something else")
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
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='dataset splitter',
        description='Split dataset in training and test set and saves them to disk as separate files',
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-d", "--dataset_path", type=str, default="./dataset/MulTweEmo.csv")
    parser.add_argument("-t", "--train_path", type=str, default="./dataset/train_MulTweEmo.csv")
    parser.add_argument("-s", "--test_path", type=str, default="./dataset/test_MulTweEmo.csv")
    args = parser.parse_args()
    

    dataset = pd.read_csv(args.dataset_path)
    train, test = train_test_split(dataset, test_size=args.test_size, random_state=args.seed)
    train.reset_index(names="original_index").to_csv(args.train_path, index=False)
    test.reset_index(names="original_index").to_csv(args.test_path, index=False)

    mode="M"
    train, _ = MulTweEmoDataset.load(csv_path=args.train_path, mode=mode, drop_something_else=True, force_override=True, test_split=None)
    test, _ = MulTweEmoDataset.load(csv_path=args.test_path,mode=mode, drop_something_else=True, force_override=True, test_split=None)
    dataset, _ = MulTweEmoDataset.load(csv_path="./dataset/MulTweEmo.csv", mode=mode, drop_something_else=True, force_override=True, test_split=None, seed=0)

    count_fun = count_labels
    print("Total:\n\ndataset: ", count_fun(dataset), "\ntrain: ", count_fun(train), "\ntest: ", count_fun(test))

    count_fun = compute_percent
    print("\n\n\n\n\nPercent:\n\ndataset: ", count_fun(dataset), "\ntrain: ", count_fun(train), "\ntest: ", count_fun(test))