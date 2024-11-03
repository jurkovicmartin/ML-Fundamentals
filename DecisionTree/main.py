import pandas as pd

from tree import DecisionTree

def main():
    df = pd.read_csv("DecisionTree/data/Iris.csv")
    # Randomly shuffles rows (in the file they are ordered by specie (label))
    # Ordered dataset is problem for future training/testing data splitting
    df = df.sample(frac=1).reset_index(drop=True)
    # .values converts the dataframe type to numpy array
    data = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
    labels = df["Species"].values

    # Splitting dataset into training and testing data
    TRAIN_SPLIT = 130

    train_data = data[0:TRAIN_SPLIT]
    train_labels = labels[0:TRAIN_SPLIT]

    test_data = data[TRAIN_SPLIT:]
    test_labels = labels[TRAIN_SPLIT:]


    tree = DecisionTree()
    tree.build_tree(data=train_data, labels=train_labels, max_depth=5, min_split=2)
    # tree.print_tree()
    accuracy = tree.test(test_data, test_labels)
    print(f"Testing result: {accuracy} accuracy from {len(test_data)} testing samples")

if __name__ == "__main__":
    main()