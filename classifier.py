import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

class Dataset:
    def __init__(self):
        self.df = None

    def load(self, dataset):
        lb_make = LabelEncoder()
        if dataset == "iris":
            iris_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
            iris_df["species"] = lb_make.fit_transform(iris_df["species"])
            self.df = iris_df
        elif dataset == "krkopt":
            krkopt_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data", names=["White_King_file", "White_King_rank", "White_Rook_file", "White_Rook_rank", "Black_King_file", "Black_King_rank", "target"])
            krkopt_df["White_King_file"] = lb_make.fit_transform(krkopt_df["White_King_file"])
            krkopt_df["White_Rook_file"] = lb_make.fit_transform(krkopt_df["White_Rook_file"])
            krkopt_df["Black_King_file"] = lb_make.fit_transform(krkopt_df["Black_King_file"])
            krkopt_df["target"] = lb_make.fit_transform(krkopt_df["target"])
            self.df = krkopt_df
        elif dataset == "mushroom":
            mushroom_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", names=["poisonous", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"])
            for col in list(mushroom_df):
                mushroom_df[col] = lb_make.fit_transform(mushroom_df[col])
        elif dataset == "optdigits": 
            self.df = pd.read_csv("https://datahub.io/machine-learning/optdigits/r/optdigits.csv")
        else:
            print("Unknown dataset")

    def evaluate(self, target):
        leaf_count = []
        score_record = []

        x = self.df.loc()[:, self.df.columns != target]
        y = self.df.loc()[:, target]
        N = self.df.shape[0]

        clfs = {}
        for i in np.arange(2, N * 0.1, int((N * 0.1)/10)):
            clfs[f"Decision Tree min_samples_leaf={int(i)}"] = tree.DecisionTreeClassifier(min_samples_leaf=int(i))
            leaf_count.append(i)

        # build models
        for name, clf in clfs.items():
            scores = cross_val_score(clf, x, y, cv=5)
            score_record.append(scores)

        fig = plt.figure()
        plt.boxplot(score_record)
        ax = fig.add_subplot(111)
        ax.set_xticklabels(leaf_count)
        plt.show()

dataset = Dataset()
dataset.load("optdigits")
dataset.evaluate("class")

datasets = ["iris", "krkopt", "mushroom", "optdigits"]