import os
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

class Dataset:
    def __init__(self):
        self.df = None

    def load(self, dataset):
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        lb_make = LabelEncoder()
        if dataset == "iris":
            iris_df = pd.read_csv(os.path.join(SITE_ROOT, "static/dataset", "iris.data"), names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
            iris_df["species"] = lb_make.fit_transform(iris_df["species"])
            self.df = iris_df
        elif dataset == "krkopt":
            krkopt_df = pd.read_csv(os.path.join(SITE_ROOT, "static/dataset", "krkopt.data"), names=["White_King_file", "White_King_rank", "White_Rook_file", "White_Rook_rank", "Black_King_file", "Black_King_rank", "target"])
            krkopt_df["White_King_file"] = lb_make.fit_transform(krkopt_df["White_King_file"])
            krkopt_df["White_Rook_file"] = lb_make.fit_transform(krkopt_df["White_Rook_file"])
            krkopt_df["Black_King_file"] = lb_make.fit_transform(krkopt_df["Black_King_file"])
            krkopt_df["target"] = lb_make.fit_transform(krkopt_df["target"])
            self.df = krkopt_df
        elif dataset == "mushroom":
            mushroom_df = pd.read_csv(os.path.join(SITE_ROOT, "static/dataset", "agaricus-lepiota.data"), names=["poisonous", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"])
            for col in list(mushroom_df):
                mushroom_df[col] = lb_make.fit_transform(mushroom_df[col])
            self.df = mushroom_df
        elif dataset == "optdigits": 
            self.df = pd.read_csv(os.path.join(SITE_ROOT, "static/dataset", "optdigits.csv"))
        else:
            print("Unknown dataset")
        self.targets = self.df.columns

    def evaluate_knn(self, target):
        leaf_count = []
        score_record = []

        # shuffle the dataset
        self.df = shuffle(self.df)

        # split datas for data (x) and target (y)
        x = self.df.loc()[:, self.df.columns != target]
        y = self.df.loc()[:, target]
        N = self.df.shape[0]

        # build classifiers
        clfs = {}
        for i in np.arange(2, N * 0.1, int((N * 0.1)/10)):
            clfs[f"KNN n_neighbors={int(i)}"] = KNeighborsClassifier(n_neighbors=int(i))
            leaf_count.append(i)
            
        # build models
        for name, clf in clfs.items():
            scores = cross_val_score(clf, x, y, cv=5)
            score_record.append(scores)

        # search for the best parameter
        result = []
        for i in range(len(score_record)):
            #print(f"{leaf_count[i]} min per leaf : {np.mean(score_record[i])}")
            result.append((leaf_count[i], np.mean(score_record[i])))

        best_parameter = int(sorted(result, key=lambda x: x[1], reverse=True)[0][0])
        print(f"best parameter : {best_parameter}")

        # train with the best parameter
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

        best_cls = tree.DecisionTreeClassifier(min_samples_leaf=self.best_parameter)
        best_cls.fit(X_train, y_train)
        y_pred = best_cls.predict(X_test)       

        self.best_parameter = int(sorted(result, key=lambda x: x[1], reverse=True)[0][0])

        # cross validation result
        fig = plt.figure()
        plt.boxplot(score_record)
        ax = fig.add_subplot(111)
        ax.set_xticklabels(leaf_count)
        plt.xlabel("leaf_count")
        
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)
        return base64.b64encode(imgdata.read()).decode('utf-8')


    def evaluate_random_forest(self, target):
        leaf_count = []
        score_record = []

        # shuffle the dataset
        self.df = shuffle(self.df)

        # split datas for data (x) and target (y)
        x = self.df.loc()[:, self.df.columns != target]
        y = self.df.loc()[:, target]
        N = self.df.shape[0]

        # build classifiers
        clfs = {}
        for i in np.arange(2, N * 0.1, int((N * 0.1)/10)):
            clfs[f"Decision Tree min_samples_leaf={int(i)}"] = RandomForestClassifier(min_samples_leaf=int(i))
            leaf_count.append(i)
            
        # build models
        for name, clf in clfs.items():
            scores = cross_val_score(clf, x, y, cv=5)
            score_record.append(scores)

        # search for the best parameter
        result = []
        for i in range(len(score_record)):
            #print(f"{leaf_count[i]} min per leaf : {np.mean(score_record[i])}")
            result.append((leaf_count[i], np.mean(score_record[i])))

        self.best_parameter = int(sorted(result, key=lambda x: x[1], reverse=True)[0][0])

        # train with the best parameter
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

        best_cls = tree.DecisionTreeClassifier(min_samples_leaf=self.best_parameter)
        best_cls.fit(X_train, y_train)
        y_pred = best_cls.predict(X_test)       

        # evaluate final model
        print(f"final accuracy = {accuracy_score(y_test, y_pred)}")

        # cross validation result
        fig = plt.figure()
        plt.boxplot(score_record)
        ax = fig.add_subplot(111)
        ax.set_xticklabels(leaf_count)
        plt.xlabel("leaf_count")
        
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)
        return base64.b64encode(imgdata.read()).decode('utf-8')

    def evaluate(self, target):
        leaf_count = []
        score_record = []

        self.x = self.df.loc()[:, self.df.columns != target]
        self.y = self.df.loc()[:, target]
        N = self.df.shape[0]

        self.x = normalize(self.x, norm='l2')

        clfs = {}
        for i in np.arange(2, N * 0.1, int((N * 0.1)/10)):
            clfs[f"Decision Tree min_samples_leaf={int(i)}"] = tree.DecisionTreeClassifier(min_samples_leaf=int(i))
            leaf_count.append(i)

        # build models
        for name, clf in clfs.items():
            scores = cross_val_score(clf, self.x, self.y, cv=5)
            score_record.append(scores)

        # search for the best parameter
        result = []
        for i in range(len(score_record)):
            #print(f"{leaf_count[i]} min per leaf : {np.mean(score_record[i])}")
            result.append((leaf_count[i], np.mean(score_record[i])))

        self.best_parameter = int(sorted(result, key=lambda x: x[1], reverse=True)[0][0])

        fig = plt.figure()
        plt.boxplot(score_record)
        ax = fig.add_subplot(111)
        ax.set_xticklabels(leaf_count)
        plt.xlabel("leaf_count")
        
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)
        return base64.b64encode(imgdata.read()).decode('utf-8')


datasets = ["iris", "krkopt", "mushroom", "optdigits"]

