import io
import base64

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
 
def build_graph(x_coordinates, y_coordinates):
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
    iris_N = 150

    clfs = {}
    score_record = []
    leaf_count = []
    print(iris_N)
    for i in np.arange(2, iris_N * 0.1, (iris_N * 0.1)/10):
        clfs[f"Decision Tree min_samples_leaf={int(i)}"] = tree.DecisionTreeClassifier(min_samples_leaf=int(i))
        leaf_count.append(i)

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    x_train = iris.data
    y_train = iris.target

    for name, clf in clfs.items():
        scores = cross_val_score(clf, x_train, y_train, cv=10)
        score_record.append(scores)

    fig = plt.figure()
    plt.boxplot(score_record)
    ax = fig.add_subplot(111)
    ax.set_xticklabels(leaf_count)


    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)