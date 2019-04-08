from . import app
from flask import render_template, session, request
from .graph import build_graph
from sklearn import tree
from .classifier import datasets, Dataset

import os
import base64

from graphviz import Graph, Source

@app.route('/')
def home_():
    return render_template('home.html')

@app.route('/data', methods = ['POST','GET'])
def data_():
    if request.method == 'POST':
        if "nom_data" in request.form:
            session["nom_data"] = request.form["nom_data"]
        if "nom_target" in request.form:
            session["nom_target"] = request.form["nom_target"]

    dataset = Dataset()
    nom_data = "iris"
    if "nom_data" in session:
        dataset.load(session["nom_data"])
        nom_data = session["nom_data"]
    else:
        dataset.load("iris")

    if not session["nom_target"] in dataset.targets:
        session["nom_target"] = dataset.targets[0]

    return render_template('data.html', dataframe = dataset.df, datas = [x for x in datasets if x != nom_data], targets =  [x for x in dataset.targets if x != session["nom_target"]])

@app.route('/tree', defaults={'min': 2})
@app.route('/tree/<int:min>')
def tree_(min):
    dataset = Dataset()
    if "nom_data" in session:
        dataset.load(session["nom_data"])
    else:
        dataset.load("iris")

    target = dataset.targets[0]

    if "nom_target" in session:
        target = session["nom_target"]
    dataset.evaluate(target)
    
    decision_tree = tree.DecisionTreeClassifier(min_samples_split=min)
    decision_tree = decision_tree.fit(dataset.x, dataset.y)
    dot_data = tree.export_graphviz(decision_tree, feature_names=[x for x in dataset.df.columns if x != target], filled=True)
    chart_data = Source(dot_data)
    chart_output = chart_data.pipe(format='png')
    chart_output = base64.b64encode(chart_output).decode('utf-8')
    return render_template('tree.html', decision_tree = decision_tree, chart_output = chart_output, mim = min, best = dataset.best_parameter)

@app.route('/resultat')
def resultat_():
    dataset = Dataset()
    if "nom_data" in session:
        dataset.load(session["nom_data"])
    else:
        dataset.load("iris")

    target = dataset.targets[0]

    if "nom_target" in session:
        target = session["nom_target"]
    image1 = dataset.evaluate(target)
    best1 = dataset.best_parameter
    image2 = dataset.evaluate_random_forest(target)
    best2 = dataset.best_parameter
    image3 = dataset.evaluate_knn(target)
    best3 = dataset.best_parameter
    return render_template('resultat.html', image1 = image1, best1 = best1, image2 = image2, best2 = best2, image3 = image3, best3 = best3)