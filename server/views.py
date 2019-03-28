from . import app
from flask import render_template
from .graph import build_graph

from sklearn.datasets import load_iris
from sklearn import tree
import base64

from graphviz import Graph, Source

@app.route('/')
def graphs():
    items = ["Salut", "Test"]
    iris = load_iris()
    decisionTree = tree.DecisionTreeClassifier()
    decisionTree = decisionTree.fit(iris.data, iris.target)

    chart_data = Source(tree.export_graphviz(decisionTree))
    chart_output = chart_data.pipe(format='png')
    chart_output = base64.b64encode(chart_output).decode('utf-8')

    return render_template('index.html', items = items, decisionTree = decisionTree, chart_output=chart_output)


@app.route('/r/<list:subreddits>')
def subreddit_home(subreddits):
    """Show all of the posts for the given subreddits."""
    posts = []
    for subreddit in subreddits:
        posts.extend(subreddit.posts)

    return render_template('/r/index.html', posts=posts)