from flask import Flask

app = Flask(__name__)

from .util import ListConverter

app.url_map.converters['list'] = ListConverter
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

from .views import *