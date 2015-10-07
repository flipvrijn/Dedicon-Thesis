import cPickle
import glob
import json
import os.path
import time
import utils

from flask import Flask, render_template
from flask.ext.navigation import Navigation

app = Flask(__name__)
app.debug = True
nav = Navigation(app)

# Navigation bar
nav.Bar('top', [
	nav.Item('Overzicht', 'index'),
	nav.Item('Opties', 'options')
])

@app.route('/')
def index():
	''' Render the overview page with the available models '''
	models_directory = '/media/Data/flipvanrijn/models/'.rstrip('/')

	# List of caption generation models available
	models = []
	files = glob.glob('{}/*.pkl'.format(models_directory))
	for f in files:
		filename = os.path.split(f)[1]
		name 	 = filename[:-4] # without the '.pkl'
		modified = time.ctime(os.path.getmtime(f))
		running  = utils.check_model_running(filename)
		with open(f, 'r') as handler:
			options  = cPickle.load(handler)
		models.append({
			'name': name, 
			'modified': modified, 
			'options': options, 
			'running': running
		})

	return render_template('index.html', models=models)

@app.route('/options')
def options():
	return render_template('options.html')

@app.route('/wordcloud')
def get_wordcloud():
	dictionary_file = '/media/Data/flipvanrijn/datasets/coco/dictionary.pkl'
	with open(dictionary_file) as f:
		dictionary = cPickle.load(f)

	word_cloud = []
	for word, count in dictionary.items():
		word_cloud.append({'text': word, 'weight': count})

	return json.dumps(word_cloud)

@app.route('/model/start/<name>')
def start_model(name):
	''' Start a model based on a model name and the options '''
	if not utils.check_model_running(name):
		utils.start_model(name)

	return ''

if __name__ == '__main__':
	app.run()