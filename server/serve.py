from flask import Flask, render_template
import cPickle, json

app = Flask(__name__)
app.debug = True

@app.route('/')
def home():

	return render_template('home.html')

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


if __name__ == '__main__':
	app.run()