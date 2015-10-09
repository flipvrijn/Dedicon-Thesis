import cPickle
import glob
import json
import os.path
import time
import utils
import numpy as np
import shutil

import flask
from flask import Flask, render_template, request, url_for
from flask.ext.navigation import Navigation
from werkzeug import secure_filename

models_directory = '/media/Data/flipvanrijn/models'
image_directory  = '/media/Data/flipvanrijn/datasets/coco/images'
upload_directory = 'static/uploads'

app = Flask(__name__)
app.debug = True
app.config['UPLOAD_FOLDER'] = upload_directory

# Navigation bar
nav = Navigation(app)
nav.Bar('top', [
    nav.Item('Overzicht modellen', 'overview_models'),
    nav.Item('Overzicht status', 'overview_status'),
    nav.Item('Test model', 'test_model'),
    #nav.Item('Opties', 'options')
])

@app.route('/')
def overview_models():
    ''' Render the overview page with the available models '''

    # List of caption generation models available
    models = []
    for f in glob.glob('{}/*.pkl'.format(models_directory)):
        filename = os.path.split(f)[1]
        name     = filename[:-4] # without the '.pkl'
        modified = time.ctime(os.path.getmtime(f))
        status   = utils.status_model(name)
        with open(f, 'r') as handler:
            options  = cPickle.load(handler)
        models.append({
            'name': name, 
            'modified': modified, 
            'options': options, 
            'status': status
        })

    return render_template('models.html', models=models)

@app.route('/overview/status')
def overview_status():

    status_to_text = {
        0: 'doet niks',
        1: 'model herladen',
        2: 'data laden',
        3: 'model bouwen',
        4: 'sampler bouwen',
        5: 'data optimaliseren',
        6: 'training loop',
        7: 'NaN gedetecteerd!',
        8: 'opslaan',
        9: 'sampling',
        10: 'valideren',
        11: 'klaar',
        12: 'gecrashed',
    }
    status_to_class = {
        0: 'active',
        1: 'active',
        2: 'active', 
        3: 'active', 
        4: 'active', 
        5: 'active', 
        6: 'active',
        7: 'danger',
        8: 'active',
        9: 'active',
        10: 'active',
        11: 'success',
        12: 'danger',
    }

    status_files = {}
    for f in glob.glob('{}/*_status.json'.format(models_directory)):
        filename = os.path.split(f)[1]
        name     = filename.split('_status')[0]
        running  = True#utils.check_training_running(name)
        with open(f) as handler:
            status = json.load(handler)
        status_files[name] = {
            'status': status['status'],
            'status_text': status_to_text[status['status']],
            'epoch': status['epoch'],
            'costs': ','.join([str(x) for x in status['costs']]),
            'samples': zip(status['samples'], status['truths']),
            'errors': status['history_errors'],
            'early_stop': status['early_stop'],
            'class': status_to_class[status['status']],
            'error_message': status['error_message']
        }

    return render_template('status.html', files=status_files)

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

    return flask.jsonify(word_cloud)

@app.route('/model/start/<name>')
def start_model(name):
    ''' Start a model based on a name '''
    status = False
    if not utils.check_model_running(name):
        print 'Starting model {}'.format(name)
        status = utils.start_model(name)

    return flask.jsonify(**{'status': status})

@app.route('/model/stop/<name>')
def stop_model(name):
    ''' Stop a model based on a name '''
    status = False
    if utils.check_model_running(name):
        print 'Stopping model {}'.format(name)
        status = utils.stop_model(name)

    return flask.jsonify(**{'status': status})

@app.route('/model/test')
def test_model():
    ''' Renders page for testing the model '''
    name = utils.running_model_name()

    return render_template('test.html', name=name)

@app.route('/model/status/<name>')
def status_model(name):
    ''' Returns the status of the model during startup '''
    status_to_text = {
        0: 'niet gestart',
        1: 'opties laden',
        2: 'vocabulair laden',
        3: 'CNN laden',
        4: 'model bouwen',
        5: 'klaar',
    }
    status = utils.status_model(name)

    return flask.jsonify(**{'status': status, 'status_text': status_to_text[status]})

@app.route('/image/random/<introspect>')
def random_image(introspect):
    ''' Pick a random image from the validation set '''
    with open('{}/val2014list.txt'.format(image_directory)) as f:
        image_names = f.read().split()

    random_idx = np.random.randint(len(image_names))
    random_image = image_names[random_idx]
    src_path = os.path.join(image_directory, 'val', random_image)
    new_file = '{}.jpg'.format(time.time())
    dest_path = os.path.join('static', 'images', new_file)

    shutil.copy(src_path, dest_path)

    introspect = bool(int(introspect))

    try:
        caption = utils.query_model(dest_path, introspect=introspect)

        return flask.jsonify(**{'caption': caption.split(' '), 'image': new_file, 'n': len(caption.split(' ')), 'introspect': introspect})
    except Exception, e:
        print 'Querying model failed: ' + str(e)
        
        return flask.jsonify(**{'error': str(e)})    

@app.route('/image/upload', methods=['POST'])
def upload_image():
    f = request.files['file']
    if f:
        filename = secure_filename(f.filename)
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(dest_path)

        introspect = bool(int(request.form['introspect']))

        try:
            caption = utils.query_model(dest_path, introspect=introspect)

            return flask.jsonify(**{'caption': caption.split(' '), 'image': filename, 'n': len(caption.split(' ')), 'introspect': introspect})
        except Exception, e:
            print 'Querying model failed: ' + str(e)

            return flask.jsonify(**{'error': str(e)})

        

if __name__ == '__main__':
    app.run()