import cPickle
import glob
import json
import os.path
import os
import time
import utils
import numpy as np
import shutil
import pandas as pd

import flask
from flask import Flask, render_template, request, url_for, redirect
from flask.ext.navigation import Navigation
from werkzeug import secure_filename

from collections import OrderedDict

config_file = 'config.pkl'
with open(config_file) as f_config:
    config = cPickle.load(f_config)

app = Flask(__name__)
app.config.update(config)

# Navigation bar
nav = Navigation(app)
nav.Bar('top', [
    nav.Item('Modellen', 'models', url='#', items=[
        nav.Item('Getrainde modellen', 'overview_models'),
        nav.Item('Status trainen', 'status_training'),
        nav.Item('Test model', 'test_model'),
        nav.Item('Metrieken', 'metrics'),
    ]),
    nav.Item('Dataset', 'dataset', url='#', items=[
        nav.Item('Context valideren', 'context'),
    ]),
    nav.Item('Opties', 'options')
])

# Mapping of status code to text + class (markup) from models
# that are being trained
building_status_to_text = {
    0: ('doet niks', 'active'),
    1: ('model herladen', 'active'),
    2: ('data laden', 'active'),
    3: ('model bouwen', 'active'),
    4: ('sampler bouwen', 'active'),
    5: ('data optimaliseren', 'active'),
    6: ('training loop', 'active'),
    7: ('NaN gedetecteerd!', 'danger'),
    8: ('opslaan', 'active'),
    9: ('sampling', 'active'),
    10: ('valideren', 'active'),
    11: ('klaar', 'success'),
    12: ('gecrashed', 'danger'),
    13: ('gestopt', 'danger'),
}

# Mapping of status code to text from models
# that are running
running_status_to_text = {
    0: 'niet gestart',
    1: 'opties laden',
    2: 'vocabulair laden',
    3: 'CNN laden',
    4: 'model bouwen',
    5: 'klaar',
}

@app.route('/')
def overview_models():
    ''' Render the overview page with the available models '''
    models = utils.get_models()

    return render_template('models.html', models=models)

@app.route('/overview/status')
def status_training():
    ''' Shows the status of a training model '''
    status_files = OrderedDict()
    files = glob.glob('{}/*_status.json'.format(app.config['MODELS_FOLDER']))
    files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
    one_training = False
    for f in files:
        filename = os.path.split(f)[1]
        name     = filename.split('_status')[0]
        training = utils.check_model_training(name)
        if training:
            one_training = True
        with open(f) as handler:
            status = json.load(handler)
        training_time = time.time() - status['time_created'] if training else \
                        status['time_modified'] - status['time_created']
        training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
        status_files[name] = {
            'status': status['status'],
            'status_text': building_status_to_text[status['status']][0],
            'epoch': status['epoch'],
            'costs': ','.join([str(x) for x in status['costs']]),
            'samples': zip(status['samples'], status['truths']),
            'errors': status['history_errors'],
            'early_stop': status['early_stop'],
            'class': building_status_to_text[status['status']][1],
            'error_message': status['error_message'],
            'training': training,
            'training_time': training_time,
        }

    return render_template('status.html', files=status_files, one_training=one_training)

@app.route('/training/start/<name>')
def start_training(name):
    ''' Starts a model for training '''
    print 'Starting the model {}'.format(name)
    if not utils.training_model_name():
        status = utils.start_training(name)

    return redirect(url_for('status_training'))

@app.route('/training/stop/<name>')
def stop_training(name):
    ''' Stops a model being trained '''
    if utils.check_model_training(name):
        print 'Stop training model {}'.format(name)
        status = utils.stop_training(name)

        # Change status file
        status_file = '{}/{}_status.json'.format(app.config['MODELS_FOLDER'], name)
        with open(status_file, 'r') as handler:
            content = json.load(handler)

        content['status'] = 13
        json.dump(content, open(status_file, 'w'))

    return redirect(url_for('status_training'))

@app.route('/model/metrics/save', methods=['POST'])
def save_metrics():
    ''' Saves metrics to a file '''
    with open(app.config['METRICS_FILE'], 'r') as f:
        metrics = json.load(f)

    name = request.form['metric-name']
    values = request.form['metric-values'].split(',')
    values = [float(v) if v else None for v in values]

    metrics[name] = values

    with open(app.config['METRICS_FILE'], 'w') as f:
        json.dump(metrics, f)

    return flask.jsonify(**{'status': True, 'name': name})

@app.route('/model/metrics', methods=['GET', 'POST'])
def metrics():
    ''' Processes the metrics of a model '''
    if request.method == 'POST':
        metrics = request.form['metrics'].split(',')

        hypotheses_file = request.files['hypotheses_file']
        references_file = request.files['references_file']
        if hypotheses_file and references_file:
            hypotheses_dest_path = os.path.join(app.config['UPLOAD_FOLDER'], '{}.{}'.format(time.time(), hypotheses_file.filename))
            references_dest_path = os.path.join(app.config['UPLOAD_FOLDER'], '{}.{}'.format(time.time(), references_file.filename))
            hypotheses_file.save(hypotheses_dest_path)
            references_file.save(references_dest_path)

        with open(hypotheses_dest_path, 'r') as hypotheses, open(references_dest_path, 'r') as references:
            scores = utils.scores(hypotheses, references, metrics)

        return flask.jsonify(**scores)
    # endif

    # Read metrics
    with open(app.config['METRICS_FILE'], 'r') as f:
        metrics = json.load(f)

    # Get all available hypotheses
    hypotheses_files = glob.glob(os.path.join(app.config['MODELS_FOLDER'], '*.hypotheses.txt'))
    generating_hypos = utils.generating_caps()

    # Mark the best method for each metric
    df = pd.DataFrame(metrics.values())
    col_max = df.idxmax().values

    # List of models
    models = utils.get_models()

    return render_template('metrics.html', metrics=metrics, col_max=col_max, models=models, hypotheses_files=hypotheses_files, generating_hypos=generating_hypos)

@app.route('/captions/generate/<name>')
def generate_captions(name):
    saveto = os.path.join(app.config['MODELS_FOLDER'], name+'.hypotheses.txt')
    status = utils.generate_caps(name, saveto)

    return flask.jsonify(**{'status': status})

@app.route('/context/next')
def context_next():
    data = utils.context_next()

    # Copy image to static folder
    shutil.copy(os.path.join(app.config['IMAGES_FOLDER'], 'train', data['img']), os.path.join(app.config['UPLOAD_FOLDER'], 'context_image.jpg'))

    return flask.jsonify(**data)

@app.route('/context/validate/<id>/<valid>')
def validate_context(id, valid):
    valid = bool(int(valid))
    id    = int(id)

    print 'validating ' + str(id)
    utils.context_validate(id, valid)

    data = utils.context_next()

    print 'sending out ' + str(data['idx'])

    # Copy image to static folder
    shutil.copy(os.path.join(app.config['IMAGES_FOLDER'], 'train', data['img']), os.path.join(app.config['UPLOAD_FOLDER'], 'context_image.jpg'))

    return flask.jsonify(**data)

@app.route('/context/analyse')
def analyse_context():
    return render_template('analyse_context.html')

@app.route('/context')
def context():
    ''' Interface to validate context '''
    stats = utils.context_stats()
    data  = utils.context_next()

    # Copy image to static folder
    shutil.copy(os.path.join(app.config['IMAGES_FOLDER'], 'train', data['img']), os.path.join(app.config['UPLOAD_FOLDER'], 'context_image.jpg'))

    return render_template('context.html', stats=stats, data=data)

@app.route('/options', methods=['GET', 'POST'])
def options():
    ''' Edits the server options '''
    if request.method == 'POST':
        # Update config settings
        config['UPLOAD_FOLDER'] = request.form['upload_folder']
        config['DATA_FOLDER']   = request.form['data_folder']
        config['IMAGES_FOLDER'] = request.form['images_folder']
        config['MODELS_FOLDER'] = request.form['models_folder']
        config['DEBUG']         = bool(int(request.form['debug']))

        app.config.update(config)

        with open(config_file, 'w') as f_config:
            cPickle.dump(dict(app.config.items()), f_config)

    return render_template('options.html')

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

        # Change status file
        json.dump({'status': 0}, open('{}/{}_runningstatus.json'.format(models_directory, name), 'w'))

    return flask.jsonify(**{'status': status})

@app.route('/model/test')
def test_model():
    ''' Renders page for testing the model '''
    name = utils.running_model_name()
    if name:
        status = utils.status_model(name)
        status_text = running_status_to_text[status]
    else:
        status = None
        status_text = None

    return render_template('test.html', name=name, status=status, status_text=status_text)

@app.route('/model/status/<name>')
def status_model(name):
    ''' Returns the status of the model during startup '''
    status = utils.status_model(name)

    return flask.jsonify(**{'status': status, 'status_text': running_status_to_text[status]})

@app.route('/image/random/<introspect>')
def random_image(introspect):
    ''' Pick a random image from the validation set '''
    with open('{}/val2014list.txt'.format(app.config['DATA_FOLDER'])) as f:
        image_names = f.read().split()

    random_idx = np.random.randint(len(image_names))
    random_image = image_names[random_idx]
    src_path = os.path.join(app.config['IMAGES_FOLDER'], 'val', random_image)
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
    ''' Processes the uploaded image '''
    f = request.files['file']
    if f:
        filename = '{}.jpg'.format(time.time())
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(dest_path)

        introspect = bool(int(request.form['introspect']))

        try:
            caption = utils.query_model(dest_path, introspect=introspect)

            return flask.jsonify(**{'caption': caption.split(' '), 'image': filename, 'n': len(caption.split(' ')), 'introspect': introspect})
        except Exception, e:
            print 'Querying model failed: ' + str(e)

            return flask.jsonify(**{'error': str(e)})

@app.route('/image/attention/<image>')
def attention_image(image):
    dest_path = os.path.join('static', 'images', image)

    try:
        caption = utils.query_model(dest_path, introspect=True)

        return flask.jsonify(**{'caption': caption.split(' '), 'image': image, 'n': len(caption.split(' ')), 'introspect': True})
    except Exception, e:
        print 'Querying model failed: ' + str(e)

        return flask.jsonify(**{'error': str(e)})

if __name__ == '__main__':
    app.run()