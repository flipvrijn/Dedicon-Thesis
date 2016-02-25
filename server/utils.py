import psutil
import subprocess
import thread
import re
import os.path
import os
import socket
import json
import struct
import cPickle
import time 
import glob

import h5py
import numpy as np

from PIL import Image
from collections import OrderedDict

from IPython import embed

# this requires the coco-caption package, https://github.com/tylin/coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from serve import config
import handle_book

models_directory = config['MODELS_FOLDER']
network_directory = '../models/attention'
context_file = '{}/../sents_train.h5'.format(config['IMAGES_FOLDER'])

def scores(hypo, refs, metrics):
    ''' Returns all requested scores '''
    def _load_file(hypotheses, references):
        hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypotheses)}
        refs = {idx: lines.strip().split('<>') for (idx, lines) in enumerate(references)}
        if len(hypo) != len(refs):
            raise ValueError("There is a sentence number mismatch between the inputs")
        return hypo, refs

    hypo, refs = _load_file(hypo, refs)

    metric_to_scorer = {
        'bleu': Bleu(4),
        'meteor': Meteor(),
        'rouge': Rouge(),
        'cider': Cider(),
    }
    print metrics

    final_scores = {}
    for metric, scorer in metric_to_scorer.items():
        if metric in metrics:
            score, scores = scorer.compute_score(refs, hypo)
            final_scores[metric] = score
    return final_scores

def format_duration(tdelta):
    ''' Formats a timedelta object into hours:minutes:seconds where hours can be more than 24. '''
    hours, rem = divmod(tdelta.total_seconds(), 3600)
    minutes, seconds = divmod(rem, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)

def context_stats():
    handler = h5py.File(context_file, 'r')

    # Only the ones with actual context
    has_title = np.where(handler['context/titles'])[0]
    has_description = np.where(handler['context/descriptions'])[0]
    not_empty = np.unique(np.concatenate((has_title, has_description)))

    # Check whether there are tags
    mask = np.ones(handler['context/titles'].shape[0], dtype='bool')
    mask[not_empty] = False
    ix   = np.where(mask)[0]
    tags = handler['context/tags'][mask]
    # Holds the indices to samples that neither have a title,
    # description nor tags
    no_tags_indices = [] 
    for i, t in zip(ix, tags):
        if not t.size:
            no_tags_indices.append(i)

    valid = np.where(handler['context/valid'].value == 1)[0]

    data = {
        'total': handler['context/titles'].shape[0],
        'non_empty': handler['context/titles'].shape[0] - len(no_tags_indices),
        'valid': valid.shape[0],
    }

    handler.close()

    return data

def context_next():
    np.random.seed(1234)
    handler = h5py.File(context_file, 'r')

    todo_indices = np.where(handler['context/valid'].value == 0)[0]
    np.random.shuffle(todo_indices)

    next_idx = todo_indices[0]

    tag_idxs = handler['context/tags'][next_idx].tolist()
    tag_idxs = sorted(tag_idxs)

    # ugly, because logic?!: "src and dest data spaces have different sizes"
    tags = []
    for tag_i in tag_idxs:
        tags.append(handler['sentences/tags'][tag_i])

    data = {
        'idx': next_idx,
        'title': handler['context/titles'][next_idx],
        'description': handler['context/descriptions'][next_idx],
        'tags': tags,
        'img': handler['sentences/image_names'][next_idx],
    }

    handler.close()

    return data

def context_validate(id, valid):
    handler = h5py.File(context_file, 'r+')
    handler['context/valid'][id] = 1 if valid else -1
    handler.close()

def check_model_running(name):
    ''' Checks whether a model is currently running '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and 'model_server.py' in cmd and name in cmd:
                return True
        except psutil.Error:
            pass

    return False

def check_model_training(name):
    ''' Checks whether a model is being trained '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and 'train.py' in cmd and name in cmd:
                return True
        except psutil.Error:
            pass

    return False

def running_model_name():
    ''' Returns the name of the model that is currently running '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and 'model_server.py' in cmd and '.npz' in cmd and '.pkl' in cmd:
                pattern = re.compile('\/(.*?\.npz)')
                match   = pattern.search(cmd)
                if match:
                    return os.path.split(match.group())[1]
                return None
        except psutil.Error:
            pass

    return None

def get_models():
    ''' Returns a list of all models known to the server '''
    models = []
    files = glob.glob('{}/*.pkl'.format(models_directory))
    files.sort(key=lambda x: x, reverse=False)
    for f in files:
        filename = os.path.split(f)[1] # only the filename
        name     = filename[:-4] # without the '.pkl'
        modified = time.ctime(os.path.getmtime(f))
        status   = status_model(name)
        with open(f, 'r') as handler:
            options  = cPickle.load(handler)
            options = OrderedDict(sorted(options.items()))
        models.append({
            'name': name, 
            'modified': modified, 
            'options': options, 
            'status': status
        })

    return models

def get_model_options(name):
    with open('{}/{}.pkl'.format(models_directory, name)) as f:
        options = cPickle.load(f)

    return options

def rename_model(old, new):
    ''' Renames a model (pure cosmetic) '''
    # first rename all files
    for filename in os.listdir(models_directory):
        if filename.startswith(old):
            os.rename(os.path.join(models_directory, filename), os.path.join(models_directory, new + filename[len(old):]))
            
    # then modify the options
    opts_file = '{}/{}.pkl'.format(models_directory, new)
    with open(opts_file, 'rb') as f:
        opts = cPickle.load(f)
        opts['saveto'] = new
    with open(opts_file, 'wb') as f:
        cPickle.dump(opts, f)

def training_model_name():
    ''' Start training procedure for a model '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and 'train.py' in cmd and '.npz' in cmd:
                pattern = re.compile('\s(.*?\.npz)')
                match   = pattern.search(cmd)
                if match:
                    return match.groups()[0]
                return None
        except psutil.Error:
            pass

    return None

def status_model(name):
    ''' Reads the status from the model '''
    if check_model_running(name):
        content = json.load(open('{}/{}_runningstatus.json'.format(models_directory, name)))
        return content['status']

    return 0

def start_model(name):
    ''' Starts a model by name '''
    try:
        json.dump({'status': 0}, open('{}/{}_runningstatus.json'.format(models_directory, name), 'w'))

        def run_server():
            cmd = ['python2.7', 'model_server.py',
                '--model={}/{}'.format(models_directory, name), 
                '--options={}/{}.pkl'.format(models_directory, name),
                '--prototxt={}/cnn/VGG_ILSVRC_19_layers_deploy.prototxt'.format(models_directory),
                '--caffemodel={}/cnn/VGG_ILSVRC_19_layers.caffemodel'.format(models_directory),
                '--w2vmodel={}/word2vec/enwiki-latest-pages.512.bin'.format(models_directory),
                '--tfidfmodel={}/tfidf/tfidf_model.pkl'.format(models_directory),
                '--svdmodel={}/tfidf/svd_model.pkl'.format(models_directory),
                '--rawmodel={}/raw/raw_model.pkl'.format(models_directory)]
            subprocess.Popen(cmd)
        thread.start_new_thread(run_server, ())

        return True
    except:
        return False

def stop_model(name):
    ''' Stops a model by name '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and name in cmd:
                process.kill()

                return True
        except psutil.Error:
            pass
    return False

def start_training(model_name, model_type, data_folder, data_type, data_type_params, extra_params):
    ''' Starts the trainer for a model '''
    try:
        def run_trainer():
            cmd = ['python2.7', 
                os.path.join(network_directory, 'train.py'), 
                data_folder,
                models_directory,
                '{}.npz'.format(model_name),
                '--type={}'.format(model_type),
                '--preproc_type={}'.format(data_type),
                '--preproc_params={}'.format(','.join(['{}={}'.format(k,v) for k,v in data_type_params.items()])),]
            if extra_params:
                cmd += [extra_params]
            print cmd
            subprocess.Popen(cmd)
        thread.start_new_thread(run_trainer, ())

        return True
    except:
        return False

def stop_training(name):
    ''' Stops a model being trained '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and 'train.py' in cmd and name in cmd:
                process.kill()

                return True
        except psutil.Error:
            pass

    return False

def resize_image(image_path, resize=256, crop=224):
    ''' Resizes an image according to the resizing scheme of the captioning models '''
    image = Image.open(image_path)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize
    left = (width  - crop) / 2
    top  = (height - crop) / 2
    image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
    image_resized = image_resized.convert('RGB')

    image_resized.save(image_path)

def query_model(image_path, introspect, text=''):
    ''' Queries the caption model with an image (and context?) '''

    def recvall(sock, n):
        data = ''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(sock):
        raw_length = recvall(sock, 4)
        if not raw_length:
            return None
        length = struct.unpack('>I', raw_length)[0]
        return recvall(sock, length)

    # Send it to the server via socket
    HOST, PORT = "localhost", 9999
    
    # Read image and serialize it
    img = Image.open(image_path)
    data = {
        'img': {
            'pixels': img.tobytes(),
            'size': img.size,
            'mode': img.mode,
        },
        'text': text,
        'introspect_image': introspect,
        'introspect_context': introspect,
        'output_path': image_path
    }
    data = cPickle.dumps(data)

    # Create a socket (SOCK_STREAM means a TCP socket)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        data = struct.pack('>I', len(data)) + data
        sock.sendall(data)

        # Receive data from the server and shut down
        received = cPickle.loads(recv_msg(sock))
    finally:
        sock.close()

    return received['captions'][0]

def generate_caps(name, saveto):
    ''' Starts generator for caps of dataset '''
    try:
        def gen_caps():
            cmd = ['python2.7', 
                os.path.join(network_directory, 'generate_caps.py'), 
                os.path.join(models_directory, name),   # model input
                saveto]                                 # saveto
            subprocess.Popen(cmd)
        thread.start_new_thread(gen_caps, ())

        return True
    except:
        return False

def generating_caps():
    ''' Checks if caps are being generated '''
    for process in psutil.process_iter():
        try:
            cmd = ' '.join(process.cmdline())
            if 'python2.7' in process.name() and 'generate_caps.py' in cmd:
                return True
        except psutil.Error:
            pass

    return False

def start_book_parser(path):
    ''' Starts the book parser to extract images and captions '''
    try:
        def parse_book():
            cmd = ['python2.7', 
                os.path.join('handle_book.py'), 
                path,
                '-i',
                '--temp={}'.format(os.path.join(config['IDB_FOLDER'], 'temp')),
                '--db_file={}'.format(os.path.join(config['IDB_FOLDER'], config['IDB_FILE']))]
            subprocess.Popen(cmd)
        thread.start_new_thread(parse_book, ())

        return True
    except:
        return False

def export_book(loisID, path):
    ''' Exports the book with the new captions '''
    
    handle_book.handle_export({
        'loisid': loisID,
        'filename': path,
        'temp_dir': os.path.join(config['IDB_FOLDER'], 'temp'),
        'db_file': os.path.join(config['IDB_FOLDER'], config['IDB_FILE']),
    })