import psutil
import subprocess
import thread
import re
import os.path
import socket
import json
import struct

models_directory = '/media/Data/flipvanrijn/models'

def check_model_running(name):
    ''' Checks whether a model is currently running '''
    for process in psutil.process_iter():
        try:
            if 'python2.7' in process.name() and name in ' '.join(process.cmdline()):
                return True
        except psutil.Error:
            pass

    return False

def running_model_name():
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
                '--prototxt={}/VGG_ILSVRC_19_layers_deploy.prototxt'.format(models_directory),
                '--caffemodel={}/VGG_ILSVRC_19_layers.caffemodel'.format(models_directory)]
            subprocess.Popen(cmd)
        thread.start_new_thread(run_server, ())

        return True
    except:
        return False

def stop_model(name):
    ''' Stops a model by name '''
    for process in psutil.process_iter():
        try:
            if 'python2.7' in process.name() and name in ' '.join(process.cmdline()):
                process.kill()

                # Change status file
                json.dump({'status': 0}, open('{}/{}_runningstatus.json'.format(models_directory, name), 'w'))
                return True
        except psutil.Error:
            pass
    return False

def query_model(image_path, introspect):
    # Send it to the server via socket
    HOST, PORT = "localhost", 9999
    data = '{};{}'.format(image_path, 1 if introspect else 0)

    # Create a socket (SOCK_STREAM means a TCP socket)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        data = struct.pack('>I', len(data)) + data
        sock.sendall(data)

        # Receive data from the server and shut down
        received = sock.recv(1024)
    finally:
        sock.close()

    return received