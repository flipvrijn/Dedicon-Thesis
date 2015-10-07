import psutil
import subprocess

def check_model_running(name):
	''' Checks whether a model is currently running '''
	for process in psutil.process_iter():
		try:
			if 'python' in process.name() and name in process.name():
				return True
		except psutil.Error:
			pass

	return False

def start_model(name):
	''' Starts a model by name '''
	models_directory = '/media/Data/flipvanrijn/models/'.rstrip('/')
	cmd = ['python2.7', 'model_daemon.py', '--model={}/{}'.format(models_directory, name), '--options={}/{}.pkl'.format(models_directory, name)]
	subprocess.Popen(cmd)