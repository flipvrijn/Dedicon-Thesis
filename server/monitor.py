import json
from collections import deque

class Monitor(object):

    '''
    Status codes:
    0: idle
    1: reloading model
    2: loading data
    3: building model
    4: building sampler
    5: optimizing
    6: training loop
    7: nan detected
    8: saving
    9: sampling
    10: validate
    11: done
    12: failed
    '''

    def __init__(self, monitor_file):
        self.monitor_file = monitor_file
        self.window_size  = 1000
        self._status      = 0
        self._epoch       = 0
        self._update      = 0
        self._estop       = False
        self._truths      = []
        self._samples     = []
        self._cost        = deque(maxlen=self.window_size)
        self._history_errors = []
        self.options      = {}
        self.dictionary   = {}
        self.error_message = ''

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        self.save()
        
    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.save()

    @property
    def update(self):
        return self._update

    @update.setter
    def update(self, value):
        self._update = value
        self.save()

    @property 
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost.append(value)
        self.save()

    @property 
    def estop(self):
        return self._estop

    @estop.setter
    def estop(self, value):
        self._estop = True
        self.save()

    @property 
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value
        self.save()

    @property 
    def truths(self):
        return self._truths

    @truths.setter
    def truths(self, value):
        self._truths = value
        self.save()

    @property 
    def history_errors(self):
        return self._history_errors

    @history_errors.setter
    def history_errors(self, value):
        self._history_errors = value
        self.save()

    def save(self):
	if 'monitor' in self.options.keys():
		# Remove itself from the options
		del self.options['monitor']

        # Save data
        data = {
            'status'        : self._status,
            'epoch'         : self._epoch,
            'costs'         : list(self._cost),
            'options'       : self.options,
            'dictionary'    : self.dictionary,
            'early_stop'    : self._estop,
            'truths'        : self._truths,
            'samples'       : self._samples,
            'history_errors': self._history_errors,
            'error_message' : self.error_message,
        }

        json.dump(data, open(self.monitor_file, 'w'))
