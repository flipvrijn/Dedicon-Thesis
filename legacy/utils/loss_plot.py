import re
import argparse
import time
from time import strftime, gmtime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from IPython import embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='log_file', type=str, help='Log file containing Caffe training output.')
    parser.add_argument('-n',dest='interval', type=int, default=10, help='Number of seconds for the refresh interval.')

    args = parser.parse_args()

    plt.ion()
    plt.show()

    interval = args.interval

    print 'Plotting output in %s with %ds intervals' % (args.log_file, interval)
    print 'Press Ctrl+c to quit...'

    def exponential_func(x, a, b, c, d):
        return np.power(a*b,x)+c

    while True:
        # Parse log file
        with open(args.log_file) as f_log:
            losses = []
            secs_per_iter = []
            for line in f_log:
                m = re.search('Iteration (\d+), loss = (\d+\.\d+)$', line)
                if m:
                    losses.append(float(m.group(2)))
                m = re.search('^speed: (.+?)s / iter$', line)
                if m:
                    secs_per_iter.append(float(m.group(1)))

        x_data = range(len(losses))
        y_data = losses
        parameter, covariance_matrix = curve_fit(exponential_func, x_data, y_data)

        x = np.linspace(min(x_data), max(x_data), 1000)
        plt.clf()
        plt.plot(losses, 'b-', label='data')
        plt.plot(x, exponential_func(x, *parameter), 'y-', label='fit')
        plt.title('Loss plot on %s (every %ds), avg %.3fs / iter' % (strftime('%H:%M:%S', gmtime()), interval, np.mean(secs_per_iter)))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        locs, labels = plt.xticks()
        plt.xticks(locs, [str(int(l)*20) for l in locs], rotation=45)
        plt.draw()
        time.sleep(interval)
