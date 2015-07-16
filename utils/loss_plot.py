import re
import argparse
import time
from time import strftime, gmtime
import matplotlib.pyplot as plt

from IPython import embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='log_file', type=str, help='Log file containing Caffe training output.')

    args = parser.parse_args()

    plt.ion()
    plt.show()

    interval = 10

    print 'Plotting output in %s with %ds intervals' % (args.log_file, interval)
    print 'Press Ctrl+c to quit...'

    while True:
        # Parse log file
        with open(args.log_file) as f_log:
            losses = []
            for line in f_log:
                m = re.search('Iteration (\d+), loss = (\d+\.\d+)$', line)
                if m:
                    losses.append(float(m.group(2)))

        plt.plot(losses, 'b-')
        plt.title('Loss plot on %s' % strftime('%H:%M:%S', gmtime()))
        plt.draw()
        time.sleep(interval)