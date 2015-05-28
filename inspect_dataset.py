import matplotlib.pyplot as plt
import json
from progress.bar import Bar 
import pandas

if __name__ == '__main__':
    json_file = 'datasets/imageclef.json'
    dataset = json.load(open(json_file))

    word_frequency = {}
    bar = Bar('Processing', max=len(dataset['images']))
    for img in dataset['images']:
        for sent in img['sentences']:
            for token in sent['tokens']:
                if token not in word_frequency.keys():
                    word_frequency[token] = 1
                else:
                    word_frequency[token] = word_frequency[token] + 1
        bar.next()
    bar.finish()

    s = pandas.Series(word_frequency)
    s.sort(ascending=False)

    print s.keys()[:100]
    plt.plot(s.values)
    plt.show()