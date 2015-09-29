from urlparse import urlparse
import lxml.html
import math
import os
import nltk
import operator
import argparse

max_word_distance = 600

def sigm(x):
    return 10 / (1 + math.e**(-0.15*(x-35)))

def elements_containing(word, data):
    return [d for d in data if word in [w['word'] for w in d['words']]]

def weighted_data(data):
    tag_weights = {
        'img': 6,
        'title': 5,
        'h1': 4,
        'h2': 3.6,
        'h3': 3.35,
        'h4': 2.4,
        'h5': 2.3,
        'h6': 2.2,
        'b': 3,
        'em': 2.7,
        'i': 2.7,
        'strong': 2.5,
    }
    
    weighted_words = {}

    for d in data:
        for w in d['words']:
            for e in elements_containing(w['word'], data):
                weighted_words[w['word']] = weighted_words.get(w['word'], 0) + tag_weights.get(e['elem'].tag, 1) * sigm(w['dist'])

    total_sum = sum([v for v in weighted_words.values()])

    for k, v in weighted_words.items():
        weighted_words[k] = v / total_sum

    return sorted(weighted_words.items(), key=operator.itemgetter(1))

def filter_by_distance(data, threshold):
    new_data = []

    for d in data:
        d['words'] = [w for w in d['words'] if w['dist'] <= threshold]
        if d['words']:
            new_data.append(d)

    return new_data

def filter_by_tokens(data):
    new_data = []

    for d in data:
        valid = False
        for w in d['words']:
            pass # todo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', dest='image_url', type=str, help='URL of the image in the webpage')
    parser.add_argument('--xml', dest='xml_path', type=str, help='Path to XML page')
    parser.add_argument('-t', dest='threshold', type=str, default=600, help='Distance threshold')
    parser.add_argument('-n', dest='n', type=str, default=100, help='N best words')

    args = parser.parse_args()

    img_url = args.image_url
    xml_path = args.xml_path
    dist_threshold = int(args.threshold)
    n = int(args.n)

    image_name = os.path.split(urlparse(img_url).path)[1]

    print 'Processing %s' % image_name

    with open(xml_path) as fh:
        xml = fh.read()

    doc = lxml.html.fromstring(xml)
    print doc.xpath('//img[contains(@src, "%s")]' % image_name)

    title = doc.xpath('//title/text()')[0]

    data = []

    for idx, text in enumerate(doc.xpath('//*/@alt|//*/text()|//img[contains(@src, "%s")]')):
        if text.getparent().tag in ['script', 'style', 'title']:
            continue
        elif text.getparent().tag == 'img' and image_name in text.getparent().attrib['src']:
            data.append({
                'elem': text.getparent(),
                'index': idx,
                'words': [],
                'is_image': True,
            })
        else:
            stripped = text.strip()
            if stripped:
                words = []

                for word in nltk.word_tokenize(stripped):
                    words.append({
                        'word': word,
                        'dist': 0,
                    })
                data.append({
                    'elem': text.getparent(),
                    'index': idx,
                    'words': words,
                    'is_image': False
                })

    idx_image = None
    for idx, d in enumerate(data):
        if d['is_image']:
            idx_image = idx

    for idx, d in enumerate(data):
        current_distance = 0

        subset = []
        if idx == idx_image:
            continue
        elif idx < idx_image:
            subset = data[0:idx_image]

            for s in subset:
                if d['index'] > s['index']:
                    continue
                else:
                    current_distance += len(s['words'])
        elif idx > idx_image:
            subset = data[idx_image+1:]
            for s in subset:
                if d['index'] > s['index']:
                    current_distance += len(s['words'])

        new_words = []
        for idx_word, word in enumerate(d['words']):
            word['dist'] = current_distance + (len(d['words']) - (idx_word + 1)) + 1
            new_words.append(word)
        d['words'] = new_words

    print 'Number of words detected: %s' % sum([sum([1 for word in element['words']]) for element in data])

    data = filter_by_distance(data, dist_threshold)

    data.append({
        'elem': title.getparent(),
        'index': 0,
        'words': [{'word': word, 'dist': 0} for word in nltk.word_tokenize(title)]
    })

    print 'Number of words detected after filtering on word distance: %s' % sum([sum([1 for word in element['words']]) for element in data])

    print weighted_data(data)[-n:]