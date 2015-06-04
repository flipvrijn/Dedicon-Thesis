from urlparse import urlparse
import lxml.html
import math
import os
import nltk

max_word_distance = 600

def sigm(x):
    return 10 / (1 + math.e**(-0.15*(x-35)))

def word2img_distance():
    pass

if __name__ == '__main__':
    img_url = 'http://img.tfd.com/wn/C1/63998-timepiece.jpg'
    image_name = os.path.split(urlparse(img_url).path)[1]

    print 'Processing %s' % image_name

    with open('input/page.html') as fh:
        xml = fh.read()

    doc = lxml.html.fromstring(xml)
    print doc.xpath('//img[contains(@src, "%s")]' % image_name)

    data = []

    for idx, text in enumerate(doc.xpath('//*/@alt|//*/text()|//img[contains(@src, "%s")]')):
        if text.getparent().tag in ['script', 'style']:
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
                        'dist': None,
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
            subset = data[idx_image+1:][::-1]
            pass    

        new_words = []
        for idx_word, word in enumerate(d['words']):
            word['dist'] = current_distance + (len(d['words']) - (idx_word + 1))
            new_words.append(word)
        d['words'] = new_words

    print data