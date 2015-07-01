import json
from HTMLParser import HTMLParser
import nltk
from progress.bar import Bar

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return self.fed

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def sanitize_tags(tags):
    return [sanitize_text(tag) for tag in tags]

def sanitize_text(text):
    return ' '.join(
        [ token for token in nltk.word_tokenize(
            ''.join(
                [ fragment.replace('\n', '').strip().lower() for fragment in strip_tags(text) ]
            )
        ) if token.isalnum() or token in ['.', ',', '!', '?']]
    )

fd  = 'input/flickr_data.json'
fcp = 'output/flickr_data_sanitized.cp.json'
fo  = 'output/flickr_data_sanitized.json'

print 'Loading Flickr JSON data...'
flickr_data = json.load(open(fd, 'r'))

bar = Bar('Sanitizing', max=len(flickr_data['images']), suffix='%(percent)d%%')
for image_id in flickr_data['images']:
    try:
        image_data = flickr_data['images'][image_id]
        if image_data != None:
            # Check if all data is available
            image_data = {
                'tags'          : sanitize_tags(image_data['tags']),
                'description'   : sanitize_text(image_data['description']),
                'title'         : sanitize_text(image_data['title']),
                'url'           : image_data['url']
            }

            flickr_data[image_id] = image_data
        bar.next()
    except:
        # Interrupted and saving to file to, to continue later
        print '\nInterrupted, saving to file...'
        json.dump({'images': flickr_data}, open(fcp, 'w'))

        raise
bar.finish()

print 'Writing Flickr data to file...'
json.dump({'images': flickr_data}, open(fo, 'w'))