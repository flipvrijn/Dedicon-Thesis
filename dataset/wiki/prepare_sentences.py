from nltk.tokenize import word_tokenize

def main():
    with open('/media/Data_/flipvanrijn/datasets/text/enwiki', 'r') as f_in, open('/media/Data_/flipvanrijn/datasets/text/enwiki-sentences', 'w') as f_out:
        counter = 0
        for line in f_in:
            tokens = word_tokenize(line)
            if tokens:
                print >>f_out, ' '.join(tokens)
            counter += 1
            if counter % 100000 == 0:
                print counter

if __name__ == '__main__':
    main()
