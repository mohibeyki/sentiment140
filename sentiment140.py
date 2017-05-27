import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def generator(source='sample_set.csv', batch_size=128):
    with open('lexicon-dict.pickle', 'rb') as f:
        lexicon = pickle.load(f)
    while True:
        with open(source, 'r') as f:
            i = 0
            for line in f:
                label, tweet = line.split(':::', 1)

                current_words = word_tokenize(tweet.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]

                batch_features = np.zeros((batch_size, len(lexicon)), dtype=np.int)
                batch_labels = np.zeros((batch_size, 3), dtype=np.int)

                for word in current_words:
                    if word in lexicon:
                        batch_features[i][lexicon[word]] += 1

                batch_labels[i] = eval(label)

                i += 1
                if i % batch_size == 0:
                    i = 0
                    yield (
                        batch_features,
                        batch_labels
                    )


def parse_data(input_file='test_set.csv'):
    with open('lexicon-dict.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    x_list = []
    y_list = []

    with open(input_file, 'r') as f:
        for line in f:
            label, tweet = line.split(':::', 1)

            current_words = word_tokenize(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word in lexicon:
                    features[lexicon[word]] += 1

            x_list.append(list(features))
            y_list.append(eval(label))

    return x_list, y_list
