import pickle
import traceback
from collections import Counter

import numpy as np
import progressbar
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''


def init_process(fin, fout):
    outfile = open(fout, 'w')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"', '')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1, 0, 0]
                elif initial_polarity == '2':
                    initial_polarity = [0, 1, 0]
                else:
                    initial_polarity = [0, 0, 1]

                tweet = line.split(',')[-1].strip()
                outline = str(initial_polarity) + ':::' + tweet + '\n'
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()


# init_process('training.1600000.processed.noemoticon.csv', 'train_set.csv')
# init_process('testdata.manual.2009.06.14.csv', 'test_set.csv')


def create_lexicon(fin):
    words_list = []
    lexicon = []
    with open(fin, 'r', buffering=1000000, encoding='latin-1') as f:
        try:
            bar = progressbar.ProgressBar(max_value=1600000)
            i = 0
            for line in f:
                i += 1
                tweet = line.split(':::', 1)[1]
                words = word_tokenize(tweet.lower())
                for w in words:
                    words_list.append(lemmatizer.lemmatize(w))

                if i % 1000 == 0:
                    bar.update(i)

        except Exception as e:
            traceback.print_stack()
            print(str(e))

    word_counts = Counter(words_list).most_common()
    for w in word_counts:
        lexicon.append(w)

    print(len(lexicon))

    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)


create_lexicon('train_set.csv')


def create_lexicon_dict(lexicon_pickle):
    lexicon_dict = {}

    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
        for i, w in enumerate(lexicon):
            lexicon_dict[w] = i

    with open('lexicon-dict.pickle', 'wb') as f:
        pickle.dump(lexicon_dict, f)


# create_lexicon_dict('lexicon.pickle')


def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = list(features)
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)

        print(counter)


# # convert_to_vec('test_set.csv', 'processed-test-set.csv', 'lexicon-2500-2638.pickle')
# # convert_to_vec('train_set.csv', 'processed-train-set.csv', 'lexicon-2500-2638.pickle')


# def shuffle_data(fin):
#     df = pd.read_csv(fin, error_bad_lines=False)
#     df = df.iloc[np.random.permutation(len(df))]
#     print(df.head())
#     df.to_csv('train_set_shuffled.csv', index=False)
# shuffle_data('train_set.csv')


def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    # feature_sets = np.array(feature_sets)
    # labels = np.array(labels)


# # create_test_data_pickle('processed-test-set.csv')

def minify_sample(fin, fout):
    output_file = open(fout, 'w')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            output_file.write(line)
            counter += 1
            if counter >= 100000:
                break
    output_file.close()

# minify_sample('train_set_shuffled.csv', 'sample_set.csv')
