import nltk
import pandas as pd
import numpy as np
from collections import Counter
import random
import re
from nltk.corpus import stopwords
#nltk.download("stopwords")
import spacy


def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Z0-9a-z_]+', '', text)

    stop_words = set(stopwords.words('italian'))

    tokens = nltk.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(text)
    #tokens = nltk.RegexpTokenizer(r'\w+').tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if not token in stop_words]
    for token in tokens:
        if token.startswith('#'):
            token.replace('#', '')

    return tokens



def get_representation(toate_cuvintele, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features


def write_prediction(out_file, predictions):
    '''A function to write the predictions to a file.
    id,label
    5001,1
    5002,1
    5003,1
    ...
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed

def split(data, labels, validation = 0.25):
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    train_nr = int((1 - validation) * len(labels))
    train_d = data[indici[:train_nr]]
    test_d = data[indici[train_nr:]]
    train_l = labels[indici[:train_nr]]
    test_l = labels[indici[train_nr:]]
    return train_d, test_d, train_l, test_l

def cross_validate(k, data, labels):
    '''Split the data into k chunks.
    iteration 0:
        chunk 0 is for validation, chunk[1:] for train
    iteration 1:
        chunk 1 is for validation, chunk[0] + chunk[2:] for train
    ...
    iteration k:
        chunk k is for validation, chunk[:k] for train
    '''

    chunk_size = len(labels) // k
    indici = np.arange(0, len(labels))
    random.shuffle(indici)

    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i+chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i+chunk_size:]])
        valid = data[valid_indici]
        train = data[train_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid



from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']
toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 2000)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label']

test_data = corpus_to_bow(test_df['text'], wd2idx)



#classifier = MultinomialNB(alpha=2)
#classifier = FirstFFNetwork()
#classifier = MLPClassifier(hidden_layer_sizes=(10,4), activation='logistic', solver='adam',
#                          max_iter=50, learning_rate='adaptive', learning_rate_init=.001)


### cross validate
'''
scores = []

for train_d, test_d, train_l, test_l in cross_validate(10, data, labels):
    classifier.fit(train_d, train_l)
    prediction = classifier.predict(test_d)
    score = f1_score(test_l, prediction)
    scores.append(score)
plot_confusion_matrix(classifier, test_d, test_l)
plt.show()


print(scores)
print(np.mean(scores), ' ', np.std(scores))


'''

### simple split

'''
train_d, test_d, train_l, test_l = train_test_split(data, labels, test_size=0.1, shuffle=True)
classifier.fit(train_d, train_l)
pred = classifier.predict(test_d)
predictions = classifier.predict(test_data)
print(f1_score(test_l, pred))
plot_confusion_matrix(classifier, test_d, test_l)
plt.show()

write_prediction('sample_submission.csv', predictions)
'''

### ensemble

predictions = []
scores = []
one_pred_len = 0
for train_d, test_d, train_l, test_l in cross_validate(25, data, labels):
    classifier = MultinomialNB()
    classifier.fit(train_d,  train_l)
    pred = classifier.predict(test_d)
    prediction = classifier.predict(test_data)
    one_pred_len = len(prediction)
    scores.append(f1_score(test_l, pred))
    predictions.append(prediction)


print(scores)
print(np.mean(scores), np.std(scores))

submission = []
for j in range(one_pred_len):
    ones = 0
    zeros = 0
    for i in range(len(predictions)):
        if predictions[i][j] == 0:
            zeros = zeros + 1
        else:
            ones = ones + 1
    if ones > zeros:
        submission.append(1)
    else:
        submission.append(0)


write_prediction('ensemble_submission.csv', submission)







