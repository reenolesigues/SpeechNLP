import lib.text_preprocess as tp
import nltk.classify.util

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

def word_feats(words):
    return dict([(word, True) for word in words])

positive_vocab = [line.strip('\n') for line in open('resources/positive_training.set')]
negative_vocab = [line.strip('\n') for line in open('resources/negative_training.set')]
neutral_vocab = [line.strip('\n') for line in open('resources/neutral_training.set')]

positive_features = [(word_feats(tp.preprocess_phrase(pos)), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(tp.preprocess_phrase(neg)), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(tp.preprocess_phrase(neu)), 'neu') for neu in neutral_vocab]

train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)

def analyze(phrase):
    neg = 0
    pos = 0
    phrase = phrase.lower()
    words = phrase.split(' ')
    for word in words:
        classResult = classifier.classify(word_feats(word))
        if classResult == 'neg':
            neg += 1
        if classResult == 'pos':
            pos += 1
    # print('Positive: ' + str(float(pos)/len(words)))
    # print('Negative: ' + str(float(neg)/len(words)))
    return float(pos)/len(words)