import nltk
import re

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stopWords = set(stopwords.words())

def decontract(phrase):
    phrase = re.sub(r"n\'t"," not",phrase)
    phrase = re.sub(r"\'re"," are",phrase)
    phrase = re.sub(r"\'s"," is",phrase)
    phrase = re.sub(r"\'d"," would",phrase)
    phrase = re.sub(r"\'ll"," will",phrase)
    phrase = re.sub(r"\'t"," not",phrase)
    phrase = re.sub(r"\'ve"," have",phrase)
    phrase = re.sub(r"\'m"," am",phrase)
    return phrase

#Tokenize paragraph to phrases
def tokenize_phrase(paragraph):
    return sent_tokenize(paragraph, "english")

#Tokenize phrase to words
def tokenize_words(phrase):
    return word_tokenize(phrase, "english")

#Remove stop words (words that are of very little meaning)
def remove_stopwords(words):
    wordsFiltered = []
    for word in words:
        if word not in stopWords:
            wordsFiltered.append(word) 
    return wordsFiltered

def stem(words):
    ps = PorterStemmer()
    stemmedWords = []
    for word in words:
        stemmedWords.append(ps.stem(word))
    return stemmedWords

def lemmatize(words):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatizedWords = []
    for word in words:
        lemmatizedWords.append(wordnet_lemmatizer.lemmatize(word))
    return lemmatizedWords

#Tag words in a phrase see https://pythonspot.com/category/nltk/ for the list of speech codes
def tag_phrase(phrase):
    return nltk.pos_tag(tokenize_words(phrase))