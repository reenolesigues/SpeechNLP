import nltk

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

#Tokenize paragraph to phrases
def tokenize_phrase(paragraph):
    return sent_tokenize(paragraph, "english")

#Tokenize paragraph to words
def tokenize_words(paragraph):
    return word_tokenize(paragraph, "english")
    
#Remove stop words (words that are of very little meaning)
def remove_stop_words(paragraph):
    stopWords = set(stopwords.words())
    wordsFiltered = []
    words = tokenize_words(paragraph)
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