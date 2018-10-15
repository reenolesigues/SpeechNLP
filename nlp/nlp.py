import nltk

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer

#Tokenize paragraph to phrases
def tokenize_phrase(paragraph):
    return sent_tokenize(paragraph, "english")

#Tag words in a phrase see https://pythonspot.com/category/nltk/ for the list of speech codes
def tag_phrase(phrase):
    return nltk.pos_tag(nltk.word_tokenize(phrase))
    

