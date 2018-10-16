import lib.nlp

def preprocess_phrase(phrase):
    decontracted = lib.nlp.decontract(phrase)
    wordTokens = lib.nlp.tokenize_words(decontracted)
    lemmas = lib.nlp.lemmatize(wordTokens)
    baseWords = lib.nlp.remove_stopwords(lemmas)
    cleanPhrase = ""
    for w in baseWords:
        if len(cleanPhrase) > 0:
            cleanPhrase += " "
        cleanPhrase += w
    return cleanPhrase