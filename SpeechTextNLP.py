######## PIP INSTALL ########
### SpeechRecognition
### nltk
### pyjarowinkler
#############################

import convertor.SpeechConvertor as sc
import nlp.nlp as nlp
import string_metric.JaroWinkler as jarowinkler
import nlp.sentiment as sentiment

print("========================= Speech Conversion =========================")

# text_input = sc.speech_to_text("resources/english.wav")
# print(text_input)
# exit()
text_input = "This is a demo sentence. I am not good in english."

print("========================= NLP Processing =========================")

phrases = nlp.tokenize_phrase(text_input)
# print(phrases)

# for phrase in phrases: 
#     print(phrase)
#     print(nlp.tag_phrase(phrase))

text_req = "The quick  brown fox jumps over the lazy dog"
 
for phrase in phrases:
    print("=== Token ===")
    print("Analyzing phrase : " + phrase)
    # print("Stemmed:" + str(nlp.stem(nlp.word_tokenize(text_input))))
    print("Lemmatized:" + str(nlp.lemmatize(nlp.word_tokenize(text_req))))
    # print("Removed Stopwords:" + str(nlp.remove_stop_words(text_input)))
    # print("Similarity : " + repr(jarowinkler.metrics(phrase, text_req)))
    # print("Positivity: " + sentiment.analyze(phrase))
print("========================= END =========================")