######## PIP INSTALL ########
### SpeechRecognition
### nltk
### pyjarowinkler
#############################

import convertor.SpeechConvertor as sc
import lib.nlp as nlp
import string_metric.JaroWinkler as jarowinkler
import lib.text_preprocess as tp
import lib.sentiment_analyzer as sentiment

print("========================= Speech Conversion =========================")

text_input = sc.speech_to_text("resources/english.wav")
print(text_input)
# exit()
# text_input = "There aren't much to do these days. So I'll be working my ass off guys."

print("========================= NLP Processing =========================")

phrases = nlp.tokenize_phrase(text_input)

sentimentRating = 0.0
rateCount = 0

for phrase in phrases:
    print("=== Token ===")
    print("Before : " + phrase)
    print("After : " + str(tp.preprocess_phrase(phrase)))
    sentimentRating += sentiment.analyze(phrase)
    rateCount += 1

sentiment = sentimentRating/rateCount
print("Total Sentiment : " + str(sentiment))
print("========================= END =========================")