import speech_recognition as sr

def speech_to_text(filelocation):
    r = sr.Recognizer()
    english = sr.AudioFile(filelocation)
    with english as eng:
        eng_audio = r.record(eng)
    text_input = r.recognize_google(eng_audio)
    return text_input