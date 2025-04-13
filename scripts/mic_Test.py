import speech_recognition as sr

recorder = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something...")
    audio = recorder.listen(source)
    print("Audio captured!")