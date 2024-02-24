import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import smtplib

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')


engine.setProperty('voices',voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")

    elif hour >= 12 and hour < 16:
        speak("Good Afternoon!")

    else: 
        speak("Good Evening!")

    speak("I am Jarvis. Please tell me how may i assisst you Sir")

def takeCommand():
    """ It takes microphone input from the user and returns string output """

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)
        print("Say that again please....")
        return "None"
    return query

def sendEmail(to, content):
    server = smtplib.SMTP("smtp.gamil.com", 587)
    server.ehlo()
    server.starttls()
    server.login("sender@email.com", "*******")                ## Write email that you want to send from.
    server.sendmail("sender@email.com", to, content)
    server.close()


if __name__ == '__main__':
    wishMe()
    while True:
        query = takeCommand().lower()

        # Logic for executing tasks based on query
        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("Wikipedia", "")
            results = wikipedia.summary(query, sentences = 2)
            speak("According to wikipedia")
            print(results)
            speak(results)

        elif 'open youtube' in query: 
            webbrowser.open("youtube.com") 

        elif 'open google' in query: 
            webbrowser.open("google.com")  

        elif 'open stackoverflow' in query: 
            webbrowser.open("stackoverflow.com") 

        elif 'open youtube' in query: 
            webbrowser.open("youtube.com") 

        elif 'play music' in query:
            music_dir = 'D:\\music'
            songs = os.listdir(music_dir)
            print(songs)
            # song = random(songs)
            os.startfile(os.path.join(music_dir, songs[0]))

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"Sir the time is {strTime}")    

        elif 'open python' in query:                                ## It will open pycharm
            Pycharm_dir = "C:\\Program Files\\JetBrains\\PyCharm Community Edition 2023.2.2\\bin\pycharm64.exe" ## Pycharm Directory or location
            os.startfile(Pycharm_dir)
            speak("Ok")

        elif 'email to faizan' in query:
            try:
                speak("What should I send?")
                content = takeCommand()
                to = "receiver@email.com"                  ## Write email that you want to send to.
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                print(e)
                speak("Sorry sir i am not able to send this email")

    












