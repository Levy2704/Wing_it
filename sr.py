import speech_recognition as sr

# Create a recognizer instance
r = sr.Recognizer()

with sr.Microphone() as source:
    print("Press Enter to start recording...")
    input()  # Wait for Enter key to be pressed
    print("Recording audio... Speak now!")
    audio_data = r.listen(source, timeout=10)  # Listen for audio input
    
    try:
        print("Recognizing...")
        # Convert speech to text using Google Web Speech API
        text = r.recognize_google(audio_data)
        print("Transcribed Text:", text)
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print("Error requesting results from Google Web Speech API; {0}".format(e))
