import speech_recognition as sr
import pyttsx3
import panel as pn
import openai
import os
from dotenv import load_dotenv

def start_speech_recognition():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)

    # write audio to a WAV file
    with open("microphone-results.wav", "wb") as f:
        f.write(audio.get_wav_data())
        print("Audio saved.")

def make_transctiption():
# Transcription of audio to text 
    with open("microphone-results.wav", "rb") as f:
        transcription = openai.Audio.transcribe("whisper-1", f)
        print("Transcription done.")

    return transcription

def generate_response(transcription):
    response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You are an English tutor. Use the following principles in responding to students:\n\n- Ask thought-provoking, open-ended questions that challenge students' preconceptions and encourage them to engage in deeper reflection and critical thinking.\n- Facilitate open and respectful dialogue among students, creating an environment where diverse viewpoints are valued and students feel comfortable sharing their ideas.\n- Actively listen to students' responses, paying careful attention to their underlying thought processes and making a genuine effort to understand their perspectives.\n- Guide students in their exploration of topics by encouraging them to be more explicite, rather than providing direct answers, to enhance their reasoning and analytical skills.\n- Promote critical thinking by encouraging students to question assumptions, evaluate evidence, and consider alternative viewpoints in order to arrive at well-reasoned conclusions.\n- Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a growth mindset and exemplifying the value of lifelong learning."
    },
    {
      "role": "user",
      "content": f"{transcription}"
    }
  ],
  temperature=0.8,
  max_tokens=1024
)
    return response

def play_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def main(event):    
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    start_speech_recognition()

    text = make_transctiption()
    response = generate_response(text)
    play_text(response["choices"][0]["message"]["content"])


button = pn.widgets.Button(name='Start Speech Recognition', button_type='primary')
button.on_click(main)

app = pn.Column(button)
app.show()
