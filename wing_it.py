import torchaudio
import torch
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained Wav2Vec model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
recording= True
prompts = []

def transcribe_audio(audio_input, sampling_rate):
    input_values = processor(audio_input.squeeze(), return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def record_audio():
    print("Recording audio... Press Enter to stop.")
    audio_data = sd.rec(int(16000 * 10), samplerate=16000, channels=1)
    sd.wait()  # Wait until recording is done
    return audio_data, 16000  # Return audio data and sampling rate

if __name__ == "__main__":
    cont = True
    while cont == True:
        input("Press Enter to start recording...")
        audio_data, sampling_rate = record_audio()
        print("Recording complete")
            
        transcribed_text = transcribe_audio(audio_data, sampling_rate)
        print("Transcribed Text:", transcribed_text)
            
        
        user_input = input("Would you like to keep the prompt? (yes/no): ")
        if user_input.lower() == "yes":
            prompts.append(transcribed_text)
            cont = False
        elif user_input.lower() == "no":
            print("The prompt will not be used")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    



#Install pretrained models
#Open AI will be the answering model and Wav2Vec will be the converting from audio to text model
#Wav2Vec architecture is an Attention model in according with the brief
# %pip install openai
# %pip install transformers torch torchaudio
# %pip install sounddevice
# %pip install IPython

import openai
#Only 10 Dolars on the plan so don't over test it
openai.api_key = "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"


for prompt in prompts:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )
    answer = response.choices[0].text.strip()
    print(f"Q: {prompt}\nA: {answer}\n")

