import torchaudio
import torch
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained Wav2Vec model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

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
    try:
        while True:
            input("Press Enter to start recording...")
            audio_data, sampling_rate = record_audio()
            print("Recording complete")
            
            transcribed_text = transcribe_audio(audio_data, sampling_rate)
            print("Transcribed Text:", transcribed_text)
    except KeyboardInterrupt:
        print("Recording stopped.")

