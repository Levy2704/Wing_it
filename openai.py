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

prompts = [
    "What is the capital of France?",
    # Add more prompts here
]

for prompt in prompts:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )
    answer = response.choices[0].text.strip()
    print(f"Q: {prompt}\nA: {answer}\n")