from langchain_google_genai import ChatGoogleGenerativeAI
import os
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values
load_dotenv()


import asyncio
import websockets
import json
import pygame
import os

# Initialize pygame mixer for audio playback
pygame.mixer.init()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_KEY")
print(DEEPGRAM_API_KEY)

async def text_to_speech(ws, text):
    # Send text to Deepgram for TTS
    await ws.send(json.dumps({
        "text": text,
        "voice": "en-US-Wavenet-D",  # Change voice as needed
        "speed": 1.0,  # Change speed as needed
        "pitch": 0.0   # Change pitch as needed
    }))

    # Receive audio response from Deepgram
    audio_response = await ws.recv()
    audio_data = json.loads(audio_response)

    if 'audio' in audio_data:
        audio_content = audio_data['audio']
        play_audio(audio_content)  # Function to handle audio playback

def play_audio(audio_content):
    # Write the audio content to a temporary WAV file for playback
    temp_audio_file = "temp_audio.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_content)

    # Load and play the audio file using pygame
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing before cleaning up
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Clean up the temporary audio file
    os.remove(temp_audio_file)

async def stream_text_from_llm(llm):
    async with websockets.connect(
        'wss://api.deepgram.com/v1/speak?encoding=linear16&sample_rate=48000',
        extra_headers={
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
        }
    ) as ws:
        async for chunk in llm.stream(text):  # Assuming `text` is your input for the LLM
            print(chunk.content)  # Output chunk to console
            await text_to_speech(ws, chunk.content)



# Run the main streaming function
if __name__ == "__main__":
    # Replace with your actual LLM instance and input text
    class MockLLM:
        async def stream(self, text):
            for i in range(3):
                yield f"Chunk {i + 1} from: {text}"
                await asyncio.sleep(1)  # Simulate delay between chunks

    llm = ChatGoogleGenerativeAI(
    model='models/gemini-1.5-flash',
    google_api_key=os.getenv('GEMINI_API_KEY'),
    convert_system_message_to_human=True)

    text = "how MSDHONI made india won 2011 ICC Men's Cricket World cup?"


    asyncio.run(stream_text_from_llm(llm))


