import gradio as gr
import os
from dotenv import load_dotenv
import threading

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

load_dotenv()

DEEPGRAM_KEY = os.getenv("DEEPGRAM_KEY")

import gradio as gr
import asyncio
import websockets
import json
import os

# Define the WebSocket connection function
async def transcribe_live_audio(audio_filepath):
    async with websockets.connect(
        'wss://api.deepgram.com/v1/listen',
        extra_headers={
            'Authorization': f'Token {DEEPGRAM_KEY}',
        }
    ) as ws:
        with open(audio_filepath, 'rb') as audio_file:
            audio_data = audio_file.read()
            print(audio_data)
            # Send audio data to Deepgram WebSocket
            await ws.send(audio_data)

            # Get and return the transcription from Deepgram
            transcription = ""
            async for message in ws:
                response = json.loads(message)
                if 'channel' in response and 'alternatives' in response['channel']:
                    transcription = response['channel']['alternatives'][0]['transcript']
                    break  # Break after the first transcription
            return transcription

# Gradio function to process live audio stream and return transcription
def process_audio_stream(audio):
    # This will receive the audio file path from Gradio and transcribe it
    audio_filepath = audio

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the transcribe function until complete
    result = loop.run_until_complete(transcribe_live_audio(audio_filepath))
    
    # Close the loop
    loop.close()
    return result

# Gradio interface
audio_interface = gr.Interface(
    fn=process_audio_stream,
    inputs=gr.Audio(type="filepath",streaming=True),
    outputs="text",
    live=True,
    title="Live Audio Transcription",
    description="This app streams live audio and transcribes it using Deepgram."
)

audio_interface.launch()
