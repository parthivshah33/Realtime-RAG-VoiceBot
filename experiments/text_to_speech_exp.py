import requests
import os
from pydub import AudioSegment
from pydub.playback import play
import io
import pyaudio

DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_KEY")

payload = {
    "text": r"Mahendra Singh Dhoni (/məˈheɪndrə ˈsɪŋ dhæˈnɪ/; born 7 July 1981) is an Indian professional cricketer who plays as a right-handed batter and a wicket-keeper."
}

headers = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

# Initialize PyAudio for real-time audio playback
p = pyaudio.PyAudio()

# Define the PyAudio stream settings
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True)

response = requests.post(DEEPGRAM_URL, headers=headers, json=payload, stream=True)

# Processing audio chunks as they are received
for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        # Convert chunk into an audio segment and play it
        audio_segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
        
        # Convert the AudioSegment into raw audio data that can be played by PyAudio
        raw_data = audio_segment.raw_data
        
        # Play the raw audio data directly through the PyAudio stream
        stream.write(raw_data)

# Close the audio stream once the download and playback are complete
stream.stop_stream()
stream.close()
p.terminate()

print("Audio playback complete")
