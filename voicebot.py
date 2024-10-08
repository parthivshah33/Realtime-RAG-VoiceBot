import asyncio
from dotenv import load_dotenv
import os, time
from utils.process_rag import rag_pipeline, load_llm, load_embedding_client, init_vectorDB
from utils.speech_to_text import get_transcript
from utils.text_to_speech import TextToSpeech
from langchain.memory import ConversationBufferMemory

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_KEY")


import time  # Add this import at the top of your voicebot code
from logger import log_latency

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        # Load LLM and other components once
        self.llm = load_llm()
        self.embedding_client = load_embedding_client()
        self.pinecone_index = init_vectorDB()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            # Start timing for speech-to-text
            
            stt_latency = await get_transcript(handle_full_sentence)

            if "goodbye" in self.transcription_response.lower():
                break
                                                                                        
            # Start timing for RAG pipeline
            rag_result, time_taken_for_rag = rag_pipeline(
                user_query=self.transcription_response,
                llm=self.llm,
                embedding_client=self.embedding_client,
                pinecone_index=self.pinecone_index,
                memory=self.memory
            )
            rag_latency = time_taken_for_rag

            # Start timing for text-to-speech
            tts = TextToSpeech()
            time_taken_to_first_b = tts.speak(rag_result)
            tts_latency = time_taken_to_first_b  # Calculate TTS latency

            # Log the latencies
            log_latency(self.transcription_response, stt_latency, rag_latency, tts_latency)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    print("Voicebot Loading, Please Wait!")
    asyncio.run(asyncio.sleep(3))
    print("Welcome to Voicebot!")
    asyncio.run(manager.main())