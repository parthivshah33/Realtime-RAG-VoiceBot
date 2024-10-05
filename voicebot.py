# import asyncio
# from dotenv import load_dotenv
# import os , time
# from utils.process_rag import rag_pipeline
# from utils.speech_to_text import get_transcript
# from utils.text_to_speech import TextToSpeech


# load_dotenv()
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_KEY")


# class ConversationManager:
#     def __init__(self):
#         self.transcription_response = ""

#     async def main(self):
#         def handle_full_sentence(full_sentence):
#             self.transcription_response = full_sentence

#         # Loop indefinitely until "goodbye" is detected
#         while True:
#             await get_transcript(handle_full_sentence)

#             # Check for "goodbye" to exit the loop
#             if "goodbye" in self.transcription_response.lower():
#                 break

#             rag_result = rag_pipeline(self.transcription_response)

#             tts = TextToSpeech()
#             tts.speak(rag_result)

#             # Reset transcription_response for the next loop iteration
#             self.transcription_response = ""


# if __name__ == "__main__":
#     manager = ConversationManager()
#     print("Voicebot Loading, Please Wait!")
#     asyncio.run(asyncio.sleep(3))
#     print("welcome to Voicebot!")
#     asyncio.run(manager.main())



"""-----------------------------------------------------------------------------------"""

import asyncio
from dotenv import load_dotenv
import os, time
from utils.process_rag import rag_pipeline, load_llm, load_embedding_client, init_vectorDB
from utils.speech_to_text import get_transcript
from utils.text_to_speech import TextToSpeech
from langchain.memory import ConversationBufferMemory

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_KEY")


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        # Load LLM and other components once
        self.llm = load_llm()
        self.embedding_client = load_embedding_client()
        self.pinecone_index = init_vectorDB()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Store memory here

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break

            # Run RAG pipeline and keep memory alive across conversation
            rag_result = rag_pipeline(
                self.transcription_response,
                llm=self.llm,
                embedding_client=self.embedding_client,
                pinecone_index=self.pinecone_index,
                memory=self.memory
            )

            tts = TextToSpeech()
            tts.speak(rag_result)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    print("Voicebot Loading, Please Wait!")
    asyncio.run(asyncio.sleep(3))
    print("Welcome to Voicebot!")
    asyncio.run(manager.main())
