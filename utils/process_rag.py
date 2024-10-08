from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os , time
from langchain.prompts import PromptTemplate


RAG_prompt = PromptTemplate(

    input_variables=['user_query', 'retrieved_chunks', 'chat_history'],

    template='''
            You are an expert AI and Data Science assistant, and your role is to guide the user in learning data science and AI by asking relevant questions and providing accurate answers. Below is some relevant content from the book and your previous conversation history with user that has been retrieved to help you formulate your response. Based on this retrieved content along with previous chat history and the user's query, generate a helpful response and follow up with a relevant question to keep the conversation going.

            ### Retrieved Content:
            {retrieved_chunks}

            ### User Query:
            {user_query}
            
            ## previous conversation history : 
            {chat_history}

            ### Task:
            - Use the retrieved content to answer the user’s query or build on their answer.
            - Formulate a follow-up question that is directly related to the retrieved content and the user's query to guide the conversation in a relevant way.
            - Ensure the follow-up question tests the user’s understanding or introduces a new concept logically.

            Example workflow:
            1. Acknowledge the user’s query or input.
            2. Provide a concise and accurate response based on the retrieved content.
            3. Ask a relevant question with polite tone based on the user’s response and the book content to maintain an interactive conversation.

            '''
)

script_dir = os.path.dirname(
    os.path.abspath(__file__))  # accessing current dir
project_root = os.path.dirname(script_dir)  # accessing root dir

load_dotenv()
google_api_key = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')


def load_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model='models/gemini-1.5-flash',
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.3,
            convert_system_message_to_human=True
        )
        return llm
    except Exception as e:
        print(f"_Something went wrong while loading LLM_: {e}")


def load_embedding_client():
    try:
        embedding_client = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        return embedding_client
    except Exception as e:
        print(f"_Error loading embedding Client_: {e}")


def init_vectorDB():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        return index
    except Exception as e:
        print(f"_Error preparing vectorDB_: {e}")


def get_topK_embeddings(user_query, embedding_client, vectorDB_client):
    # print(f"function got this arguments --> \nuser_query:{user_query}\nembeddingsclinet:{embedding_client}\nvectroDB:{vectorDB_client}")

    try:
        embeded_query = embedding_client.embed_query(user_query)
        # print("query embeddings generated")
        try:
            retrieved_chunks = vectorDB_client.query(
                vector=embeded_query, top_k=10, include_metadata=True)
        except Exception as e:
            print(e)
        print(retrieved_chunks)
        return retrieved_chunks
    except:
        print("_Error Fetching Embeddings_! \n This could be the reasons : \n"
              "-error regarding embedding client\n"
              "-error during initiliazing vectroDB")

def run_rag_chain(retrieved_chunks, llm, user_query, memory, prompt=RAG_prompt):
    try:
        rag_chain = prompt | llm | StrOutputParser()
        rag_result = rag_chain.invoke({
            "user_query": user_query,
            "retrieved_chunks": retrieved_chunks['matches'][0]['metadata']['text'],
            "chat_history": memory.load_memory_variables({})['chat_history'][-3:]
        })
        # Append the bot's response to memory
        memory.chat_memory.add_ai_message(rag_result)
        
        return rag_result
    except Exception as e:
        print(f"_Error during running of Chain_: {e}")


def rag_pipeline(user_query, llm, embedding_client, pinecone_index, memory, rag_prompt=RAG_prompt):
    try:
        start_time = time.time()
        retrieved_chunks = get_topK_embeddings(
            user_query, embedding_client, pinecone_index)
        rag_result = run_rag_chain(
            retrieved_chunks=retrieved_chunks,
            llm=llm,
            user_query=user_query,
            prompt=rag_prompt,
            memory=memory
        )
        time_taken_for_rag = time.time() - start_time
        return rag_result , time_taken_for_rag
    except Exception as e:
        print(f"-- Error in RAG Pipeline:\n{e}")
