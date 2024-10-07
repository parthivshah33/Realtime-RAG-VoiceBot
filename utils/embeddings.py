# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# import instructor
# from InstructorEmbedding import instructor
# from helper import batch_chunks
# import pandas as pd

import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
# import torch
# import chromadb
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# # Determine the device to use and print GPU details if available
# if torch.cuda.is_available():
#     device = 'cuda'
#     gpu_details = torch.cuda.get_device_properties(0)  
#     print(f"Using GPU: {gpu_details.name}")
#     print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
#     print(f"  Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
#     print(f"  Total Memory: {gpu_details.total_memory / 1e9:.2f} GB")
# else:
#     device = 'cpu'
#     print("Using CPU")


'''Initialiazing Embeddings and Chroma Client'''
embedding_client = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", task_type="retrieval_document", google_api_key="AIzaSyB67yE_ZKd5BKhLluTEi767CzzALRzt-4Q")

''' -- PINECONE -- '''
PINECONE_API_KEY = "29dad76e-dc0c-4a43-b876-b5bdd748601c"
PINECONE_API_ENV = "us-east-1"
pinecone_index_name = "voicebot"

# '''-- Make persistant directory path if It's not already exists --'''
# script_dir = os.path.dirname(os.path.abspath(
#     __file__))
# project_root = os.path.dirname(script_dir)
# persist_directory = os.path.join(project_root, "data", "chroma")

# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)
#     print(f"Directory created: {persist_directory}")
# else:
#     print(f"Directory already exists: {persist_directory}")

# ''' ---- Make chroma client instance ----'''
# chroma_client = chromadb.PersistentClient(path=str(persist_directory))
# print(f"{chroma_client} chroma client made at {str(persist_directory)}")
# collection_name = "rag_aiml_data"


def main_pipeline(file_path):
    documents = load_pdf(file_path=file_path)
    print("=== dataframe loaded ===")

    prepare_and_inject_embeddings(documents=documents,embedding_client=embedding_client)
    print("_____ embeddings generated and Injected to PINECONE Server ______")

    # chroma_collection = build_chroma_collection()
    # print("____ VectorDB Client invoked _____")

    # add_embeddings_to_chroma_collection(
    #     embeddingsDataList=embeddingsDataList, chromaCollection=chroma_collection)

    # validate_embedding_injection(collectionName=chroma_collection)


def load_pdf(file_path: str):
    """
        Load a DataFrame from the specified CSV or Excel file.

        Args:
            file_directory (str): The directory path of the file to be loaded.

        Returns:
            list of DataFrame, str: The loaded DataFrame and the file's base name without the extension.

        Raises:
            ValueError: If the file extension is neither CSV nor Excel.
        """
    baseFileName = os.path.basename(file_path)
    print(baseFileName)
    file_name, file_extension = os.path.splitext(baseFileName)
    if file_extension == ".pdf":
        documents = PyPDFLoader(file_path).load()
        return documents
    else:
        raise ValueError("The selected file type is not supported")


def prepare_and_inject_embeddings(documents,embedding_client):
    """
    -prepare documents for data injection,
    -Generate Embeddigngs along with Injetion to Server.

    Args:
        documents (): Object of Loaded PDF Documents from Docuement Loader Class.
        file_name (str): The base name of the file for use in metadata.

    Returns: None, prints success message after injetion
    """
    
    chunk_size = 400
    chunk_overlap = 20
    length_function = len
    is_separator_regex = False

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex
    )

    doc_chunks = text_splitter.split_documents(documents)
    print(f"--- Total Documents : {len(doc_chunks)} ---")
    
    vectorstore_from_docs = PineconeVectorStore.from_documents(doc_chunks,
    index_name=pinecone_index_name,
    embedding=embedding_client)
    
    print("==============================")
    print(" -- Data is stored in PINECONE Index -- ")


# def build_chroma_collection():

#     collection = chroma_client.get_or_create_collection(
#         name=collection_name)

#     print(
#         f"ChromaDB Collection has been build!, collection object : {collection}")
#     return collection


# def add_embeddings_to_chroma_collection(embeddingsDataList, chromaCollection):
#     print(
#         f"<<<<<<<<<<<<<-------------------->{type(chromaCollection)}---------------------->>>>>>>>>>>>>>")
#     try:
#         docs, metadatas, ids, embeddings = embeddingsDataList
#         collection = chromaCollection
#         print(
#             f"___________starting adding embeddings to {collection}___________")
#         collection.add(
#             documents=docs,
#             metadatas=metadatas,
#             embeddings=embeddings,
#             ids=ids
#         )
#         print("==============================")
#         print("Data is stored in ChromaDB.")

#     except:
#         print("---Failed to add data to chroma collection---")


# def validate_embedding_injection(collectionName):
#     """
#         Validate the contents of the database to ensure that the data injection has been successful.
#         Prints the number of vectors in the ChromaDB collection for confirmation.
#     """

#     global chroma_client
#     vectordb = chroma_client.get_collection(
#         name=collection_name)
#     print("==============================")
#     print("Number of vectors in vectordb:", vectordb.count())
#     print("==============================")


# Run the code as a main function
if __name__ == "__main__":
    # file to knowledge embeddings
    file_path = r"D:\Voice based conversational bot with pdf knowledge\knowledge documents\machine_learning_tutorial.pdf"
    main_pipeline(file_path)
