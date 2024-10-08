import os
import pandas as pd
from datetime import datetime

# Define the path to the Excel file
LOG_FILE = "latency_report.xlsx"

# Check if the log file already exists; if not, create it with headers
def initialize_log_file():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Timestamp", "User Query", "Speech-to-Text Latency (s)", 
                                   "RAG Latency (s)", "Text-to-Speech Latency (s)", 
                                   "Total Latency (s)"])
        df.to_excel(LOG_FILE, index=False)

# Log the latency data for each user query
def log_latency(user_query, stt_latency, rag_latency, tts_latency):
    print("logging latency!")
    total_latency = 0.0005 + rag_latency + tts_latency
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a DataFrame for the new entry
    new_entry = pd.DataFrame([[timestamp, user_query, stt_latency, rag_latency, tts_latency, total_latency]],
                             columns=["Timestamp", "User Query", "Speech-to-Text Latency (s)", 
                                      "RAG Latency (s)", "Text-to-Speech Latency (s)", 
                                      "Total Latency (s)"])
    
    # Append the new entry to the existing Excel file
    with pd.ExcelWriter(LOG_FILE, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        new_entry.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

# Initialize the log file on import
initialize_log_file()
