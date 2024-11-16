import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import time
from functions.vector_search import get_supabase_dataframe
from transformers import AutoTokenizer, AutoModelForCausalLM
import sentencepiece

print(sentencepiece.__version__)
model_id = "Jeffsimpsons/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
filename = "llama-3.2-1b-instruct-q4_k_m.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Supabase connection details
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase: Client = create_client(url, key)
selected_stock_symbol = 'SBUX'
    
# Ensure 'df_dim' is loaded
if 'df_dim' not in st.session_state:
    response_dim = supabase.table('dim_det').select(st.secrets["supabase"]["top_query"]).eq('sym', selected_stock_symbol).execute()
    st.session_state['df_dim'] = pd.DataFrame(response_dim.data)

df_dim = st.session_state['df_dim']

input_v_ps = df_dim['v_ps'][0] # Example embedding vector for v_ps
input_v_rsi = df_dim['v_rsi'][0]   # Example embedding vector for v_rsi
df_vector_search = get_supabase_dataframe(input_v_ps, input_v_rsi, match_count=10)
st.write("Vector Search Results")
st.dataframe(df_vector_search)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("Expander with scrolling content", expanded=True):
   with st.container(height=300):
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
if prompt := st.chat_input("Ask Stock Superhero AI"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Ask a question
question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt")

# Generate and decode the response
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:", response)