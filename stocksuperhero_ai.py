import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import time
from functions.vector_search import get_supabase_dataframe
from transformers import AutoTokenizer, AutoModelForCausalLM
import sentencepiece

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Supabase connection details
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase: Client = create_client(url, key)
selected_stock_symbol = 'SBUX'
st.title(selected_stock_symbol)
    
# Create a slider
ps_weight = st.slider(
    label="PS Weight",
    min_value=0.0,  # Minimum value
    max_value=1.0,  # Maximum value
    step=0.05,  # Increment step
    value=0.05,  # Initial value
)

rsi_weight = 1 - ps_weight

# Display the selected value
st.write(f"Selected value: {ps_weight}")

if st.button(f"Run Vector Search {selected_stock_symbol}"):
    response_dim = supabase.table('dim_det').select('sym, pst, cn, ind, sec, ps, pst, v_ps_string, v_rsi_string, v_ps, v_rsi').eq('sym', selected_stock_symbol).execute()
    st.session_state['df_dim'] = pd.DataFrame(response_dim.data)
    df_dim = st.session_state['df_dim']
    st.dataframe(df_dim)

    input_v_ps = df_dim['v_ps'][0] # Example embedding vector for v_ps
    input_v_rsi = df_dim['v_rsi'][0]   # Example embedding vector for v_rsi
    df_vector_search = get_supabase_dataframe(input_v_ps, input_v_rsi, ps_weight, rsi_weight, match_count=400)
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

    with st.chat_message("assistant"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "assistant", "content": prompt})

'''
tokens_input = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(**tokens_input, 
                            min_length=200, 
                            max_length=300, 
                            temperature=0.8,  
                            #top_p=0.9 
                           )
stream = tokenizer.decode(output_ids[0], skip_special_tokens=False)
response = st.markdown(stream)
'''
    