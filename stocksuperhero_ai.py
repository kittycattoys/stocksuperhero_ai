import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import time
from functions.vector_search import get_supabase_dataframe

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

with st.expander("Expander with scrolling content", expanded=True):
   with st.container(height=300):
       
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = f"Echo: {prompt}"
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})