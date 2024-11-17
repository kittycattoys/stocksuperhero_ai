import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import time
from functions.vector_search import get_supabase_dataframe
from transformers import AutoTokenizer, AutoModelForCausalLM
import sentencepiece
import requests
from huggingface_hub import InferenceClient

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct"
headers = {"Authorization": f"Bearer {st.secrets['huggingface']['token']}"}

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Supabase connection details
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase: Client = create_client(url, key)
selected_stock_symbol = 'BAX'
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
    columns_to_keep = ["sym", "v_ps_string", "cos_sim", "cos_sim_v_ps", "v_rsi_string", "cos_sim_v_rsi"]
    st.write("Vector Search Results")

    st.dataframe(df_vector_search[columns_to_keep])



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("Expander with scrolling content", expanded=False):
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
        payload = {
            "inputs": f"{prompt}",
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        responsejson = response.json()
        generated_text = responsejson[0]["generated_text"]
        print("generated_text")
        print(generated_text)
        st.markdown(generated_text)

    st.session_state.messages.append({"role": "assistant", "content": generated_text})




'''

client = InferenceClient(api_key=st.secrets["huggingface"]["token"])

for message in client.chat_completion(
	model="meta-llama/Llama-3.2-1B-Instruct",
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")

'''



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
    