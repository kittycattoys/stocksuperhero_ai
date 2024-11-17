import streamlit as st
import pandas as pd
from supabase import create_client, Client
from functions.vector_search import get_supabase_dataframe
import requests

API_URL = st.secrets["other"]["api"]
headers = {"Authorization": f"Bearer {st.secrets['huggingface']['token']}"}

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Supabase connection details
supabase: Client = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])
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

if 'df_dim' not in st.session_state:
    response_dim = supabase.table('dim_det').select('sym, pst, cn, ind, sec, ps, pst, v_ps_string, v_rsi_string, v_ps, v_rsi').eq('sym', selected_stock_symbol).execute()
    st.session_state['df_dim'] = pd.DataFrame(response_dim.data)
    
df_dim = st.session_state['df_dim']
st.dataframe(df_dim)

# Display the selected value
st.write(f"Selected value: {ps_weight}")

options = [df_dim["sec"][0], df_dim["ind"][0], "All", "Selected Filters"]
option = st.selectbox("Search type:", options)
selected_index = options.index(option)

where_clause = None

if selected_index == 0:
    where_clause_1 = 'sec'
    where_clause_2 = option
if selected_index == 1:
    where_clause_1 = 'ind'
    where_clause_2 = option
if selected_index == 1:
    where_clause = f"WHERE sec LIKE '%'"

print(where_clause_1)
print(where_clause_2)

if st.button(f"Run Vector Search {selected_stock_symbol}"):
    input_v_ps = df_dim['v_ps'][0] # Example embedding vector for v_ps
    input_v_rsi = df_dim['v_rsi'][0]   # Example embedding vector for v_rsi
    df_vector_search = get_supabase_dataframe(input_v_ps, input_v_rsi, ps_weight, rsi_weight, where_clause_1, where_clause_2, match_count=400)
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
        st.markdown(generated_text)

    st.session_state.messages.append({"role": "assistant", "content": generated_text})