import streamlit as st
import pandas as pd
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
    response_dim = supabase.table('dim_det').select('sym, pst, cn, ind, sec, ps, pst, v_ps, v_rsi, v_ps_string, v_rsi_string').eq('sym', selected_stock_symbol).execute()
    st.session_state['df_dim'] = pd.DataFrame(response_dim.data)

df_dim = st.session_state['df_dim']

input_v_ps = df_dim['v_ps'][0] # Example embedding vector for v_ps
input_v_rsi = df_dim['v_rsi'][0]   # Example embedding vector for v_rsi
df_vector_search = get_supabase_dataframe(input_v_ps, input_v_rsi, match_count=10)
st.write("Vector Search Results")
st.dataframe(df_vector_search)
