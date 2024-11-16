import pandas as pd
import streamlit as st
from supabase import create_client, Client

# Initialize Supabase client
def init_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    supabase: Client = create_client(url, key)
    return supabase

# Function to call the match_vectors RPC
def get_supabase_dataframe(input_v_ps, input_v_rsi, match_count=100):
    supabase: Client = init_supabase()

    # RPC call to the match_vectors function
    response = supabase.rpc("match_vectors", {
        "query_v_ps": input_v_ps,
        "query_v_rsi": input_v_rsi,
        "match_count": match_count,
    }).execute()
    
    if not response.data:
        print("Error: No data found")
    else:
        # Process the response data
        print(response.data)
    
    # Convert response data to a DataFrame
    data = response.data
    df = pd.DataFrame(data)

    return df
