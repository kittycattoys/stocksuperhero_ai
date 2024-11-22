import streamlit as st
import pandas as pd
from supabase import create_client, Client
from functions.vector_search import get_supabase_dataframe
import requests, os
from huggingface_hub import InferenceClient, login
from functions.bar import plot_bar_chart

#API_URL = st.secrets["other"]["api"]
#headers = {"Authorization": f"Bearer {st.secrets['huggingface']['token']}"}

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide")

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=st.secrets['huggingface']['token'])

# Define the function definition
function_definitions = """[
    {   "type": "function",
        "name": "create_ps_price_sales_bar_chart", 
        "description": "Creates a price-to-sales ps (price to sales) bar chart comparing multiple sym (stock symbols) with sym on the x-axis and ps (price to sales ratio) on y-axis highlight highlight_symbol.", 
        "parameters": {
            "type": "object",
            "required": [
                "highlight_symbol"
            ],
            "properties": {
                "highlight_symbol": {
                    "type": "string",
                    "description": "The stock symbol required for chart."
                }
            }
        }
    },
    {   "type": "function",
        "name": "calculate_average_metric", 
        "description": "Calculates the average of a ps column in a dataframe for the required sym symbol", 
        "parameters": {
            "required": ["sym"]
        }
    },
       {   "type": "function",
        "name": "ask_pandas_dataframe_table_question", 
        "description": "Perform simple calculations or summarizations from a data table via prompt input and provide and answer.", 
        "parameters": {
            "required": ["question"]
        }
    }
]"""

# Define the system prompt using your preferred format
system_prompt = """You are an expert in composing functions. Use the provided JSON functions to answer the question. 
Return function calls in this format: [func_name(param1=value1, param2=value2)]. Do not make up any data. Use only what is provided.
If no function applies return Error No Function Found. If required parameters are missing return Missing Params.
Do not include any other text.\n\n{functions}\n""".format(functions=function_definitions)

query = "calculate average"

# Create the input prompt
input_prompt = f"""
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

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


if selected_index == 0:
    where_clause_1 = 'sec'
    where_clause_2 = option
    operand = ' = '
if selected_index == 1:
    where_clause_1 = 'ind'
    where_clause_2 = option
    operand = ' = '
if selected_index == 2:
    where_clause_1 = 'ind'
    where_clause_2 = "blank"
    operand = ' != '

print(where_clause_1)
print(where_clause_2)

if st.button(f"Run Vector Search {selected_stock_symbol}"):
    input_v_ps = df_dim['v_ps'][0] # Example embedding vector for v_ps
    input_v_rsi = df_dim['v_rsi'][0]   # Example embedding vector for v_rsi
    df_vector_search = get_supabase_dataframe(input_v_ps, input_v_rsi, ps_weight, rsi_weight, where_clause_1, where_clause_2, operand, match_count=400)
    columns_to_keep = ["sym", "ps", "dy", "rsi", "v_ps_string", "cos_sim", "cos_sim_v_ps", "v_rsi_string", "cos_sim_v_rsi"]
    st.write("Vector Search Results")

    st.dataframe(df_vector_search[columns_to_keep])
    st.session_state['ai_dataframe'] = df_vector_search[columns_to_keep]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Extract the assistant data
def ask_pandas_dataframe_table_question(question):
    # Split the response by the "assistant" keyword and get the part after it
    # Represent the DataFrame as a string
    df_string = st.session_state['ai_dataframe'].to_string(index=False)

    # Input prompt to instruct the model
    input_prompt = f"""
    You are an AI assistant that analyzes data tables. 
    Here is the table:
    {df_string}

    Answer the following question:
    {question}

    Use only the data provided and Return the FINAL answer along with a python formula that can be used. Do not return any other text."""
    return input_prompt    

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
        big_boy_jeffy = None
        jeffy = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{prompt}"},
                {"role": "assistant", "content": ""},
                ]
        chater = client.chat_completion(jeffy, max_tokens=170, stream=False)
        chater_extract = chater.choices[0].message.content

        if chater_extract == "Error No Function Found" or chater_extract == "Missing Params":
            st.markdown("Function Not Found")
            st.markdown(chater_extract)
            big_boy_jeffy = chater_extract
        else:
            st.markdown("Function Found")
            st.markdown(chater_extract)
            function_call = chater_extract.split('(')[1].split(')')[0]
            print("!!!!!!!!!!!!!!!!! AUDIT FUNCTION CALL !!!!!!!!!!!!!!!!!")
            print(function_call.split("=")[1])     
            symbol = function_call.split("=")[1]  
            cleaned_extract = symbol.replace('"', '')

            if "ask_pandas_dataframe_table_question" in chater_extract:
                get_full_formatted_prompt = ask_pandas_dataframe_table_question(cleaned_extract)
                print("!!!! get_full_formatted_prompt !!!!")
                print(get_full_formatted_prompt)
                jeffy_second = [
                {"role": "system", "content": "You are an AI assistant that analyzes data tables and answers questions"},
                {"role": "user", "content": f"{get_full_formatted_prompt}"},
                {"role": "assistant", "content": ""},
                ]

                chaterzzz = client.chat_completion(jeffy_second, max_tokens=700, stream=False)
                chater_extractzz = chaterzzz.choices[0].message.content
                
                # Output the response
                print("\n=== LLM Response 2===")
                if chaterzzz:
                    print(chater_extractzz)
                    st.markdown(chater_extractzz)
                    big_boy_jeffy = chater_extractzz
                else:
                    print("No chaterzzz response generated.")

            elif "create_ps_price_sales_bar_chart" in chater_extract:
                st.markdown("Trying to Create Bar Chart")
                fig_bar = plot_bar_chart(st.session_state['ai_dataframe'], cleaned_extract)

                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                    big_boy_jeffy = st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write("ERROR No data available to display in the bar chart.")

    st.session_state.messages.append({"role": "assistant", "content": big_boy_jeffy})