import os
from pathlib import Path
import tempfile
import pandas as pd

import streamlit as st

from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

from astrapy import DataAPIClient
import requests
from typing import Optional
import warnings
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None


print("Started")
st.set_page_config(page_title='RAG Chatbot Using Astra DB and Langflow', page_icon='🚀')


###############
### Globals ###
###############

global lang_dict
global language
global session
global embedding
global vectorstore
global astra_database

#################
### Functions ###
#################

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(file.name)
            astra_docs = astra_database.get_collection("document_list")
            astra_doc = {"name": file.name}
            if astra_docs.find_one(astra_doc):
                print("doc already loaded - no action taken")
                st.info(f"Document has already been loaded to Astra.")
                pass
            else:
                astra_docs.insert_one(astra_doc)
                
                print(f"""Processing: {file}""")
                temp_filepath = os.path.join(temp_dir.name, file.name)
                with open(temp_filepath, 'wb') as f:
                    f.write(file.getvalue())

                # Process TXT
                if uploaded_file.name.endswith('txt'):
                    file = [uploaded_file.read().decode()]

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 1500,
                        chunk_overlap  = 100
                    )

                    texts = text_splitter.create_documents(file, [{'source': uploaded_file.name}])
                    vectorstore.add_documents(texts)
                    st.info(f"Document has successfully loaded to Astra. Chunks: {len(texts)}")
                
                # Process PDF
                if uploaded_file.name.endswith('pdf'):
                    docs = []
                    loader = PyPDFLoader(temp_filepath)
                    docs.extend(loader.load())

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 1500,
                        chunk_overlap  = 100
                    )

                    pages = text_splitter.split_documents(docs)
                    vectorstore.add_documents(pages)  
                    st.info(f"Document has successfully loaded to Astra. Chunks: {len(pages)}")

                # Process CSV
                if uploaded_file.name.endswith('csv'):
                    docs = []
                    loader = CSVLoader(temp_filepath)
                    docs.extend(loader.load())

                    vectorstore.add_documents(docs)
                    st.info(f"Document has successfully loaded to Astra. Chunks: {len(docs)}")


# Load data from URLs
def vectorize_url(urls):
    print("calling vectorize_url")
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )

    for url in urls:
        print(url)
        astra_docs = astra_database.get_collection("document_list")
        astra_doc = {"name": url}
        if astra_docs.find_one(astra_doc):
            print("doc already loaded - no action taken")
            st.info(f"Document has already been loaded to Astra.")
            pass
        else:
            astra_docs.insert_one(astra_doc)
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()    
                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)  
                st.info(f"Document has successfully loaded to Astra. Chunks: {len(pages)}")
            except Exception as e:
                st.info(f"An error occurred:", e)
            print("inserted doc")

# Get the Retriever
def load_retriever():
    print(f"""load_retriever""")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

def load_localization(locale):
    print("load_localization")
    # Load in the text bundle and filter by language locale
    df = pd.read_csv("./customizations/localization.csv")
    df = df.query(f"locale == '{locale}'")
    # Create and return a dictionary of key/values.
    lang_dict = {df.key.to_list()[i]:df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return lang_dict

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  application_token: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    print("run flow called")
    api_url = f"https://api.langflow.astra.datastax.com/lf/{st.secrets['LANGFLOW_ID']}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }

    tweaks = {
        "TextInput-rGiXS": {},
        "OpenAIModel-IrFqp": {},
        "TextOutput-1MwwT": {}
    }
    payload["tweaks"] = tweaks
    headers = None
    if application_token:
        headers = {"Authorization": "Bearer " + application_token, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    print("got response from api")
    return response.json()

#############
### Login ###
#############
lang_dict = load_localization("en_US")

#################
### Resources ###
#################

def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    return OpenAIEmbeddings()

def load_vectorstore():
    print(f"load_vectorstore")
    # Get the load_vectorstore store from Astra DB
    return AstraDB(
        embedding=embedding,
        collection_name=f"vector_context_datastax",
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=st.secrets["ASTRA_ENDPOINT"],
    )

def load_astra():
    print("load_astra")
    client = DataAPIClient(st.secrets["ASTRA_TOKEN"])
    database = client.get_database(st.secrets["ASTRA_ENDPOINT"])
    print(f"* Database: {database.info().name}\n")
    return database


#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

###############
### Sidebar ###
###############


with st.sidebar:
    # Show the DataStax logo
    st.image('./customizations/logo/datastax.svg', use_column_width="always")

    st.divider()
    
    # Initialize
    embedding = load_embedding()
    vectorstore = load_vectorstore()
    astra_database = load_astra()

    # Include the upload form for new data to be Vectorized
    uploaded_files = st.file_uploader(lang_dict['load_context'], type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
    upload = st.button(lang_dict['load_context_button'])
    if upload and uploaded_files:
        vectorize_text(uploaded_files)

    # Include the upload form for URLs be Vectorized
    urls = st.text_area(lang_dict['load_from_urls'], help=lang_dict['load_from_urls_help'])
    urls = urls.split(',')
    upload = st.button(lang_dict['load_from_urls_button'])
    if upload and urls:
        vectorize_url(urls)

    st.divider()

    # Drop the vector data and start from scratch
    submitted = st.button(lang_dict['delete_context_button'])
    if submitted:
        with st.spinner(lang_dict['deleting_context']):
            vectorstore.clear()
            astra_database.get_collection("document_list").delete_many({})
            st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]
    st.caption(lang_dict['delete_context'])

    st.divider()

    st.caption("Documents currently loaded in Astra:")
    doc_list = []
    for astra_doc in astra_database.get_collection("document_list").find({}):
        doc_list.append(astra_doc["name"])
    if doc_list:
        st.dataframe(doc_list, column_config={"value": "Document Name"})
    else:
        st.text("None")
    print(doc_list)


############
### Main ###
############

# Show a custom welcome text or the default text
st.markdown(Path('./customizations/welcome/default.md').read_text())

# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
question = st.chat_input(lang_dict['assistant_question'])


if question:
    print(f"Got question: {question}")
           
    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print(f"Draw prompt")
    with st.chat_message('human'):
        st.markdown(question)

    # Get retriever
    retriever = load_retriever()

    # Retrieve documents from Astra
    content = ''
    relevant_documents = retriever.get_relevant_documents(query=question, k=5)
    # print("relevant documents:", relevant_documents)

    # Get the results from Langflow
    print(f"Chat message")
    with st.chat_message('assistant'):
        content = ''

        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        prompt = f"""
            Answer the user's question to the best of your ability using the provided context. If there is no context, or the provided context cannot help you answer the question, please state that you did not use any context and are making a guess.
            
            Context:
            {relevant_documents}
            
            Question:
            {question}
        """

        # Call Langflow to get a response
        response = run_flow(
            message=prompt,
            endpoint=st.secrets["FLOW_ID"],
            application_token=st.secrets["ASTRA_TOKEN"]
        )
        text_output = response["outputs"][0]["outputs"][0]["results"]["text"]["data"]["text"]

        print(text_output)
        
        content += text_output

        content += "  "

        content += f"""
            *{lang_dict['sources_used']}*  
        """
        sources = []
        for doc in relevant_documents:
            # print (f"""DOC: {doc}""")
            source = doc.metadata['source']
            page_content = doc.page_content
            if source not in sources:
                content += f"""
                    [{os.path.basename(os.path.normpath(source))}]  
                """
                sources.append(source)

        # Write the final answer without the cursor
        response_placeholder.markdown(content)

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))
