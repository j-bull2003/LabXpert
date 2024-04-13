import openai
import time
import streamlit as st
from dotenv import load_dotenv
import os
import shelve
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from datetime import datetime
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_gsheets import GSheetsConnection

url = "https://docs.google.com/spreadsheets/d/1Ao-pNzVZXMPw13FAF8ZQL_V9TazZCuStAVIut6OLUQ0/edit#gid=363208242"

conn = st.experimental_connection("gsheets", type=GSheetsConnection)

# data = conn.read(spreadsheet=url, usecols=[0, 8])
# st.dataframe(data)



load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Define URL and connect to Google Sheets
url = "https://docs.google.com/spreadsheets/d/1Ao-pNzVZXMPw13FAF8ZQL_V9TazZCuStAVIut6OLUQ0/edit#gid=363208242"
conn = st.experimental_connection("gsheets", type=GSheetsConnection)
data = conn.read(spreadsheet=url)

# Create DataFrame with specific columns
df = pd.DataFrame(data, columns=['PMID', 'Title', 'Author(s) Full Name', 'Author(s) Affiliation', 'Journal Title', 'Place of Publication', 'Date of Publication', 'Publication Type', 'Abstract'])
df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Function to search for similar texts based on TF-IDF
def search_similar_texts(query, data_frame, k=3):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['combined_text'])
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-k:][::-1]
    return data_frame.iloc[top_indices], cosine_similarities[top_indices]


def init_data_analysis():
    if "messages_data_analysis" not in st.session_state:
        st.session_state.messages_data_analysis = []

    if "run_data_analysis" not in st.session_state:
        st.session_state.run_data_analysis = None

    if "file_ids_data_analysis" not in st.session_state:
        st.session_state.file_ids_data_analysis = []
    
    if "thread_id_data_analysis" not in st.session_state:
        st.session_state.thread_id_data_analysis = None
        
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def init_research_assistant():
    if "messages_research_assistant" not in st.session_state:
        st.session_state.messages_research_assistant = load_chat_history()

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"


def run_research_assistant_chatbot():
    st.title("Research Assistant 🔬")
    st.caption('Analyse your experimental data')
    st.markdown('Your personal Data Anaylist tool ')
    st.divider()

    CHROMA_PATH = "chroma"
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"


    
    def load_chat_history():
        with shelve.open("chat_history") as db:
            return db.get("messages", [])

    def save_chat_history(messages):
        with shelve.open("chat_history") as db:
            db["messages"] = messages

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    with st.sidebar:
        if st.button("Delete Chat History"):
            st.session_state.messages = []
            save_chat_history([])

    class CustomOpenAIEmbeddings(OpenAIEmbeddings):
        def __init__(self, openai_api_key, *args, **kwargs):
            super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
            
        def _embed_documents(self, texts):
            return super().embed_documents(texts)

        def __call__(self, input):
            return self._embed_documents(input)


    def formulate_response(prompt):
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        chat_history = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
        prompt_with_history = f"Previous conversation:\n{chat_history}\n\nYour question: {prompt}"
        df = pd.DataFrame(data, columns=['PMID', 'Title', 'Author(s) Full Name', 'Author(s) Affiliation', 'Journal Title', 'Place of Publication', 'Date of Publication', 'Publication Type', 'Abstract'])
        df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        # Search for similar texts in the database
        results_df, similarity_scores = search_similar_texts(prompt_with_history, df, k=3)
        
        with st.spinner("Thinking..."):
            # Decide whether to use DB, GPT, or DB+GPT based on the similarity scores and the content availability
            if results_df.empty or all(score < 0 for score in similarity_scores):
                # If no similar texts are found or all texts are below threshold, use GPT model
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                response_text = model.predict(prompt_with_history)
                response = f"gpt: {response_text}"
            else:
                # If similar texts are found, prepare to integrate DB and GPT
                combined_texts = []
                sources = []
                for doc, score in zip(results_df.itertuples(), similarity_scores):
                    if score >= 0.5:
                        doc_content = doc.page_content
                        authors = doc.metadata.get('authors', 'Unknown').split(',')[0] + " et al."
                        year = doc.metadata.get('year', 'Unknown')
                        citation = f"({authors}, {year})"
                        combined_texts.append(f"{doc_content} {citation}")
                        source_info = (
                            f"\n🦠 {authors}\n"
                            f"({year}),\n"
                            f"\"{doc.title}\",\n"
                            f"PMID: {doc.metadata.get('pub_id', 'N/A')},\n"
                            f"Available at: {doc.metadata.get('url', 'N/A')},\n"
                            f"Accessed on: {datetime.today().strftime('%Y-%m-%d')}\n"
                        )
                        sources.append(source_info)

                # Combine database texts and re-query the model for an integrated response
                if combined_texts:
                    combined_input = " ".join(combined_texts)
                    model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                    integrated_prompt = f"{combined_input}\n\nQuestion: {prompt}\nPlease integrate the above information to answer the question."
                    integrated_response = model.predict(integrated_prompt)
                    sources_formatted = "\n".join(sources)
                    response = f"db+gpt: {integrated_response}\nSources:\n{sources_formatted}"
                else:
                    model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                    response_text = model.predict(prompt_with_history)
                    response = f"gpt: {response_text}"

            # Add response to session state and display
            if 'sources' in locals() and sources:
                st.session_state.messages.append({"role": "assistant", "content": response, "citations": sources_formatted})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            display_messages()




    def typewriter(container, text: str, speed: int):
        """Display text with a typewriter effect, preserving newline characters."""
        lines = text.split('\n')
        curr_full_text = ''
        
        for line in lines:
            tokens = line.split()
            for index in range(len(tokens) + 1):
                curr_line = " ".join(tokens[:index])
                curr_full_text_with_line = f"{curr_full_text}\n{curr_line}" if curr_full_text else curr_line
                container.markdown(curr_full_text_with_line, unsafe_allow_html=True)
                time.sleep(1 / speed)
            curr_full_text += f"{line}\n"

    def display_messages():
        """Function to display all messages in the chat history and show citations for the last response."""
        total_messages = len(st.session_state.messages)
        for index, message in enumerate(st.session_state.messages):
            avatar = "🧬" if message["role"] == "user" else "🤖"
            text = f"{avatar} {message['content']}"
            
            if message["role"] == "user":
                st.markdown(text, unsafe_allow_html=True)
            else:
                container = st.empty()
                if index == total_messages - 1:
                    typewriter(container, text, speed=50)
                else:
                    container.markdown(text, unsafe_allow_html=True)
            if "citations" in message and message["citations"]:
                citations_button_label = "Show Citations"
                with st.expander(citations_button_label):
                    st.markdown(message["citations"], unsafe_allow_html=True)


    user_prompt = st.chat_input("How can I help?")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        formulate_response(user_prompt)

    save_chat_history(st.session_state.messages)





import io
from openai import OpenAI

import streamlit as st
import pandas as pd
import os
import time
import tempfile
import requests
import csv
import json
from PIL import Image
def run_data_analysis_chatbot():
    st.title("Lab Assistant 📊")
    class CustomOpenAIEmbeddings(OpenAIEmbeddings):
        def __init__(self, openai_api_key, *args, **kwargs):
            super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
            
        def _embed_documents(self, texts):
            return super().embed_documents(texts)

        def __call__(self, input):
            return self._embed_documents(input)
    CHROMA_PATH = "chroma"
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    embedding_function = CustomOpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_database = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    def init():
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "run" not in st.session_state:
            st.session_state.run = None

        if "file_ids" not in st.session_state:
            st.session_state.file_ids = []
        
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = None

    def set_apikey():
        # st.sidebar.header('Your personal Scientific Data Analyst')
        # st.sidebar.markdown('This AI Lab Assistant is design to analyse scientific data.')
        # st.sidebar.header('Configure')
        # api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
        api_key = st.secrets["OPENAI_API_KEY"]
        return api_key
        

    def config(client):
        my_assistants = client.beta.assistants.list(
            order="desc",
            limit="20",
        )
        assistants = my_assistants.data
        for assistant in assistants:
            if assistant.name == "Lab.ai":
                return assistant.id
        print("Lab.ai assistant not found.")
        return None


    def upload_file(client, assistant_id, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.close()
            with open(tmp_file.name, "rb") as f:
                response = client.files.create(
                file=f,
                purpose = 'assistants'
                )
                print(response)
                os.remove(tmp_file.name)
        assistant_file = client.beta.assistants.files.create(
            assistant_id=assistant_id,
            file_id=response.id,
        )
        return assistant_file.id
            
    def assistant_handler(client, assistant_id):
        def delete_file(file_id):
            client.beta.assistants.files.delete(
                        assistant_id=assistant_id,
                        file_id=file_id,
                    ) 

        
        assistant = client.beta.assistants.retrieve(assistant_id)
        with st.sidebar:
            # assistant_name = st.text_input("Name", value = assistant.name)
            assistant_instructions = "You are a data analyst"
            model_option = 'gpt-3.5-turbo-0125'
            uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv"])
        
            if st.button("Upload File"):
                assistant = client.beta.assistants.update(
                    assistant_id,
                    instructions = assistant_instructions,
                    name = 'Lab.ai',
                    model = 'gpt-3.5-turbo-0125',

                )   
                if uploaded_file is not None:
                    new_file_id = upload_file(client, assistant_id, uploaded_file)
                    print(new_file_id)
                    st.session_state.file_ids.append(new_file_id)
                st.success("Assistant updated successfully")
        return assistant, model_option, assistant_instructions

    def create_assistant(client):
        assistants_dict = {"Create Assistant": "create-assistant"}
        assistant_name = st.text_input("Name")
        assistant_instructions = st.text_area("Instructions")
        model_option = st.radio("Model", ('gpt-3.5-turbo-0125'))
        def create_new_assistant():
            new_assistant = client.beta.assistants.create(
                name=assistant_name,
                instructions=assistant_instructions,
                model=model_option,
                tools =[
                    {
                        "type": "code_interpreter",
                    }
                ]
            )

        my_assistants = client.beta.assistants.list(
            order="desc",
            limit="20",
        ).data
        assistants_dict = {"Create Assistant": "create-assistant"}
        for assistant in my_assistants:
            assistants_dict[assistant.name] = assistant.id
        if assistant_name not in assistants_dict:
            new_assistant = st.button("Create Assistant", on_click=create_new_assistant)
            if new_assistant:
                my_assistants = client.beta.assistants.list(
                    order="desc",
                    limit="20",
                ).data
                assistants_dict = {"Create Assistant": "create-assistant"}
                for assistant in my_assistants:
                    assistants_dict[assistant.name] = assistant.id
                st.success("Assistant created successfully")
                st.stop()
                print(assistants_dict)
                print("\n NEW: ", assistants_dict[assistant_name])
                return assistants_dict[assistant_name]
        else:
            st.warning("Assistant name does exist in assistants_dict. Please choose another name.")
      
    def chat_prompt(client, assistant_option):
        if prompt := st.chat_input("Enter your message here"):
            # run_research_assistant_chatbot.formulate_response(prompt)
            # Append the user's message to the chat history for later display
            user_message = client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt,
            )

            # Ensure the messages list is updated correctly
            if st.session_state.messages is None:
                st.session_state.messages = [user_message]
            else:
                st.session_state.messages.append(user_message)

            # Updating the assistant's configuration
            st.session_state.current_assistant = client.beta.assistants.update(
                st.session_state.current_assistant.id,
                instructions=st.session_state.assistant_instructions,
                name=st.session_state.current_assistant.name,
                tools=st.session_state.current_assistant.tools,
                model=st.session_state.model_option,
                file_ids=st.session_state.file_ids,
            )

            # Processing the prompt
            st.session_state.run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_option,
                tools=[{"type": "code_interpreter"}],
            )

            pending = False
            while st.session_state.run.status != "completed":
                with st.spinner("Thinking..."):
                    if not pending:
                        # Show a temporary message while the assistant is processing
                        # with st.chat_message("assistant"):
                        #     st.markdown("Lab.ai is thinking...")
                            pending = True
                    time.sleep(3)
                    st.session_state.run = client.beta.threads.runs.retrieve(
                        thread_id=st.session_state.thread_id,
                        run_id=st.session_state.run.id,
                    )

            if st.session_state.run.status == "completed":
                st.empty()
                chat_display(client)
            

    def typewriter(container, text: str, speed: int):
        """Display text with a typewriter effect, preserving newline characters."""
        lines = text.split('\n')
        curr_full_text = ''
        
        for line in lines:
            tokens = line.split()
            for index in range(len(tokens) + 1):
                curr_line = " ".join(tokens[:index])
                curr_full_text_with_line = f"{curr_full_text}\n{curr_line}" if curr_full_text else curr_line
                container.markdown(curr_full_text_with_line, unsafe_allow_html=True)
                time.sleep(1 / speed)
            curr_full_text += f"{line}\n"

    def chat_display(client):
        st.session_state.messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        ).data

        for message in reversed(st.session_state.messages):
            if message.role in ["user", "assistant"]:
                # Define the avatar based on the message role
                avatar = "🧬" if message.role == "user" else "🤖"
                for content in message.content:
                    if content.type == "text":
                        # Prepend the avatar to the text content
                        text_with_avatar = f"{avatar} {content.text.value}"
                        # Prepend the avatar to the text content

                        container = st.empty()
                        # Corrected: Pass the 'text_with_avatar' variable directly
                        st.markdown(text_with_avatar)

                        container = st.empty()
                        # typewriter(container, text_with_avatar, speed=50)  # Adjust the speed as needed
                    elif content.type == "image_file":
                        # Image files are handled normally, as before
                        image_file = content.image_file.file_id
                        image_data = client.files.content(image_file)
                        image_data = image_data.read()
                        # Save image to temp file
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file.write(image_data)
                        temp_file.close()
                        # Display image
                        image = Image.open(temp_file.name)
                        st.image(image)
                    else:
                        # For other content types, display them directly
                        st.markdown(content)

    def main():
        st.caption('Analyse your experimental data')
        st.markdown('Your personal Data Anaylist tool ')
        st.divider()
        api_key = set_apikey()
        if api_key:
            client = OpenAI(api_key=api_key)
            assistant_option = config(client)
            print ("Use existing assistant")
            st.session_state.current_assistant, st.session_state.model_option, st.session_state.assistant_instructions = assistant_handler(client, assistant_option)
            if st.session_state.thread_id is None:
                st.session_state.thread_id = client.beta.threads.create().id
                print(st.session_state.thread_id)
            chat_prompt(client, assistant_option)
            
        else:
            st.warning("Please enter your OpenAI API key")
                


    if __name__ == '__main__':
        init()
        main() 
        print(st.session_state.file_ids)

def main():
    st.sidebar.title("Science.ai")
    chatbot_mode = st.sidebar.radio("Choose an AI to assistant you:", ('Data Analysis 📊', 'Research Assistant 🔬'))
    
    if chatbot_mode == 'Data Analysis 📊':
        init_data_analysis()
        run_data_analysis_chatbot()
    elif chatbot_mode == 'Research Assistant 🔬':
        init_research_assistant()
        run_research_assistant_chatbot()

if __name__ == '__main__':
    main()