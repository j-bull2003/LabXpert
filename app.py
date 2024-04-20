import zipfile
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

def estimate_complexity(question):
    # List of simpler words
    simple_words = [ 'name', 'show', 'list', 'tell', 'define', 'what', 'who', 'where']

    # List of difficult words
    difficult_words = ['design', 'experiment', 'compare', 'contrast', 'details', 'theory', 'research', 'evaluate', 'discuss', 'analyze']
    
    # Make the question lowercase to match words correctly without case issues
    question_lower = question.lower()
    
    # Count the occurrences of each simple word
    simple_score = sum(question_lower.count(word) for word in simple_words)

    # Count the occurrences of each difficult word
    difficult_score = sum(question_lower.count(word) for word in difficult_words)
    
    # Total complexity score, considering both simple and difficult words
    complexity_score = difficult_score * 2 - simple_score  # Weight difficult words more

    # Decide the complexity level based on the complexity score
    if complexity_score > 3:
        return 10  # High complexity
    elif complexity_score > 0:
        return 5   # Moderate complexity
    else:
        return 1   # Low complexity
  
    
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


import subprocess
load_dotenv()
def run_research_assistant_chatbot():
    st.title("Research Xpert 📄")
    st.caption('Ask questions about REAL scientific articles')
    st.markdown('Enjoy fully cited responses, Harvard style.')
    st.divider()
    

    # List of files and their GCS paths
#     CHROMA = [
#     "https://storage.googleapis.com/chromaproto/chroma.sqlite3",
#     "https://storage.googleapis.com/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/data_level0.bin",
#     "https://storage.googleapis.com/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/header.bin",
#     "https://storage.googleapis.com/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/index_metadata.pickle",
#     "https://storage.googleapis.com/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/length.bin",
#     "https://storage.googleapis.com/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/link_lists.bin",
# ]

#     def download_files(urls, target_folder):
#         # Ensure the base target folder exists
#         os.makedirs(target_folder, exist_ok=True)
        
#         # Iterate over each URL
#         for url in urls:
#             # Extract the full path segments from the URL
#             path_segments = url.split('/')[3:]  # This skips the 'https://' part and domain name
#             subdirectory_path = os.path.join(target_folder, *path_segments[:-1])
            
#             # Ensure the target subdirectory exists
#             os.makedirs(subdirectory_path, exist_ok=True)
            
#             # Full path for the file to be saved
#             file_path = os.path.join(subdirectory_path, path_segments[-1])
            
#             # Check if the file already exists
#             if os.path.exists(file_path):
#                 print(f'Using cached file {file_path}')
#             else:
#                 # Download and save the file if not present
#                 response = requests.get(url)
#                 if response.status_code == 200:
#                     with open(file_path, 'wb') as file:
#                         file.write(response.content)
#                     print(f'Downloaded {path_segments[-1]} to {file_path}')
#                 else:
#                     print(f'Failed to download {path_segments[-1]} with status code {response.status_code}')

#     # Directory where the files will be saved
#     target_directory = 'chromaproto'

#     # Call the function to download the files
#     download_files(CHROMA, target_directory)
#     CHROMA_PATH = 'chromaproto/chromaproto'




    
    # CHROMA_PATH = "https://storage.googleapis.com/storage/chromaproto/chroma.sqlite3"
    # "https://storage.googleapis.com/storage/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/data_level0.bin"
    # "https://storage.googleapis.com/storage/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/header.bin"
    # "https://storage.googleapis.com/storage/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/index_metadata.pickle"
    # "https://storage.googleapis.com/storage/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/length.bin"
    # "https://storage.googleapis.com/storage/chromaproto/6f79f50d-77fa-4352-914d-e3e97df18086/link_lists.bin"

    # https://storage.googleapis.com/storage/chromaproto/chroma.sqlite3
    # import os
    # import requests
    # import zipfile
    # from io import BytesIO

    # ZIP_URL = 'https://drive.usercontent.google.com/download?id=1iO8NAOULW6nfWwP_kQwVZOUegkerlDig&export=download&authuser=0&confirm=t&uuid=fe14b4e1-2c0b-4d4f-b312-085e19f4eddf&at=APZUnTU63Vu18T0v-kjSfd-jxXy5%3A1713340767172'
    # CHROMA = 'extracted_folder/'

    # def download_and_extract_zip(url, extract_to):
    #     if not os.path.exists(extract_to):
    #         response = requests.get(url, stream=True)
    #         response.raise_for_status()

    #         with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
    #             zip_ref.extractall(extract_to)

    # download_and_extract_zip(ZIP_URL, CHROMA)
    # CHROMA_PATH = 'extracted_folder/chroma'
    
    # Ensure the ZIP is extracted
    # def ensure_zip_extracted(zip_path, extract_to):
    #     if not os.path.exists(extract_to):
    #         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #             zip_ref.extractall(extract_to)

    # ensure_zip_extracted(ZIP_FOLDER, CHROMA_PATH)
    

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
    CHROMA_PATH = "chroma"
    def formulate_response(prompt):
        citations = ""
        openai_api_key = os.environ["OPENAI_API_KEY"]
        embedding_function = CustomOpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        chat_history = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
        prompt_with_history = f"Previous conversation:\n{chat_history}\n\nYour question: {prompt} Answer the question directly."
        k = estimate_complexity(prompt)
        results = db.similarity_search_with_relevance_scores(prompt_with_history, k=k)
        with st.spinner("Thinking..."):
            if len(results) == 0 or results[0][1] < 0.85:
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                # query the assistant here instead
                response_text = model.predict(prompt_with_history)      
                response = f" {response_text}"
                a = estimate_complexity(prompt)
                follow_up_results = db.similarity_search_with_relevance_scores(response_text, k=a)
                very_strong_correlation_threshold = 0.75
                high_scoring_results = [result for result in follow_up_results if result[1] >= very_strong_correlation_threshold]
                if high_scoring_results:
                    sources = []
                    combined_texts = []
                    for i, (doc, _score) in enumerate(high_scoring_results):
                        doc_content = doc.page_content
                        first_author = doc.metadata['authors'].split(',')[0] if 'authors' in doc.metadata and doc.metadata['authors'] else "Unknown"
                        citation_key = f"({first_author} et al., {doc.metadata.get('year', 'Unknown')})"
                        combined_texts.append(f"{doc_content} {citation_key}")
                        source_info = (
                            f"\n🦠 {doc.metadata.get('authors', 'Unknown')}\n"
                            f"({doc.metadata.get('year', 'Unknown')}),\n"
                            f"\"{doc.metadata['title']}\",\n"
                            f"PMID: {doc.metadata.get('pub_id', 'N/A')},\n"
                            f"Available at: {doc.metadata.get('url', 'N/A')},\n"
                            f"Accessed on: {datetime.today().strftime('%Y-%m-%d')}\n"
                        )
                        sources.append(source_info)
                    combined_input = " ".join(combined_texts)
                    # query_for_llm = f"{combined_input} Answer the question with citation to the paragraphs. For every sentence you write, cite the book name and paragraph number as (author, year). At the end of your commentary, suggest a further question that can be answered by the paragraphs provided."
                    query_for_llm = (
                        "Answer the question directly."
                        f"Question: {prompt}\n\n"
                        f"Answer the question with citations to each sentence:{combined_input}\n\n"
                        f"Question: {prompt}\n\n"
                        "Please answer the question directly with a lot of extra detail, citing relevant sections (author, year) for support."
                        f"If the question ({prompt}) has asked you to design an experiment then suggest a further question/experiment that relates, and cite it if possible, if the question didn't ask you to, then don't"
                    )
                    integrated_response = model.predict(query_for_llm)
                    sources_formatted = "\n".join(sources) 
                    citations = sources_formatted
                    
                    response = f"{integrated_response}\n"
            else:
                context_texts = []
                sources = []
                for doc, _score in results:
                    source_info = (
                        f"\n🦠 {doc.metadata.get('authors', 'Unknown')}\n"
                        f"({doc.metadata.get('year', 'Unknown')}),\n"
                        f"\"{doc.metadata['title']}\",\n"
                        f"PMID: {doc.metadata.get('pub_id', 'N/A')},\n"
                        f"Available at: {doc.metadata.get('url', 'N/A')},\n"
                        f"Accessed on: {datetime.today().strftime('%Y-%m-%d')}\n"
                    )
                    sources.append(source_info)
                context_text = "\n\n---\n\n".join(context_texts)
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                formatted_prompt = prompt_template.format(context=context_text, question=prompt_with_history)
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                response_text = model.predict(formatted_prompt)
                sources_formatted = "\n\n".join(sources)
                citations = sources_formatted    
                response = f" {response_text}\n"
                
        if citations:
            st.session_state.messages.append({"role": "assistant", "content": response, "citations": citations})
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
    st.title("Data Xpert 📊")
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
        api_key = os.environ["OPENAI_API_KEY"]
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
        st.caption('Upload your data file/s and I can produce graphs from your experiment')
        st.markdown('Your data analysis Xpert!')
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
    st.sidebar.title("LabXpert 🧬")
    st.sidebar.image("pic.png")
    st.sidebar.caption("Copyright © 2024 LabXpert, Inc. All rights reserved.")
    st.sidebar.divider()
    # Set 'Research Xpert 🔬' as the default selected option
    chatbot_mode = st.sidebar.radio("Select an AI Xpert", ('Research Xpert 📄', 'Data Xpert 📊'), index=0)
    if chatbot_mode == 'Research Xpert 📄':
        init_research_assistant()
        run_research_assistant_chatbot()
    elif chatbot_mode == 'Data Xpert 📊':
        init_data_analysis()
        run_data_analysis_chatbot()

if __name__ == '__main__':
    main()
