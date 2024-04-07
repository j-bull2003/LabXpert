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

load_dotenv()
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
      
    def chat_prompt(client, prompt):
        # if prompt := st.chat_input("Enter your message here"):
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
            


    class CustomOpenAIEmbeddings(OpenAIEmbeddings):
        def __init__(self, openai_api_key, *args, **kwargs):
            super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
            
        def _embed_documents(self, texts):
            return super().embed_documents(texts)

        def __call__(self, input):
            return self._embed_documents(input)
        
    def formulate_response(prompt):
        citations = ""
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        embedding_function = CustomOpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        chat_history = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
        prompt_with_history = f"Previous conversation:\n{chat_history}\n\nYour question: {prompt}"
        results = db.similarity_search_with_relevance_scores(prompt_with_history, k=3)
        with st.spinner("Thinking..."):
            if len(results) == 0 or results[0][1] < 0.85:
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4')
                # query the assistant here instead
                client = OpenAI(api_key=openai_api_key)
                response_text = chat_prompt(client, prompt_with_history)      
                

                response = f" {response_text}"
                follow_up_results = db.similarity_search_with_relevance_scores(response_text, k=3)
                very_strong_correlation_threshold = 0.7
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
                        f"Answer the question with citations to each sentence:\n{combined_input}\n\n"
                        f"Question: {prompt}\n\n"
                        "Please answer the question directly with a lot of extra detail, citing relevant sections (author, year) for support. Everything that is taken word for word from a source should be in quotation marks."
                        f"At the end, Suggest a further question/experiment that relates, and cite them as (author, year): {combined_input}",
                        "generate graphs and do some data analysis"
                    )

                    response = f" {response_text}"
                    client = OpenAI(api_key=openai_api_key) 
                    integrated_response = chat_prompt(client, query_for_llm)
                    sources_formatted = "\n".join(sources) 
                    citations = sources_formatted
                    
                    response = f" {integrated_response}\n"
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
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4')
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
    
def main():
    st.sidebar.title("Science.ai")
    init_research_assistant()
    run_research_assistant_chatbot()

if __name__ == '__main__':
    main()
