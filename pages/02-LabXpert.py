from pathlib import Path
import pickle
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
import streamlit as st
# from decouple import config
from PIL import Image




# st.markdown('## LabXpert')
# col1, col2 = st.columns((2,1))
# with col1:
#     st.markdown(
#         f"""
#         #### [Sign Up Now 🤘🏻]('https://buy.stripe.com/test_bIY7vu4Ob2WF9UseUU')
#         """
#     )
# # with col2:
# #     image = Image.open('./assets/DALL·E 2023-01-08 17.53.04 - futuristic knight robot on a horse in cyberpunk theme.png')
# #     st.image(image)


# st.markdown('### Already have an Account? Login Below👇🏻')
# with st.form("login_form"):
#     st.write("Login")
#     email = st.text_input('Enter Your Email')
#     password = st.text_input('Enter Your Password')
#     submitted = st.form_submit_button("Login")


# if submitted:
#     if password == config('SECRET_PASSWORD'):
#         st.session_state['logged_in'] = True
#         st.text('Succesfully Logged In!')
#     else:
#         st.text('Incorrect, login credentials.')
#         st.session_state['logged_in'] = False


# if 'logged_in' in st.session_state.keys():
#     if st.session_state['logged_in']:
#         st.markdown('## Ask Me Anything')
#         question = st.text_input('Ask your question')
#         if question != '':
#             st.write('I drink and I know things.')

# https://buy.stripe.com/test_bIY7vu4Ob2WF9UseUU

# from st_paywall import add_auth
st.set_page_config(page_icon='🧬', page_title='LabXpert', layout="wide")






import streamlit as st

# from st_login_form import login_form


    
    






with st.sidebar:
    st.sidebar.title("LabXpert ")
    st.sidebar.caption("Copyright © 2024 LabXpert, Inc. All rights reserved.")
    
    # Adding text and formatting to the sidebar
    st.markdown(
        """
        <span style='font-size: 13px; line-height: 0;'>
            Enhancing the scientific research process with AI:<br>
            - Trained on a wealth of scientific literature.<br>
            - Reduces hours of literature searching.<br>
            - Data analysis to just a few seconds.<br>
            <br>
        </span>
        """,
        unsafe_allow_html=True
    )

# add_auth(required=True, login_sidebar=False)
# login_button_color="#364390"
# # st.sidebar.write(f'Welcome {st.session_state.email}!')
# st.sidebar.markdown(
#     f"""
#     <span style='font-size: 13px; line-height: 1.2;'>
#         Welcome {st.session_state.email}!
#         <br>
#     </span>
#     """,
#     unsafe_allow_html=True
# )

# st.write("You're all set and subscribed and ready to go! 🎉")
# import streamlit_authenticator as stauth
# # --- USER AUTHENTICATION ---
# names = ["Peter Parker", "Rebecca Miller"]
# usernames = ["pparker", "rmiller"]

# # load hashed passwords
# file_path = Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open("rb") as file:
#     hashed_passwords = pickle.load(file)

# authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#     "sales_dashboard", "abcdef", cookie_expiry_days=30)

# name, authentication_status, username = authenticator.login("Login", "main")

# if authentication_status == False:
#     st.error("Username/password is incorrect")

# if authentication_status == None:
#     st.warning("Please enter your username and password")

# if authentication_status:
    
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def estimate_relevance(question):
    # List of simpler word
    
    unrelated_words = ['hi', 'hello', 'how are you', 'good morning', 'good evening']

    # Check if the entire question matches any of the unrelated phrases
    if any(question.strip().lower() == word for word in unrelated_words):
        return 0  # Return a score of 0 for unrelated words or simple greetings


def estimate_complexity(question):
    # List of simpler word
    
    
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
        return 5  # High complexity
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


load_dotenv()
def run_research_assistant_chatbot():


    st.markdown("<h1 style='text-align: center;'>Research Xpert 📄</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><i>Ask questions about REAL scientific articles, with fully cited responses, Harvard style.</i></p>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center;'>Enjoy fully cited responses, Harvard style.</p>", unsafe_allow_html=True)

    # st.divider()

    


    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"



    def load_chat_history():
        with shelve.open("chat_history") as db:
            # return db.get("messages", [])
            return []
        

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
    
    from pinecone import Pinecone, ServerlessSpec

    
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    # if 'pinecone' not in pc.list_indexes().names():
    #     pc.create_index(
    #         name='pinecone', 
    #         dimension=1536, 
    #         metric='euclidean',
    #         spec=ServerlessSpec(
    #             cloud='aws',
    #             region='us-west-1'
    #         )
    #     )

    def formulate_response(prompt):
        openai_api_key = os.environ["OPENAI_API_KEY"]
        db = pc.Index("pinecone")
        chat_history = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])
        prompt_with_history = f"Previous conversation:\n{chat_history}\n\nYour question: {prompt} Answer the question directly."
        k = estimate_complexity(prompt)     
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        xq = client.embeddings.create(input=prompt, model="text-embedding-3-small").data[0].embedding
        results = db.query(vector=[xq], top_k=k, include_metadata=True)
        citations = ""
        with st.spinner("Thinking..."):
            print(results.matches[0].score)
            if results.matches and results.matches[0].score > 0.0:
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                db_response = db.query(vector=[xq], top_k=k, include_metadata=True)
                sources = {}
                for match in db_response.matches:
                    metadata = match.metadata
                    authors = metadata.get('authors', 'Unknown')
                    year = metadata.get('year', 'Unknown')
                    citation_key = f"({authors.split(',')[0]} et al., {year})"
                    if citation_key not in sources:
                        sources[citation_key] = (
                            f"\n🦠 {metadata.get('authors', 'Unknown')}\n"
                            f"({metadata.get('year', 'Unknown')}),\n"
                            f"\"{metadata['title']}\",\n"
                            f"PMID: {metadata.get('pub_id', 'N/A')},\n"
                            f"Available at: {metadata.get('url', 'N/A')},\n"
                            f"Accessed on: {datetime.today().strftime('%Y-%m-%d')}\n"
                        )
                citations = "\n".join(sources.values())
                query_for_llm = (
                    f"Answer directly with detail: Question: {prompt_with_history}\n\n"
                    f"Cite each sentence as (author, year) {citations} \n"
                    "Do NOT list references."
                )
                integrated_response = model.predict(query_for_llm)
                response = f"{integrated_response}\n"
            else:
                model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
                response = model.predict(prompt_with_history)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "citations": citations
        })

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
            
    import openai

    def generate_prompts_experiment(base_prompt):

        openai_api_key = st.secrets["OPENAI_API_KEY"]

        prompt_variations = (
            "Only return the prompt"
            f"Using 10 words, come up a concise prompt based on the question about designing an experiment: {base_prompt}"

        )
        model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
        integrated_response = model.predict(prompt_variations)
        return integrated_response
    def generate_prompts_explain(base_prompt):

        openai_api_key = st.secrets["OPENAI_API_KEY"]

        prompt_variations = (
            "Only return the prompt"
            f"Using 10 words, come up a concise prompt based on the question about making the concept easier to understand: {base_prompt}"

        )
        model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
        integrated_response = model.predict(prompt_variations)
        return integrated_response

    def generate_prompts_previous(base_prompt):

        openai_api_key = st.secrets["OPENAI_API_KEY"]

        prompt_variations = (
            "Only return the prompt"
            f"Using 10 words, come up a concise prompt asking about either an experiment done in the past about it, designing an experiment about it, or about outstanding areas of research on that topic, whichever question suits best to follow on with.: {base_prompt}"

        )
        model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
        integrated_response = model.predict(prompt_variations)
        return integrated_response

    def generate_prompts_outstanding(base_prompt):

        openai_api_key = st.secrets["OPENAI_API_KEY"]

        prompt_variations = (
            "Only return the prompt"
            f"Using 10 words, come up a concise prompt based on the question about outstanding areas of research on that topic: {base_prompt}"

        )
        model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-0125")
        integrated_response = model.predict(prompt_variations)
        return integrated_response





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
                    
            # with st.container():
            #     with st.spinner("Generating prompts..."):
                    # label = generate_prompts_experiment(st.session_state.messages)
    # label2 = generate_prompts_explain(st.session_state.messages)
    # label3 = generate_prompts_previous(st.session_state.messages)
    # label4 = generate_prompts_outstanding(st.session_state.messages)

    # selected_prompt = None

    # col1, col2 = st.columns(2)
    # with col1:
    # if st.button(label=label3, key = f"{label3}1"):
    #     selected_prompt = label3
    # with col2:
    #     if st.button(label=label2, key = f"{label3}2"):
    #         selected_prompt = label2

    # col3, col4 = st.columns(2)
    # with col3:
    #     if st.button(label=label3, key = f"{label3}3"):
    #         selected_prompt = label3
    # with col4:
    #     if st.button(label=label4, key = f"{label3}4"):
    #         selected_prompt = label4

    # if selected_prompt is not None:
    #     st.session_state.messages.append({"role": "user", "content": selected_prompt})
    #     formulate_response("selected", selected_prompt)

                    
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
    # st.title("Data Xpert 📊")
    st.markdown("<h1 style='text-align: center;'>Data Xpert 📊</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><i>Upload your data file/s and I can produce graphs from your experiment</i></p>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center;'>Your data analysis Xpert!</p>", unsafe_allow_html=True)


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
        # st.caption('Upload your data file/s and I can produce graphs from your experiment')
        # st.markdown('Your data analysis Xpert!')
        # st.divider()
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
    # st.sidebar.image("pic.png"
    st.sidebar.divider()
    # Set 'Research Xpert 🔬' as the default selected option
    # tab1, tab2 = st.sidebar.tabs(["Research Xpert", "Data Xpert"])

    # with tab1:
    #     init_research_assistant()
    #     run_research_assistant_chatbot()

    # with tab2:
    #     init_data_analysis()
    #     run_data_analysis_chatbot()
    
    from streamlit_pills import pills
    
    # client = login_form()

    # if st.session_state["authenticated"]:
    #     if st.session_state["username"]:
    #         st.success(f"Welcome {st.session_state['username']}")
    #     else:
    #         st.success("Welcome guest")
    # else:
    #     st.error("Not authenticated")

    selected = pills("Choose which Lab Xpert you would like to chat with:", ["Research Xpert", "Data Xpert"], ["📄", "📊"])
    
    
    
    import webbrowser
    url = "https://docs.google.com/forms/d/e/1FAIpQLScFrXl_pc9fG8dx2vTPgj9UUPtDt3vBl-LSj59i1hgDQFrVEA/viewform?usp=sf_link"

    # Create three columns



    if st.button('Leave us some feedback'):
        webbrowser.open_new_tab(url)
    
    st.divider()
    
    if selected == 'Research Xpert':
        init_research_assistant()
        run_research_assistant_chatbot()
    elif selected == 'Data Xpert':
        init_data_analysis()
        run_data_analysis_chatbot()
        
        
        
        
    
    # chatbot_mode = st.sidebar.radio("Select an AI Xpert", ('Research Xpert ', 'Data Xpert '), index=0)
    # if chatbot_mode == 'Research Xpert ':
    #     init_research_assistant()
    #     run_research_assistant_chatbot()
    # elif chatbot_mode == 'Data Xpert ':
    #     init_data_analysis()
    #     run_data_analysis_chatbot()

    
    

if __name__ == '__main__':


    # Initialize the session state key
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    
    if st.session_state["authenticated"]:
        main()
    elif st.session_state['authenticated'] == False:
        st.error("Please create an account before using LabXpert")
        
        
