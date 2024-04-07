import streamlit as st
import openai
import os
import shelve
import tempfile
from datetime import datetime
from openai import OpenAI

# Assuming dotenv is used for OpenAI API key loading
from dotenv import load_dotenv

# Custom Classes - Ensure these classes are properly defined in your environment
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores.chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI

# Initialize environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Assuming OpenAI() instantiation is unnecessary if using 'openai' directly after import
openai.api_key = openai_api_key

def load_chat_history():
    with shelve.open("chat_history.db") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history.db") as db:
        db["messages"] = messages

def init_research_assistant():
    if "messages_research_assistant" not in st.session_state:
        st.session_state.messages_research_assistant = load_chat_history()

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

def query_assistant(user_prompt, assistant_id):
    # This example assumes 'assistant_id' and 'thread_id' are managed elsewhere in your code
    # Ensure 'thread_id' is properly initialized in your session state or passed as a parameter
    thread_id = st.session_state.get("thread_id", None)
    if thread_id is None:
        st.error("Thread ID is not set. Unable to query assistant.")
        return "No thread ID."

    response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[{"role": "system", "content": "Analyze the following data."}, 
                  {"role": "user", "content": user_prompt}],
        assistant_id=assistant_id
    )

    return response.choices[0].message["content"] if response.choices else "Failed to get a response."

def run_research_assistant_chatbot():
    st.title("Research Assistant 🔬")
    st.caption("Analyze your experimental data")

    user_prompt = st.text_input("Enter your query:", key="user_query")
    if user_prompt and st.button("Ask"):
        # Example assistant ID - replace with your actual assistant ID
        assistant_id = "your-assistant-id"
        response = query_assistant(user_prompt, assistant_id)
        st.session_state.messages_research_assistant.append({"user": user_prompt, "assistant": response})
        save_chat_history(st.session_state.messages_research_assistant)
        st.write(response)

def main():
    init_research_assistant()
    run_research_assistant_chatbot()

if __name__ == "__main__":
    main()
