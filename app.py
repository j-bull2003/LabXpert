import openai
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import shelve
from datetime import datetime

# Load OpenAI API key from .env for security and best practices
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize necessary global variables or configurations
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def upload_file(uploaded_file):
    """
    Uploads a file to OpenAI for use with the Assistants API.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()  # Ensure all data is written
        response = openai.File.create(
            file=open(tmp_file.name, "rb"),
            purpose='assistants'
        )
        os.unlink(tmp_file.name)  # Cleanup temporary file
    return response["id"]

def create_assistant_with_file(file_id):
    """
    Creates an Assistant configured for data visualization with the uploaded .csv file.
    """
    response = openai.Assistant.create(
        name="Data Visualizer",
        description="Generates data visualizations from .csv files.",
        model="gpt-4-turbo-preview",
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
        file_ids=[file_id]
    )
    return response["id"]

def query_assistant(user_prompt):
    """
    Queries the Assistant and returns a response based on the user's prompt.
    """
    thread_response = openai.Thread.create(
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    thread_id = thread_response["id"]
    
    run_response = openai.Thread.run(
        thread_id=thread_id,
        model="gpt-4-turbo-preview",
        instructions="Generate a detailed analysis based on the .csv data.",
    )
    # Extract and return the latest message from the Assistant as the response
    latest_message = run_response["messages"][-1]["content"]
    return latest_message

def main():
    """
    Main Streamlit app function for interaction with the research assistant.
    """
    st.sidebar.title("Science.ai - Research Assistant")
    st.title("Research Assistant 🔬")
    st.caption('Analyze your experimental data with AI')
    
    uploaded_file = st.file_uploader("Upload your data file (.csv only)", type="csv")
    if uploaded_file is not None:
        file_id = upload_file(uploaded_file)
        assistant_id = create_assistant_with_file(file_id)
        st.success(f"Assistant ID: {assistant_id} - ready to analyze your data.")
    
    user_prompt = st.text_input("Ask a question or describe the analysis you need:")
    if user_prompt and uploaded_file:
        response = query_assistant(user_prompt)
        st.write("Assistant's response:", response)

if __name__ == '__main__':
    main()
