# LabXpert

I've created an AI chatbot that uses a combination of the openai llm, the openai embedding model and a pinecone vector db storage to create a RAG ML pipeline to create an AI Lab assitant that can cite all it's sources using Harvard referencing system from sources like pubmed and bioarchive. 

Checkout the `sheroku` branch to see the latest version of project that is deployed live here:

# https://labxpert.streamlit.app/

On the `main` branch I use a local chroma database along with the embeddings model, but the issue I faced is that because I needed to push my code to github for the deployment process, as a rule of thumb, github is not for storage! I wanted to make the database as big as possible to give the RAG model as much scientific information asd possible to reduce risk of hallucination.

I tried various methods which you can checkout on multiple branches of storing the vector database, such as:

- storing the vectors db on:
    - AWS
    - locally on a .zip file
    - locally as a regular folder (small database size due to github restrictions (even with using git lfs))
    - Pinecone storgae (which was the golden answer!)

Using pinecone as cloud storgae, I could save my db with a cosine embedding and easily get the openai embeddings model to make queries with the user prompts - and the database can hold up to 2M vectors!

### Yes, that's right - 2M vectors

(Create an account to use or login as a guest)

You can also checkout my simple website to learn more about this personal project: https://www.labxpert.co.uk/

Watch a video demo here:

[Download and watch the video](./sample-video.mov)
