import streamlit as st
# from utils.auth import create_account_st_form
# from streamlit_extras.switch_page_button import switch_page
# https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQ-lODtKMmdRefE1cTUwOH_0AR5qNyJKDvsUqIsOhcvFBJVfyVX 
# st.set_page_config(page_title='Streamlit x Flask x Stripe Example', page_icon='🌶️', initial_sidebar_state="auto", menu_items=None)

# if create_account_st_form():
#     switch_page('your app')
from st_login_form import login_form 
import webbrowser
from streamlit.source_util import _on_pages_changed, get_pages
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages_from_config, add_page_title
st.set_page_config(page_icon='🧬', page_title='LabXpert', layout="centered", initial_sidebar_state="auto")
st.write("<h1 style='text-align: center;'>Welcome to LabXpert</h1>", unsafe_allow_html=True)
# st.write("<div style='text-align: center;'>LabXpert is an AI lab assistant which assists with all stages of the scientific process including hypothesis generation, experimental design, troubleshooting, and data analysis. With a simple chat interface, scientists can easily interact with LabXpert, trained on a wealth of scientific literature, reducing hours of literature searching and data analysis to just a few seconds.</div><br>", unsafe_allow_html=True)



# import base64

# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()
# img = get_img_as_base64("image.png")

# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#     background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
#                 url("data:image/png;base64,{img}");
#     background-size: 99%;
#     background-position: top left;
#     background-repeat: no-repeat;
#     background-attachment: local;
# }}

# [data-testid="stHeader"] {{
#     background: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"] {{
#     right: 2rem;
# }}
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)







st.markdown(
    """
    <div style='text-align: center;'>
    <br>
        LabXpert enhances your experience beyond the capabilities of a standard chat GPT by ensuring 
        complete data privacy as our models are not trained on your data,
        accurate and reliable information sourced directly from trusted platforms like PubMed and BioArchive,
        and efficient research without the need to sift through countless articles.<br>
        <br>
    </div>
    """,
    unsafe_allow_html=True
)

# url = 'https://www.labxpert.co.uk/'
# url = "https://docs.google.com/forms/d/e/1FAIpQLScFrXl_pc9fG8dx2vTPgj9UUPtDt3vBl-LSj59i1hgDQFrVEA/viewform?usp=sf_link"

import streamlit as st
import webbrowser

url = 'https://www.labxpert.co.uk/'

col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 2, 1, 1, 1])

# Place the button in the middle column
with col4:
    # if st.button('Learn more here'):
    #     webbrowser.open_new_tab(url)
    st.link_button("Learn more here", "https://www.labxpert.co.uk/")




# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
# add_page_title()

# show_pages_from_config()

client = login_form(create_title="Create a new account", login_title="Login to existing account")
# add_page_title()

# show_pages_from_config()

# pages = ["Home", "Sign_Up", "LabXpert"]

def hide_pages(pages_to_hide):
    for page in pages_to_hide:
        st.sidebar.markdown(f"## {page}")
        st.sidebar.markdown("This page is hidden.")

if st.session_state["authenticated"]:
    if st.session_state["username"]:
        st.success(f"Welcome {st.session_state['username']}")
        # st.link_button("Go", "./LabXpert")
    else:
        st.success("Welcome guest")
        switch_page('LabXpert')
else:
    st.error("Not authenticated")
 
 
# show_pages(
#     [
#         Page("streamlit_app.py", "Home", "🏠"),
#         Page("another.py", "Another page"),
#     ]
# )

# hide_pages(["Another page"])

# import streamlit as st

hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)