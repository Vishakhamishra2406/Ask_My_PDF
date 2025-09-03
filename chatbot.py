import streamlit as st
from pdf_chatbot import PDFChatbot
from utils import load_api_key, save_uploaded_file

# Page ka title aise dete hn
st.set_page_config(page_title="PDF Chatbot", layout="centered")

# Headline ki designing kri hai html,css use krke
st.markdown(
    "<h1 style='text-align: center; color: Red; font-family: Courier;'>AskMyPDF</h1>",
    unsafe_allow_html=True
)

# Change background and sidebar color : just for designing css use kra h
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #e6e6fa;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
<div style='font-family: Arial; font-size: 15px; color: white;'>
<h4> How to Use AskMyPDF</h4>
<ol>
<li>📄 <b>Upload</b> a PDF file using the upload button.</li>
<li>💬 <b>Ask questions</b> about the content of the uploaded PDF.</li>
<li>✨ Enjoy your AI-powered chatbot!</li>
</ol>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        try:
            google_api_key = load_api_key()
            chatbot = PDFChatbot(google_api_key)
            temp_file_path = save_uploaded_file(uploaded_file)
            raw_text = chatbot.extract_text(temp_file_path)
            chunks = chatbot.split_text(raw_text)
            chatbot.embed_and_store(chunks)
            st.session_state['chatbot'] = chatbot
            st.session_state['file_uploaded'] = True
            st.success("PDF processed and ready for questions!")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a greeting from the assistant to start the conversation
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am ready to answer your questions about the uploaded PDF. Please upload one to begin."})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if st.session_state.get('file_uploaded'):
    if prompt := st.chat_input("Ask a question about the PDF:"):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chatbot = st.session_state['chatbot']
                    answer = chatbot.ask(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Failed to get answer: {e}")
else:
    # A simple chat input placeholder when no file is uploaded
    st.chat_input("Please upload a PDF first to ask questions.", disabled=True)