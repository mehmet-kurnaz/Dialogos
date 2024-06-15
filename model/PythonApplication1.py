import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from LocalLlama import RagLLama
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
from localLLM import GPT2Assistant
from RAG import RagLLM


def staticmodel():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


def main():
    st.set_page_config(page_title="Promptus",
                       page_icon=":books:")
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    st.write(current_directory)
    load_dotenv()

    st.write(css, unsafe_allow_html=True)
    staticmodel()

    button_durumu = st.button("RAG")
    button_durumu2 = st.button("LocalLLM")
    button_durumu3 = st.button("FOR English")
    user_question = st.text_input("Ask a question about your documents:")

    if button_durumu:
        model = RagLLM()

        raw_text = model.get_text()

        # get the text chunks
        text_chunks = model.get_text_chunks(raw_text)

        # create vector store
        vectorstore = model.get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = model.get_conversation_chain(
            vectorstore)

        if user_question:
            model.handle_userinput(user_question)

    if button_durumu2:
        model = GPT2Assistant()
        st.write("Gpt2ModeliFineTuneEdiliyor...")

        if user_question:
            respond = model.generate_response(user_question)
            print("user question")
            st.write(respond)
        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if button_durumu3:
        model = RagLLama()

        raw_text = model.get_text()

        # get the text chunks
        text_chunks = model.get_text_chunks(raw_text)

        # create vector store
        vectorstore = model.get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = model.get_conversation_chain(
            vectorstore)
        if user_question:
            model.handle_userinput(user_question)



if __name__ == '__main__':
    print("başla")
    main()
