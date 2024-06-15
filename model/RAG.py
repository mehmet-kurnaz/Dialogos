from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

import streamlit as st
import torch
import os


class RagLLM:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_text(self):
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)

        file_path = 'app\\PythonApplication1\\PythonApplication1\\content\\meta2.txt'

        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
            print("it was here")
        return self.text

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        print("text chunking")

        self.chunks = text_splitter.split_text(text)
        return self.chunks

    def get_vectorstore(self, text_chunks):
        self.embeddings = OpenAIEmbeddings(openai_api_key='sk-proj-LjVCVqVnggFaW7wOkLVAT3BlbkFJqQAWFJ7HWv3mYEBepkA3')
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        self.vectorstore = FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)
        print("vector store")
        return self.vectorstore

    def get_conversation_chain(self, vectorstore):
        llm = ChatOpenAI()
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        print("conservation")
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def handle_userinput(self, user_question):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
