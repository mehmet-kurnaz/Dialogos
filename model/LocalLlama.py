#pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a PHILOSPHTRG",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
#prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
#print(outputs[0]["generated_text"])

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
import torch
import os
import streamlit as st


class RagLLama:
    def __init__(self):
        self.text = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_text(self):
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)

        file_path = "PythonApplication1/PythonApplication1/content/METAeng.txt"

        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
            print("it was here")
        return self.text

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=400,
            chunk_overlap=200,
            length_function=len
        )
        print("text chunking")

        self.chunks = text_splitter.split_text(text)
        return self.chunks

    def get_vectorstore(self, text_chunks):
        print("embedding")
        self.embeddings = GPT4AllEmbeddings()
        print("embeding bitti vectorstore")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        #self.vectorstore = FAISS.from_texts(texts=text_chunks, embedding=self.embeddings)
        #self.vectorstore.save_local("faiss")

        self.vectorstore = FAISS.load_local("faiss", self.embeddings, allow_dangerous_deserialization=True)
        print("vector store bitti")
        return self.vectorstore

    def get_conversation_chain(self, vectorstore):
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain_community.llms import Llamafile
        from transformers import pipeline
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



        pipe = pipeline("text-generation", model="PythonApplication1/PythonApplication1/LLAMA_model", torch_dtype=torch.bfloat16, device_map="auto")
        llm = HuggingFacePipeline(pipeline=pipe)

        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        print("conservation")
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory
        )
        del pipe
        torch.cuda.empty_cache()
        return conversation_chain

    def handle_userinput(self, user_question):

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        st.write("hımm biraz düşünmeme izin ver")
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
