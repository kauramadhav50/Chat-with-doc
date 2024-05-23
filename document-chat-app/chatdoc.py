import os
from apikey2 import apikey

import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = apikey
st.title('Chat with Document')
uploaded_file = st.file_uploader('upload a file:', type=['pdf', 'docx', 'txt'])
add_file = st.button('Add File')

if uploaded_file and add_file:
    with st.spinner('Reading, chunking and embedding file....'):
        bytes_data = uploaded_file.read()
        file_name = os.path.join('./', uploaded_file.name)
        with open(file_name, 'wb') as f:
            f.write(bytes_data)

        name, extension = os.path.splitext(file_name)
        
        if extension =='.pdf':
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_name)
        
        elif extension == '.docx':
            from langchain.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_name)
        
        elif extension == '.txt':
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_name)
        
        else:
            st.write('Document format is not supported!')
        
        # loader = TextLoader(file_name)
        # loader = TextLoader(file_name, encoding='your_file_encoding')
        document = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        chunks = text_splitter.split_documents(document)
        
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)
        retriever = vector_store.as_retriever()
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)
        st.session_state.crc = crc
        st.success('File uploaded, chunked and embedded successfully')


question = st.text_input('Input your question')

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        response = crc.run({'question': question, 'chat_history': st.session_state['history']})
        st.session_state['history'].append((question, response))
        st.write(response)

        for prompts in st.session_state['history']:
            st.write("Question:" + prompts[0])
            st.write("Answer:" + prompts[1])
        
        
        