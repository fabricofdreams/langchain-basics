from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Instantiate the language model, the document loader, the embeddings and the text splitter
llm = ChatOpenAI()
loader = WebBaseLoader("https://docs.smith.langchain.com")
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
output_parser = StrOutputParser()

# Load the documents, split them into chunks and create a vector store from chunks and embeddings
docs = loader.load()
chunks = text_splitter.split_documents(docs)
vector = FAISS.from_documents(chunks, embeddings)

retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_template("""You are world class technical documentation writer. Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")


# document_chain = create_stuff_documents_chain(llm, prompt)
document_chain = prompt | llm | output_parser

retrieval_chain = create_retrieval_chain(
    retriever, document_chain)

st.title("LangChain - Retrieval Chain")
st.info('This chain uses the OpenAI Language Model to retrieve documents from a vector store feeded with documents from "https://docs.smith.langchain.com".')

with st.chat_message('Human'):
    question = st.chat_input("Question")

if question is not None and question != '':
    with st.chat_message('Human'):
        st.write(question)

    with st.spinner("Waiting for response..."):
        response = retrieval_chain.invoke(
            {"input": question})
        with st.chat_message('AI'):
            st.write(response.get("answer"))
