from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import streamlit as st

load_dotenv()

# Instantiate the language model, the document loader, the embeddings and the text splitter
llm = ChatOpenAI()
loader = WebBaseLoader("https://docs.smith.langchain.com")
embeddings = OpenAIEmbeddings()
text_spliter = RecursiveCharacterTextSplitter()

# Load the documents, split them into chunks and create a vector store from chunks and embeddings
docs = loader.load()
chunks = text_spliter.split_documents(docs)
vector = FAISS.from_documents(chunks, embeddings)

# Create a retriever from the vector store
retriever = vector.as_retriever()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# # Create the prompt that will instruct the system about its role
# prompt = ChatPromptTemplate.from_template("""You are world class technical documentation writer. Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""").extend(
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
# )

# # Create the document chain
# document_chain = create_stuff_documents_chain(llm, prompt)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

# Create the retriever chain
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Create a conversational chain with the retriever_chain in mind
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

st.title("Langchain - Conversation Retrieval Chain")
st.info('This chain uses the OpenAI Language Model to retrieve documents from a vector store feeded with documents from "https://docs.smith.langchain.com".')


question = st.chat_input("Write your question here...")

if question is not None and question != '':
    with st.spinner("Waiting for response..."):
        response = retrieval_chain.invoke(
            {"input": question, "chat_history": st.session_state.chat_history})

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(
        AIMessage(content=response.get("answer")))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)
