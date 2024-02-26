from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


llm = ChatOpenAI()

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm | output_parser

st.title("LangChain - Simple LLM Chain")
st.info('This chain uses the OpenAI Language Model to answer questions. It is a simple example of a chain. It is not intended to be used in production. It does not have memory or retain a history of the conversation')
# question = "How can langsmith help with testing?"

with st.chat_message('Human'):
    question = st.chat_input("Question")

if question is not None and question != '':

    with st.chat_message('Human'):
        st.write(question)

    with st.spinner("Waiting for response..."):
        response = chain.invoke({"input": question})
        with st.chat_message('AI'):
            st.write(response)
