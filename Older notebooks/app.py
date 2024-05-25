# !pip install pypdf langchain_community rapidocr-onnxruntime langchain_openai faiss-cpu langchain streamlit

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_retriever(data):

    faiss_index = FAISS.from_documents(data, OpenAIEmbeddings())
    return faiss_index.as_retriever()

def get_conversational_chain(retriever):

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    ### Answer question ###

    qa_system_prompt = """
    You are an assistant for creating MCQ questions based on the context provided. \
    Use the following pieces of retrieved context to formulate the MCQ question. \
    Don't repeat same question. \
    Display one MCQ question and options only, don't display answer. \
    Question in one line and options in separate line and wait for the user response. \
    If user answer correct say your answer is correct, and ask would you like to try another question from same topic.\
    If user answer is wrong say your answer is wrong and display the correct answer with explanation, and ask would you like to try another question from same topic.\
    Don't repeat same question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    
    rag_chain = create_retrieval_chain(retriever_chain, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


def process_user_input(user_input:str) -> str:
    chain = st.session_state.chain
    return chain.invoke( {"input": user_input}, config={"configurable": {"session_id": "abc123"}},)["answer"]

def main():
    st.set_page_config("MCQ from PDF file")
    st.header("MCQ from PDF using OpenAI")

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # Initialize history session state if not already initialized
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Add all previous messages to the chat history UI
        for msg in st.session_state.history:
            messages.chat_message(msg['role']).write(msg['msg'])

        # Add user asked question to chat history and session state history
        st.session_state.history.append({"role":"user", "msg":prompt})
        messages.chat_message("user").write(prompt)

        # Get the output chain
        output = process_user_input(prompt)

        # Add chain respose to the chat history UI and session state history
        st.session_state.history.append({"role":"assistant", "msg":output})
        messages.chat_message("assistant").write(f"Echo: {output}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                with open("/tmp/1.pdf", "wb") as f:
                    f.write(pdf_docs.getvalue())

                data = PyPDFLoader("/tmp/1.pdf")
                # data = get_pdf_data("document.pdf")
                retriever = get_retriever(data.load())
                chain = get_conversational_chain(retriever)
                st.session_state['chain'] = chain
    
    print()
                

if __name__ == "__main__":
    main()