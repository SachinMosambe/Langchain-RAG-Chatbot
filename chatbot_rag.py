import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="Chatbot", page_icon=":robot:", layout="wide")

# Sidebar content
with st.sidebar:
    st.title("Welcome to the intelligent chatbot")
    st.markdown("Made by Sachin Mosambe powered by OpenAI, Langchain and Pinecone.")
    st.markdown("---")
    st.markdown("### Chat History")
    
    # Display the previous chat history (if any)
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if isinstance(message, HumanMessage):
                st.markdown(f"**User**: {message.content}")
            elif isinstance(message, AIMessage):
                st.markdown(f"**Assistant**: {message.content}")

# Main content area
st.title("Chatbot")

# Welcome quote
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))
    st.markdown("### Welcome to the Chatbot! How can I assist you today?")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize the Pinecone vector store
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize embeddings model and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Create the chat message history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
prompt = st.chat_input("Type your message here...")

# If user submits a message
if prompt:
    # Add user message to the chat
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Initialize LLM for the assistant
    llm = ChatOpenAI(model="gpt-4o", temperature=1)

    # Creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5}
    )
    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # System prompt for the assistant
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}:"""

    # Format the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    # Add system prompt to the chat history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # Invoke the model and get a response
    result = llm.invoke(st.session_state.messages).content

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))



