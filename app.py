import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.tools import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environment Variables (API Keys)
load_dotenv()

# 2. Streamlit Page Configuration
st.set_page_config(page_title="Team AI Tutor", page_icon="🤖", layout="centered")
st.title("🤖 Team AI Knowledge Base")
st.markdown("Ask me anything about our training materials, documentation, or project guidelines.")

# 3. Initialize the Backend Services (Cached so they don't reload on every keystroke)
@st.cache_resource
def initialize_rag_pipeline():
    # A. Setup the Embedding and Database (Same as before)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index_name="off-tutor", embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # B. Define Tool 1: Internal Database
    pinecone_tool = create_retriever_tool(
        retriever,
        name="search_internal_docs",
        description="Searches the internal office knowledge base. Always use this FIRST for questions about company guidelines, projects, or team documents."
    )
    
    # C. Define Tool 2: The Live Internet
    web_tool = DuckDuckGoSearchRun(
        name="search_internet",
        description="Searches the live internet for up-to-date information, news, or general knowledge that isn't in the internal docs.",
        max_results=1
    )
    
    # D. Give the tools to the AI
    tools = [pinecone_tool, web_tool]
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.2, 
        max_tokens=1024
    )
    
    # E. The Agent's Brain (Prompt)
    system_prompt = (
        "You are an intelligent assistant for our office team. "
        "You have access to internal documents and the internet. "
        "If a user asks about internal matters, use the 'search_internal_docs' tool. "
        "If they ask about current events or general knowledge, use the 'search_internet' tool. "
        "CRITICAL RULE: When you use a tool, you MUST use the text it returns to write a complete, helpful answer to the user's question. "
        "Do not just say that you searched the internet. Give the actual facts and details you found. "
        "At the very end of your answer, add a short sentence stating if you used internal docs or the internet."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # This is the "scratchpad" where the AI does its thinking
    ])
    
    # F. Assemble the Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # The Executor is the loop that actually runs the tools and feeds data back to the LLM
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Start the pipeline
rag_chain = initialize_rag_pipeline()

# 4. Streamlit Chat Interface Logic
# This stores the chat history so it doesn't disappear when the page refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Input Box (Where your teammates type their question)
if user_question := st.chat_input("Ask a question about the team docs..."):
    
    # Show the user's question on the screen immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Show a loading spinner while Groq and Pinecone do the work
    with st.chat_message("assistant"):
        # 1. Create the UI Callback Handler
        st_callback = StreamlitCallbackHandler(st.container())
        
        # 2. Pass the callback to the Agent when you invoke it
        response = rag_chain.invoke(
            {"input": user_question},
            {"callbacks": [st_callback]} # <-- THIS IS THE MAGIC LINE
        )
        
        # Extract the actual answer string from the response object
        final_answer = response["output"]
        
        # Display the final answer
        st.markdown(final_answer)
        
        # Save the answer to the chat history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})