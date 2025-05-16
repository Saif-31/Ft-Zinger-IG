import streamlit as st
from app import create_interior_design_chatbot
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = create_interior_design_chatbot()

# Sidebar with New Chat button
with st.sidebar:
    if st.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("Zinger Interior Design Bot")
st.markdown("###### Welcome to the Zinger Interior Design Bot! What can I help you today?")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about interior design..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        response = st.session_state.chatbot.invoke(
            {"user_input": prompt},
            config={"configurable": {"session_id": "default_session"}}
        )
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.markdown(response.content)

