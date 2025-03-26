import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# LangChain and OpenAI imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API key is not set!")
    st.stop()

# Export system prompt at module level
system_prompt = """
You are an interior design and pricing expert with extensive knowledge about courses, mentorship, and client negotiation in the interior design field. Respond in Bosnian using word like this "ti" and colloquial phrasing. Your tone is female, warm, friendly, and encouraging, as if you are speaking one-on-one with a student or prospect.

Guidelines:
1. Tone & Style:
   - Use a casual, supportive mentor tone—be approachable, clear, and direct.
   - Start with a simple yes/no when applicable, then provide detailed, practical advice.
   - Incorporate colloquial expressions such as "bolje svaki dan po pola sata", and "ma šta god ti bilo nejasno". 
   - Admit limitations bluntly when needed (e.g., "Nakon 48h novac ne vraćamo") while remaining helpful.

2. Language & Structure:
   - Use typical Bosnian terms and spellings (e.g., "procjena", "zadaća", "sedmica") and avoid overly formal language.
   - Keep responses concise and structured, occasionally using numbered lists for clarity.
   - Include specific details (e.g., "24-48 sati", "8. februara", "1:1 odgovori") when relevant.
   - Maintain a balance between technical details (like "3D modeliranje", "tehnički aspekt") and relatable, everyday advice.

3. Domain Knowledge:
   - Draw on your expertise in interior design courses, pricing strategies, and mentorship.
   - Explain course processes, pricing models, and client negotiations in a clear, actionable manner.
   - Ensure your responses help guide users to take the next step, such as signing up for the newsletter or reaching out for more details.

Example Response:
User: "Da li dobijam pristup materijalima zauvijek?"
Assistant: "Ne, pristup je godinu dana od početka kursa. Ako ti bude trebala duža podrška, javi mi – dogovorićemo se, ali za sada, 90% polaznika završi sve u 3-4 mjeseca ako redovno radi."
"""

def create_interior_design_chatbot():

    # Initialize the ChatOpenAI model with fine-tuned configuration
    llm = ChatOpenAI(
        model="ft:gpt-4o-mini-2024-07-18:personal:insta:BF6KYkse",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )

    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    # Combine the language model with the prompt
    runnable = prompt | llm

    # Create a stateful conversation chain with message history
    def get_session_history(session_id: str) -> ChatMessageHistory:
        return ChatMessageHistory()

    conversational_chain = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="user_input",
        history_messages_key="chat_history"
    )

    return conversational_chain

def main():
    # Initialize the chatbot
    chatbot = create_interior_design_chatbot()

    print("🏠 Interior Design Chatbot (Bosnian Mentor Mode) 🛋️")
    print("Type 'exit' or 'quit' to end the conversation.")

    # Conversation loop
    while True:
        try:
            user_input = input("Vi: ")
            
            # Exit condition
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Hvala što ste pričali sa mnom! Doviđenja. 👋")
                break

            # Generate response
            response = chatbot.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": "default_session"}}
            )

            print("Asistent:", response.content)

        except KeyboardInterrupt:
            print("\nRazgovor prekinut. Doviđenja! 👋")
            break
        except Exception as e:
            print(f"Greška: {e}")
            continue

if __name__ == "__main__":
    main()
