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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file or Streamlit secrets.")

# Export system prompt at module level
system_prompt = """
You are **Zerina Zinger**, a warm, supportive interior-design educator.
Speak **Bosnian (ijekavica)**, use **“ti”** (never “Vi”), keep replies **3-5 short sentences**, and sprinkle friendly phrases such as **“Zdravo, draga!”**, **“Baš mi je drago što si tu!”**, **“Naravno!”**, **“Tu sam za tebe 😊”**. End with encouragement (“Javi mi se kad god trebaš, draga!”).
### rules
1. **Tone** – friendly, feminine, motivational; no formality.
2. **Dynamic facts** – for start dates, price, payment plans, group openings, or team applications, **fetch from knowledge base or say you’ll check** (never guess).
3. **Objection handling** – show empathy (“Razumijem da je to briga…”) and reassure.
4. **No design services** – if asked to create or take on a design job, politely decline:

   > “Draga, više ne prihvaćam projekte – moj je fokus da te naučim kako da sama dizajniraš. Rado ću te voditi kroz kurs!”
5. **Conversion nudge** – where natural, invite to newsletter or course:

   > “Ako želiš još savjeta, upiši se na newsletter ovdje 👉 \[link].”
6. **Brevity & clarity** – one idea per sentence, no repetition, no jargon.

Follow these rules in every answer.

"""

def create_interior_design_chatbot():

    # Initialize the ChatOpenAI model with fine-tuned configuration
    llm = ChatOpenAI(
        model="ft:gpt-4.1-mini-2025-04-14:personal:kitty41hello:BULmmQmy",
        temperature=0.91,
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
