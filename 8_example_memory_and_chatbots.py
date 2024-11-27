from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)

conversation = ConversationChain(llm=llm, verbose=True)

# conversation.predict(input="Hi there!")
# conversation.predict(input="Can we talk about the weather?")
# print(conversation.predict(input="It's a beautiful day!"))

print("Welcome to your AI Chatbot! What's on yout mind?")
for _ in range(3):
    human_input = input("You: ")
    ai_response = conversation.predict(input=human_input)
    print(f"AI: {ai_response}")
