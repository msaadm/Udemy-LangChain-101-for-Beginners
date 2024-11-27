from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    HumanMessage,
    AIMessage,
    messages_from_dict,
    messages_to_dict,
)

history = ChatMessageHistory()
history.add_messages([HumanMessage(content="Hello! let's talk about girrafes")])
history.add_messages([AIMessage(content="Hi! I am down to talk about girrafes")])

# print(history.messages)

llm = OpenAI(temperature=0)

history = ChatMessageHistory(messages=history.messages)
buffer = ConversationBufferMemory(chat_memory=history, return_messages=True)

# Replace the ConversationChain with the newer RunnableWithMessageHistory approach
prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="history"), ("human", "{input}")]
)

chain = prompt | llm


# Create a function to get session history
def get_session_history(session_id: str):
    return history


chain_with_history = RunnableWithMessageHistory(
    chain,
    memory=buffer,
    input_messages_key="input",
    history_messages_key="history",
    get_session_history=get_session_history,
)

# Use invoke instead of predict
response = chain_with_history.invoke(
    {"input": "What are they?"}, config={"configurable": {"session_id": "my_session"}}
)
print(response)
