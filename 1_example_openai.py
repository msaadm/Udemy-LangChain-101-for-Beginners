from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAI

# temperature is the randomness of the output between 0 and 1 (0 is deterministic, 1 is random)
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

prompt = "What whould be 5 good name for a company that makes colorful socks?"

print(llm.invoke(prompt))
