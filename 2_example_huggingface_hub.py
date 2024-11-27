import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7, "max_length": 64}
)

# Generate text

prompt = "What are good fitness tips?"

print(llm.invoke(prompt))
