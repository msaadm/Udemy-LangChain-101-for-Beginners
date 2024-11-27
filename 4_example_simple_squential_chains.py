from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

template = "What is a good name for a company that makes {product}?"

prompt = PromptTemplate.from_template(template)

print(prompt.format(product="socks"))

llm = OpenAI(temperature=0)
first_chain = prompt | llm

# print(first_chain.invoke({"product": "socks"}))

template2 = "Write a catch phrase for the following company: {company_name}"

prompt2 = PromptTemplate.from_template(template2)

second_chain = prompt2 | llm

overall_chain = (first_chain | second_chain).with_config({"verbose": True})

catchphrase = overall_chain.invoke({"product": "colorful socks"})
print(catchphrase)
