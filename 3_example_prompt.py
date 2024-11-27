from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

template = "You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?"

prompt = PromptTemplate.from_template(template)

print(prompt.format(company="ABC Startup", product="colorful socks"))

llm = OpenAI(temperature=0.9)
chain = prompt | llm

print(chain.invoke({"company": "ABC Startup", "product": "colorful socks"}))
