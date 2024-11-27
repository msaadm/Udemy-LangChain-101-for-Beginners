from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

loader = TextLoader("./state-of-the-union-23.txt")
documents = loader.load()
# print(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    texts, embeddings, collection_name="state-of-the-union"
)

llm = OpenAI(temperature=0)
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

print(retrieval_qa.invoke({"query": "What did Biden talk about Ohio?"}))
