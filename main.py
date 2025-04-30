from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever

import os

#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

# Load the text file
loader = TextLoader("horoscope.txt")
docs = loader.load()

# Split the text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Use a lightweight and free model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Store in vector store
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# Load QA chain with Mistral via OpenRouter
llm = ChatOpenAI(
    model_name="meta-llama/llama-3-70b-instruct",  # or any OpenRouter-compatible chat model
    temperature=0.7,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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"
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query the bot
while True:
    query = input("Ask me anything: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.invoke({"query": query})
    print(result['result'])
