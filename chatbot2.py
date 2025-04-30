from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Define your custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial advisor. The person chatting with you is the individual described below.

[User Profile]
{context}

[User's Question]
{question}

Answer as a helpful and trustworthy financial advisor. Give clear and actionable advice.
"""
)

#-----------------------------------------------------------------------------#

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st

# Sample simulated user data (fetched from database after login)
user_data_text = """
Name: Rahul Mehra  
Age: 32  
Employment Status: Full-time Software Engineer  
Annual Income: ₹15,00,000  
Monthly Expenses: ₹70,000  
Rent: ₹25,000  
Loan EMI: ₹15,000  
Savings: ₹4,50,000  
Investments:  
  - Mutual Funds: ₹2,00,000  
  - Stocks: ₹1,50,000  
  - Fixed Deposits: ₹1,00,000  
Insurance:  
  - Term Life Insurance: ₹1 Cr cover, ₹10,000 yearly premium  
  - Health Insurance: ₹5 Lakh cover, ₹12,000 yearly premium  
Debts:  
  - Home Loan: ₹30,00,000 outstanding at 8.5% interest, 15 years remaining  
  - Credit Card: ₹50,000 outstanding, 36% interest rate  
Credit Score: 735  

Financial Goals:  
  - Buy a car in 1 year (Budget: ₹8,00,000)  
  - Build an emergency fund of ₹3,00,000  
  - Save ₹25,00,000 for child’s education in 12 years  
  - Retire at 60 with a corpus of ₹3 Cr  

Risk Appetite: Moderate  
Preferred Investment Type: Mutual Funds and SIPs  
Tax Saving Instruments Used: ELSS, PPF  
"""

# Step 1: Turn this into a Document
docs = [Document(page_content=user_data_text)]

# Step 2: Chunk it
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Step 3: Embeddings and Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# Step 4: QA Chain
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct:free",
    temperature=0.7,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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"
)

advisor_chain = custom_prompt | llm

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 5: Streamlit Chat UI
st.set_page_config(page_title="Personal Finance Assistant",layout="wide")
st.title("💸 Personal Finance Advisor")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about your finances...")
if user_input:
    response = advisor_chain.invoke({"context": user_data_text, "question": user_input},config={"stop": None})
    print(response)
    st.session_state.chat_history.append(("🧑", user_input))
    st.session_state.chat_history.append(("🤖", response))

for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)
