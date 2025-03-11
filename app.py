import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain 
import time

groq_api_key = os.getenv('GROQ_API_KEY')
#genai.configure(api_key = api_key)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
#prompt1 = st.text_input("What do you want to ask from the documents?")
pdf_directory = "./us_census"
st.set_page_config(page_title="Q&A Using Groq", page_icon="📄")
st.header("📖 Q&A Using Groq")

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(pdf_directory)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_docs ,st.session_state.embeddings )


def get_conversational_chain(prompt1):
    """Processes user question using vector store & LLM."""
    if "vectorstore" not in st.session_state:
        st.error("⚠️ Vector store not found! Please click the button to create it first.")
        return

    # ✅ Use Correct Groq Model
    #prompt1 = st.text_input("What do you want to ask from the documents?")
    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible from the provided context.
    Make sure to provide the correct answer.
    If you don't know the answer, just say "I don't know", but do not provide the wrong answer.

    <context>
    {context}
    <context>
    Question: {input}
    Answer:""")


    llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=groq_api_key)

    #prompt1 = st.text_input("📝 Ask a question from the documents:")

    if prompt1:
        # ✅ Create Retrieval Chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectorstore.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        # ✅ Measure Processing Time
        start_time = time.process_time()
        response = retriever_chain.invoke({"input":prompt1})
        st.write(response["answer"])  # ✅ FIXED: Pass only string input
        processing_time = time.process_time() - start_time

        # ✅ Display AI Response
        st.subheader("💡 AI Response:")
        st.write(response['answer'])
        st.write(f"⏳ Response Time: {processing_time:.2f} seconds")

        # ✅ Show Relevant Documents (Expandable Section)
        with st.expander("📄 Document Similarity Search Results"):
            for i, doc in enumerate(response['context']):
                st.write(f"🔹 **Document {i+1}**")
                st.write(doc.page_content)
                st.write("—" * 50)



def main():
    
    
    prompt1 = st.text_input("📝 Ask a question from the documents:")
    if st.button("CReating the Vector Store:"):
        vector_embeddings()
        st.write("vector store db is ready")
    if prompt1:
        get_conversational_chain(prompt1)

    

if __name__ == "__main__":
    main()
   


