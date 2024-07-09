# necessary Imports
import streamlit as st
# Set the page configuration
st.set_page_config(
    page_title="Wiki-App ",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)
from docx import Document
from PyPDF2 import PdfReader
from pptx import Presentation
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts  import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

pdf_file = open("Required_Text.pdf",'rb')

# extracting pdf data
pdf_text = ""
pdf_reader = PdfReader(pdf_file)
for page in pdf_reader.pages:
    pdf_text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200, # This is helpul to handle the data loss while chunking.
        length_function = len,
        separators=['\n', '\n\n', ' ', '']
    )
chunks = text_splitter.split_text(text = pdf_text)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_texts(chunks, embedding = embeddings)
# creating retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                not contained in the context, just return "answer not available in context"s\n\n
                Context: \n {context}?\n
                Question: \n {question} \n
                Answer:"""

prompt = PromptTemplate.from_template(template=prompt_template)
# function to create a single string of relevant documents given by Faiss.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import os
# RAG Chain

def generate_answer(question):
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key='sRmFY97EVTJa7VaaaQha5oH7lScl1rxTZv8x6KrV')

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


import cohere
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from pydantic import BaseModel
from typing import List

# Set up Cohere API key
cohere_api_key = 'sRmFY97EVTJa7VaaaQha5oH7lScl1rxTZv8x6KrV'
co = cohere.Client(cohere_api_key)

# Initialize Wikipedia Retriever
getti = WikipediaRetriever(lang='en', doc_content_chars_max=500000, top_k_results=2, sleep_time=0, max_retry=12)

class Document(BaseModel):
    page_content: str
    metadata: dict

def combine_documents_fn(docs: List[Document]) -> str:
    document_context = ""
    for doc in docs:
        document_context += doc.page_content
    return document_context

def retrieve_docs(question: str) -> List[Document]:
    documents = getti.get_relevant_documents(question)
    return documents

def generate_aer(context, question):
    prompt_template = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt_template,
        max_tokens=150
    )
    return response.generations[0].text.strip()

def ask_question(question: str):
    try:
        docs = retrieve_docs(question)
        context = combine_documents_fn(docs)
        answer = generate_aer(context, question)
        return answer
    except Exception as e:
        return str(e)


def main():
  st.header('Thapar GPT')
  ques= st.text_input("Ask", key="question")
  ans = generate_answer(ques)
  if st.button("Ask", key="process_button"):
      with st.spinner("Kaha se laate ho itne ache Question..."):

        if ans==" answer not available in context":
          w=ask_question(ques)
          st.write(w)
        else:
          st.write(ans)
if __name__ == "__main__":
    main()
