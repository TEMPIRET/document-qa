import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

loader=PyPDFLoader("first.pdf")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40, add_start_index=True)
all_splits=text_splitter.split_documents(documents)


huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

## VectorStore Creation
vectorstore=FAISS.from_documents(all_splits[:120],huggingface_embeddings)

## Query using Similarity Search
query="WHAT IS PERIODIC TABLE?"
relevant_docments=vectorstore.similarity_search(query)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})

# import os
# os.environ["HUGGINGFACE_API_TOKEN"] = st.secrets["well"]

import os
st.write(
    os.environ["HUGGINGFACE_API_TOKEN"] == st.secrets["well"],
)

from langchain_community.llms import HuggingFaceHub

hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1,"max_length":500}

)

prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
 """

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

retrievalQA=RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)

query=st.text_area('Enter your question: ')

# Call the QA chain with our query.

if query:
    result = retrievalQA.invoke({"query": query})
    st.write(result['result'])