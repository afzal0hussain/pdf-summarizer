
'''

new task convert the pdf into markdown also check if summarisation is faster using markdown 
are there llms  that can read markdown table or table inside pdf efficiently  
read about text generation 

'''


from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import pprint
import streamlit as st
import tempfile
from langchain.docstore.document import Document 

def loader(pdf):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(pdf.read())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    return pages


def splitter(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)
    return docs


def embedding_and_vectorStore(docs):
    embeddings_model = CohereEmbeddings(
        cohere_api_key="RjBWLb5m1anLP2tIWKH1mgNmR2p8ppcxBKfFBDtI",
        model="embed-english-v3.0"  # Use correct embedding model
    )
    persist_directory = "./chroma"
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings_model, persist_directory=persist_directory
    )
    return vectordb


def prompt():
    prompt_template = (
        """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
    )
    prompt_i = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt_i


def llm():
    llm_i = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0, "max_length": 180},
        huggingfacehub_api_token="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",
    )
    return llm_i


def summarizer(llm, vectordb, prompt):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    response = qa_chain("please summarize this book")

    pp = pprint.PrettyPrinter(indent=0)

    pp.pprint(response["result"])
    return response["result"]


st.title("Summary generator")

pdf_files = st.file_uploader(
    "Upload PDF please", type="pdf", accept_multiple_files=False
)

if pdf_files:

    if st.button("Generate Summary"):
        st.write("Summaries:")
        pages = loader(pdf_files)
        docs = splitter(pages)
        st.cache_data.clear()
        vectordb = embedding_and_vectorStore(docs)
        prompt_i = prompt()
        llm_i = llm()
        summaries = summarizer(llm_i, vectordb, prompt_i)
        st.write(summaries)