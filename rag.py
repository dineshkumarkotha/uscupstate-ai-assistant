import os
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_pdfs(docs_folder="docs"):
    documents = []
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_folder, filename)
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1
                        }
                    ))
    print(f"Loaded {len(documents)} pages from {docs_folder}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    print("Building vector store...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store built and saved")
    return vector_store

def load_vector_store():
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def hybrid_search(query, vector_store, chunks, k=5):
    vector_results = vector_store.similarity_search(query, k=k)
    
    tokenized_corpus = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
    bm25_results = [chunks[i] for i in top_bm25_indices]
    
    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    
    return combined[:k]

def get_answer(query, vector_store, chunks):
    relevant_chunks = hybrid_search(query, vector_store, chunks)
    
    context = ""
    citations = []
    for i, chunk in enumerate(relevant_chunks):
        context += f"\n[Source {i+1}: {chunk.metadata['source']}, Page {chunk.metadata['page']}]\n"
        context += chunk.page_content + "\n"
        citations.append(f"{chunk.metadata['source']} (Page {chunk.metadata['page']})")
    
    prompt = f"""You are the USC Upstate Student Services AI Assistant.

Answer the following question using ONLY the provided context from official USC Upstate policy documents.
If the answer is not in the context, say "I don't have that specific information in my current knowledge base. Please contact the relevant USC Upstate office directly."
Always be helpful, accurate, and professional.
Always mention which document your answer comes from.

Context:
{context}

Question: {query}

Answer:"""
    
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )
    
    response = llm.invoke(prompt)
    return response.content, list(set(citations))

def initialize_rag():
    if os.path.exists("faiss_index"):
        print("Loading existing vector store...")
        vector_store = load_vector_store()
        documents = load_pdfs()
        chunks = chunk_documents(documents)
    else:
        print("Building new vector store...")
        documents = load_pdfs()
        chunks = chunk_documents(documents)
        vector_store = build_vector_store(chunks)
    return vector_store, chunks
