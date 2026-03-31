# USC Upstate Student Services AI Assistant

An AI-powered student services chatbot built specifically for the 
University of South Carolina Upstate. This assistant answers questions 
about university policies, financial aid, housing, academic requirements, 
and more — grounded exclusively in official USC Upstate policy documents.

## Live Demo
🔗 https://uscupstate-ai-assistant-zm3kkds6nnxirqyw73bjwn.streamlit.app/

## Features
- RAG pipeline with hybrid retrieval (BM25 + Vector Search)
- Citation enforcement on every answer
- Grounded in official USC Upstate public documents
- FERPA-compliant design — no PII stored
- Clean, accessible chat interface

## Knowledge Base
- SOAR Student Guide 2025
- USC Upstate Graduate Student Handbook

## Production Architecture
This prototype demonstrates the core RAG pipeline. 
In production this would scale to:
- Microsoft Copilot Studio (enterprise deployment)
- Azure OpenAI (FERPA-compliant AI layer)
- Microsoft Fabric (unified data foundation)
- Dataverse (interaction logging & audit trail)
- Azure AI Foundry (model monitoring & governance)

## Tech Stack
- Python
- OpenAI GPT-3.5-turbo
- LangChain
- FAISS vector store
- Streamlit
- Hybrid retrieval: BM25 + semantic search

## Governance & Compliance
- Answers grounded only in official documents
- Citations provided for every response
- No hallucination by design
- FERPA considerations built into architecture
- Responsible AI: system refuses to answer outside document scope

## Built By
Dinesh Kotha | AI Developer Candidate
MS Data Science, University at Buffalo
linkedin.com/in/dineshkumar1716
