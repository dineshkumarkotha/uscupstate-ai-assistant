import streamlit as st
from rag import initialize_rag, get_answer

# Page configuration
st.set_page_config(
    page_title="USC Upstate Student Services AI Assistant",
    page_icon="🎓",
    layout="centered"
)

# Header
st.image("https://uscupstate.edu/wp-content/themes/uscupstate/images/logo.png", width=200)
st.title("USC Upstate Student Services AI Assistant")
st.markdown("*Powered by Azure OpenAI & RAG — Answers grounded in official USC Upstate policy documents*")
st.divider()

# Sidebar
with st.sidebar:
    st.header("About This Assistant")
    st.markdown("""
    This AI assistant answers questions using official USC Upstate policy documents including:
    - SOAR Student Guide 2025
    - Graduate Student Handbook
    
    **All answers are grounded in official documents with citations.**
    
    ⚠️ For critical decisions always verify with the relevant USC Upstate office.
    """)
    
    st.divider()
    st.header("Try Asking:")
    st.markdown("""
    - What is FERPA?
    - How do I register for classes?
    - What is the financial aid process?
    - What are the housing options?
    - What is the academic integrity policy?
    - How do I contact the IT help desk?
    """)
    
    st.divider()
    st.caption("Built by Dinesh Kotha | USC Upstate AI Developer Candidate")
    st.caption("Prototype → Production path: Azure OpenAI + Copilot Studio + Microsoft Fabric")

# Initialize RAG
@st.cache_resource
def load_rag():
    with st.spinner("Loading USC Upstate knowledge base..."):
        vector_store, chunks = initialize_rag()
    return vector_store, chunks

vector_store, chunks = load_rag()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm the USC Upstate Student Services AI Assistant. I can answer questions about university policies, financial aid, housing, academic requirements, and more. How can I help you today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about USC Upstate policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching USC Upstate documents..."):
            answer, citations = get_answer(prompt, vector_store, chunks)
        
        st.markdown(answer)
        
        # Show citations
        if citations:
            st.divider()
            st.caption("📚 Sources:")
            for citation in citations:
                st.caption(f"• {citation}")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
