import streamlit as st
import os
import json
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from rag_pipeline import MultiModalRAG
import shutil

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide", page_icon="🤖")

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        color: #58a6ff;
        margin-bottom: 2rem;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #2ea043;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6edf3;
        font-weight: 600;
    }
    
    /* Chat Message Styling */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #1f6feb22;
        border-left: 4px solid #1f6feb;
    }
    
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #23863622;
        border-left: 4px solid #238636;
    }
    
    /* Success/Error/Warning Messages */
    .stAlert {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #161b22;
        color: #c9d1d9;
        border-radius: 6px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Application Header
with st.container():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 🧠")
    with col2:
        st.title("Synapse: Multi-Modal RAG System")
        st.markdown("*Advanced Document Analysis with Text, Tables, and Images (Powered by OpenAI)*")

st.divider()

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.markdown("### Authentication")
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.text_input("Enter OpenAI API Key", type="password", help="Your OpenAI ChatGPT API Key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.rerun() # Rerun to remove input field if desired, or just leave it
    
    st.markdown("---")
    
    st.markdown("### Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    st.markdown("---")
    
    process_btn = st.button("🚀 Process Document")
    
    if process_btn:
        if uploaded_file and os.getenv("OPENAI_API_KEY"):
            with st.spinner("🔍 Analyzing document structure..."):
                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process
                processor = DocumentProcessor()
                documents = processor.process_pdf("temp.pdf")
                
                # Ingest documents into the vector store
                # (Collection reset is handled in rag_pipeline.py to avoid locking issues)
                    
                rag = MultiModalRAG()
                rag.ingest_documents(documents)
                
                st.sidebar.success(f"✅ Indexed {len(documents)} pages!")
        elif not os.getenv("OPENAI_API_KEY"):
            st.sidebar.error("⚠️ API Key missing")
        else:
            st.sidebar.warning("⚠️ No file uploaded")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your document..."):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please configure your API Key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Synapse is thinking..."):
                try:
                    rag = MultiModalRAG()
                    qa_chain = rag.get_qa_chain()
                    
                    # Get the answer from the chain (it returns a string directly)
                    answer = qa_chain.invoke(prompt)
                    
                    # Get source documents separately from the retriever
                    retriever = rag.vector_store.as_retriever(search_kwargs={"k": 5})
                    source_docs = retriever.invoke(prompt)
                    
                    st.markdown(answer)
                    
                    # Store answer first to ensure UI consistency
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Collect unique images to display
                    unique_images = set()
                    images_to_display = []
                    
                    # Logic to decide if we should show images based on user query
                    # Only show images if the user asks for them (e.g., "chart", "image", "graph")
                    image_keywords = ['image', 'chart', 'graph', 'figure', 'visual', 'picture', 'plot', 'diagram', 'photo', 'drawing', 'illustration']
                    show_images = any(keyword in prompt.lower() for keyword in image_keywords)

                    if show_images:
                        for doc in source_docs:
                            if "image_paths" in doc.metadata:
                                try:
                                    image_paths = json.loads(doc.metadata["image_paths"])
                                    for img_path in image_paths:
                                        if os.path.exists(img_path) and img_path not in unique_images:
                                            unique_images.add(img_path)
                                            images_to_display.append((img_path, doc.metadata.get('page', 'N/A')))
                                except json.JSONDecodeError:
                                    pass
                    
                    # Display images directly in the chat if relevant
                    if images_to_display:
                        st.markdown("### Relevant Images:")
                        cols = st.columns(min(len(images_to_display), 3))
                        for idx, (img_path, page_num) in enumerate(images_to_display):
                            with cols[idx % 3]:
                                st.image(img_path, caption=f"Page {page_num}", use_container_width=True)
                    
                    with st.expander("📚 View Source Context"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                            st.caption(doc.page_content[:500] + "...") 
                            
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
