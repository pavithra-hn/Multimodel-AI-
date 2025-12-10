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
            with st.spinner("🔍 Analyzing document structure (Text + Vision)..."):
                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.info("ℹ️ Note: Deep Vision analysis is enabled. Processing takes ~10-15s per page.")
                
                # Process
                processor = DocumentProcessor()
                documents = processor.process_pdf("temp.pdf")
                
                # Ingest documents into the vector store
                # (Collection reset is handled in rag_pipeline.py to avoid locking issues)
                    
                rag = MultiModalRAG()
                rag.ingest_documents(documents)
                
                # Clear cache so the new RAG instance picks up the new database
                st.cache_resource.clear()
                
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

@st.cache_resource
def get_rag_system():
    return MultiModalRAG()

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
                    # Use cached RAG system
                    rag = get_rag_system()
                    qa_chain = rag.get_qa_chain()
                    
                    # Get the answer from the chain (it returns a string directly)
                    answer = qa_chain.invoke(prompt)
                    
                    # Get source documents separately from the retriever
                    # Using the vector store from the cached rag instance
                    retriever = rag.vector_store.as_retriever(search_kwargs={"k": 5})
                    source_docs = retriever.invoke(prompt)
                    
                    st.markdown(answer)
                    
                    # Store answer first to ensure UI consistency
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Collect all candidates (and check for strict ID links in text)
                    candidates = []
                    all_images_data = [] 
                    
                    import re
                    # Regex to find IDs in text: e.g. [Detected Table ID: 52a1b3c4]
                    id_pattern = re.compile(r"ID:\s*([a-zA-Z0-9-]+)\]")
                    
                    # 1. Collect all IDs explicitly mentioned in the retrieved text chunks
                    found_ids_in_context = set()
                    for doc in source_docs:
                        matches = id_pattern.findall(doc.page_content)
                        found_ids_in_context.update(matches)
                    
                    unique_id_counter = 0
                    for doc in source_docs:
                        if "image_metadata" in doc.metadata:
                            try:
                                img_meta_list = json.loads(doc.metadata["image_metadata"])
                                for item in img_meta_list:
                                    path = item.get("path")
                                    if path and os.path.exists(path):
                                        # Check if this visual's ID was cited in the text
                                        # Note: Old docs might not have "id" field, handle gracefully
                                        visual_id = item.get("id", "")
                                        is_directly_linked = visual_id in found_ids_in_context
                                        
                                        desc_prefix = "[DIRECTLY LINKED] " if is_directly_linked else ""
                                        
                                        candidates.append({
                                            "id": unique_id_counter,
                                            "type": item.get("type", "figure"),
                                            "description": f"{desc_prefix}{item.get('description', '')[:300]}",
                                            "is_linked": is_directly_linked
                                        })
                                        
                                        all_images_data.append({
                                            "path": path,
                                            "page": doc.metadata.get('page', 'N/A'),
                                            "markdown": item.get("markdown", ""),
                                            "is_linked": is_directly_linked
                                        })
                                        unique_id_counter += 1
                            except json.JSONDecodeError:
                                pass
                    
                    # Select the visual matches
                    winner_idx_list = [] # Keep track of what we showed prominently
                    
                    if candidates:
                        # st.write(f"DEBUG: Found {len(candidates)} candidates.") # Uncomment for UI debug
                        
                        with st.spinner("🔍 Analyzing visual intent..."):
                            selection = rag.select_best_visual_match(prompt, candidates)
                        
                        if selection:
                            selected_indices = selection.get("selected_indices", [])
                            # Fallback for backward compatibility if model returns old format
                            if not selected_indices and selection.get("selected_index") is not None:
                                selected_indices = [selection["selected_index"]]
                                
                            intent = selection.get("intent", "specific")
                            reason = selection.get("reason", "")
                        else:
                            selected_indices = []
                            intent = "specific"
                            reason = "Selection failed."
                        
                        if selected_indices:
                            winner_idx_list = selected_indices
                            
                            # Determine Mode: Gallery (All) vs Focus (Specific)
                            if intent == "all":
                                st.markdown(f"### 🖼️ Visual Gallery: {selection.get('visual_type_requested', 'All').capitalize()}s")
                                if reason:
                                    st.success(f"**Showing Matches:** {reason}")
                                
                                cols = st.columns(2) # Grid layout
                                for i, idx in enumerate(selected_indices):
                                    if 0 <= idx < len(all_images_data):
                                        item = all_images_data[idx]
                                        with cols[i % 2]:
                                            st.image(item["path"], caption=f"Page {item['page']}", use_container_width=True)
                                            if item.get("markdown"):
                                                with st.expander("📄 Table Data"):
                                                    st.markdown(item["markdown"])
                                                    
                            else:
                                # Specific Mode - Take the first one (or iterate if multiple specific matches found)
                                for idx in selected_indices:
                                    if 0 <= idx < len(all_images_data):
                                        winner = all_images_data[idx]
                                        
                                        st.markdown(f"### 🎯 Most Relevant Visual")
                                        if reason:
                                            st.info(f"**Selected:** {reason}")
                                        st.image(winner["path"], caption=f"Selected Visual from Page {winner['page']}", use_container_width=True)
                                        
                                        if winner.get("markdown"):
                                            st.markdown("#### 📄 Extracted Table Data")
                                            st.markdown(winner["markdown"])
                                        
                                        # Limit to 1 in specific mode to avoid clutter unless strictly needed
                                        break 
                        else:
                            st.warning("ℹ️ No specific chart or table matched your query in the retrieved context.")
                    else:
                        st.info("No visual elements found in the context.")

                    # Show OTHER visuals in a gallery if they exist (so users aren't blind)
                    # Only show this fallback if we were in "Specific" mode or if we missed showing some items
                    if len(all_images_data) > 0 and intent != "all":
                        with st.expander("🖼️ All Detected Visuals in Context", expanded=False):
                            cols = st.columns(3)
                            shown_count = 0
                            for i, img_data in enumerate(all_images_data):
                                if i in winner_idx_list:
                                    continue # Skip what we just showed
                                
                                with cols[shown_count % 3]:
                                    st.image(img_data["path"], caption=f"Page {img_data['page']}", use_container_width=True)
                                    if img_data.get("markdown"):
                                        with st.popover("View Table"):
                                            st.markdown(img_data["markdown"])
                                shown_count += 1
                    
                    with st.expander("📚 View Source Context (Text Only)"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                            st.caption(doc.page_content[:500] + "...") 
                            
                            if "image_metadata" in doc.metadata:
                                st.caption("*(Contains visual elements - see 'All Detected Visuals' above)*") 
                            
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
