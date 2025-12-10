import os
import sys
from dotenv import load_dotenv
from rag_pipeline import MultiModalRAG
from document_processor import DocumentProcessor

# Load env vars
load_dotenv()

def test_retrieval():
    print("----- Starting RAG Verification Test -----")
    
    # Check if DB exists, if not ingest
    if not os.path.exists("./chroma_db"):
        print("Chroma DB not found. Processing PDF...")
        pdf_path = "multi-modal_rag_qa_assignment.pdf"
        if os.path.exists(pdf_path):
            processor = DocumentProcessor()
            docs = processor.process_pdf(pdf_path)
            rag = MultiModalRAG()
            rag.ingest_documents(docs)
            print("Ingestion complete.")
        else:
            print(f"Error: PDF {pdf_path} not found.")
            return

    rag = MultiModalRAG()
    qa_chain = rag.get_qa_chain()
    
    test_queries = [
        ("Table", "Show me the table of common indicators for Qatar."),
        ("Chart", "Show me a chart about GDP or economic growth."),
        ("Figure", "Show me a figure or diagram describing the architecture.")
    ]
    
    for type_label, query in test_queries:
        print(f"\n[TESTING {type_label.upper()} RETRIEVAL]")
        print(f"Query: {query}")
        
        # 1. Get Answer
        answer = qa_chain.invoke(query)
        print(f"Answer Preview: {answer[:150]}...")
        
        # 2. Check Retrieval & Visual Selection
        retriever = rag.vector_store.as_retriever(search_kwargs={"k": 5})
        source_docs = retriever.invoke(query)
        
        # Extract Candidates like App.py does
        candidates = []
        import json
        import re
        id_pattern = re.compile(r"ID:\s*([a-zA-Z0-9-]+)\]")
        found_ids = set()
        
        for doc in source_docs:
            found_ids.update(id_pattern.findall(doc.page_content))
            
            if "image_metadata" in doc.metadata:
                try:
                    meta = json.loads(doc.metadata["image_metadata"])
                    for item in meta:
                        candidates.append(item)
                except:
                    pass
        
        print(f"Found {len(candidates)} visual candidates in context.")
        print(f"Directly Linked IDs found in text: {found_ids}")
        
        # 3. Simulate Selection
        if candidates:
            selection = rag.select_best_visual_match(query, candidates)
            print(f"Selection Result: {json.dumps(selection, indent=2)}")
            
            if selection and selection.get("selected_indices"):
                print("✅ SUCCESS: Visual element selected!")
            elif selection and selection.get("selected_index") is not None:
                print("✅ SUCCESS: Visual element selected (legacy format)!")
            else:
                print("⚠️ WARNING: No visual selected.")
        else:
            print("❌ FAILURE: No visuals found in retrieved context.")

if __name__ == "__main__":
    test_retrieval()
