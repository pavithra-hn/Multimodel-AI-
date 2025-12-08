from document_processor import DocumentProcessor
import os

def verify():
    if not os.path.exists("multi-modal_rag_qa_assignment.pdf"):
        print("PDF not found.")
        return

    processor = DocumentProcessor()
    print("Processing PDF...")
    docs = processor.process_pdf("multi-modal_rag_qa_assignment.pdf")
    
    table_found = False
    for i, doc in enumerate(docs):
        print(f"Page {i+1} content length: {len(doc.page_content)}")
        if "|" in doc.page_content and "---" in doc.page_content:
            print(f"Potential markdown table found on page {i+1}")
            table_found = True
            # Print a snippet
            print(doc.page_content[:500])
    
    if table_found:
        print("SUCCESS: Tables detected in extracted content.")
    else:
        print("WARNING: No tables detected (or markdown format not recognized).")

if __name__ == "__main__":
    verify()
