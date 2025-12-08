import fitz

def check_tables():
    try:
        doc = fitz.open("multi-modal_rag_qa_assignment.pdf")
        page = doc[0]
        tabs = page.find_tables()
        print(f"Type of tabs: {type(tabs)}")
        
        # Try iterating
        count = 0
        for i, tab in enumerate(tabs):
             print(f"Table {i} found.")
             print(tab.to_markdown())
             count += 1
        
        print(f"Total tables found by iteration: {count}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tables()
