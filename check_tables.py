import fitz
import sys

def check_tables():
    try:
        doc = fitz.open("multi-modal_rag_qa_assignment.pdf")
        page = doc[0]
        if hasattr(page, "find_tables"):
            print("find_tables is available")
            tables = page.find_tables()
            print(f"Found {len(tables)} tables on page 0")
            if tables:
                print("First table markdown:")
                print(tables[0].to_markdown())
        else:
            print("find_tables is NOT available")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tables()
