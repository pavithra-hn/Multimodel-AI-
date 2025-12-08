
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.getcwd())
load_dotenv()

from rag_pipeline import MultiModalRAG

try:
    print("Instantiating MultiModalRAG...")
    rag = MultiModalRAG()
    
    print("Getting QA Chain...")
    chain = rag.get_qa_chain()
    
    print("Invoking Chain with test question...")
    # This requires internet and API key
    try:
        response = chain.invoke("What is the main topic of the document?")
        print(f"Chain response: {response}")
    except Exception as e:
        print(f"Chain invocation failed: {e}")
        # Print full traceback
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Setup failed: {e}")
    import traceback
    traceback.print_exc()
