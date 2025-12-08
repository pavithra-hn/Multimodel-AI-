
print("Start Debugging Imports")
try:
    print("Importing os...")
    import os
    print("Importing time...")
    import time
    print("Importing chromadb...")
    import chromadb
    print("Importing GoogleGenerativeAIEmbeddings...")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    print("Importing Chroma...")
    from langchain_chroma import Chroma
    print("Importing RecursiveCharacterTextSplitter...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("All imports successful.")
except Exception as e:
    print(f"Failed at import: {e}")
