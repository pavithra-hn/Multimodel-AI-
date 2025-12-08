import chromadb
import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class MultiModalRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        self.persist_directory = persist_directory
        self.vector_store = None
        
    def ingest_documents(self, documents):
        """
        Chunks documents and stores them in ChromaDB with rate limit handling.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Reset collection to avoid stale data (safer than deleting directory)
        try:
            client.delete_collection("rag_collection")
        except Exception:
            pass  # Collection doesn't exist or other error, which is fine for reset
            
        # Initialize the collection
        self.vector_store = Chroma(
            client=client,
            collection_name="rag_collection",
            embedding_function=self.embeddings
        )
        
        # Batch processing with retries
        batch_size = 50 # OpenAI handles larger batches better usually, but keeping it safe
        total_chunks = len(chunks)
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            retry_count = 0
            max_retries = 5
            
            while retry_count < max_retries:
                try:
                    self.vector_store.add_documents(documents=batch)
                    print(f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            if retry_count == max_retries:
                print(f"Failed to process batch causing error after {max_retries} retries.")
            
            # Rate limiting delay
            time.sleep(1)
        
    def get_qa_chain(self):
        """
        Returns a QA chain for answering questions.
        """
        if not self.vector_store:
            # Try loading existing DB
            client = chromadb.PersistentClient(path=self.persist_directory)
            self.vector_store = Chroma(client=client, collection_name="rag_collection", embedding_function=self.embeddings)
            
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use the following pieces of context to answer the question at the end. 
            The context may contain descriptions of images, tables, or charts. 
            IMPORTANT: The user CAN see the images from the context. The images are displayed directly to the user along with your response.
            Therefore, you should refer to the images in your answer (e.g., "As shown in the image on page 3...", "The chart displays...").
            Do NOT say "I cannot provide images" or "I cannot see the image". Instead, describe the content based on the provided description and explicitly state that the image is shown below/above.
            Always cite the page number.

            Context:
            {context}

            Question: {question}
            Answer:"""
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build the chain using LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
