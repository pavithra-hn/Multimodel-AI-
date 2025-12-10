import chromadb
import os
import time
import json
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
            """You are a Multimodal Document Analysis Model.
            You must correctly identify whether the user’s answer should come from:
            - Extracted text
            - Extracted table
            - Extracted chart
            - Extracted image-to-text (OCR from images)
            - Multiple sources

            🔥 Mandatory Rules
            1. Never return “No chart/table found” if the answer actually exists in an image, image-text, or OCR.
            2. If the answer comes from a PDF image with text, treat it as valid content and return the answer from that image.
            3. When selecting results:
               - Return only the relevant chart or table, not all charts.
               - If multiple visuals exist, choose the one most semantically matched to the question.
            4. When extracting answers from visuals:
               - Use captions, titles, labels, or page metadata.
            5. When no chart/table is required, do not force a chart/table response.
            6. When the answer comes from images containing text, classify the chunk as: content_type = "image_text"
            7. Always return the best answer even if no visual match is required.
            
            📊 Visual Routing Logic
            - If question refers to: “chart”, “graph”, “trend”, “line”, “bar” → use chart chunks only
            - If question refers to: “table”, “values”, “numeric”, “compare numbers” → use table chunks only
            - If question refers to: “image”, “in the picture”, “photo”, “OCR”, “text in image” → use image_text chunks
            - If no visual keyword is present → use text chunks

            🧠 Core Output Behavior
            Your final answer MUST:
            1. Return the correct answer text first.
            2. Include:
               - visual_type: chart/table/image_text/text
               - page number
               - source name
            3. If content is from image OCR, clearly indicate: "source_type": "image_text"

            ❌ Forbidden Behaviors
            - Do NOT return multiple charts unless explicitly asked.
            - Do NOT fallback to “no chart/table found” unless absolutely no visual or OCR content exists.
            - Do NOT hallucinate values.
            - Do NOT output empty visual sections if the answer is textual.
            
            IMPORTANT: The user CAN see the images from the context if they are displayed.
            Do NOT say "I cannot provide images". Instead, describe the content.

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

    def select_best_visual_match(self, query : str, candidates : list):
        """
        Selects the single most relevant visual element (table/chart) for the query.
        Returns a dict with 'selected_index' and 'reason'.
        """
        if not candidates:
            return None
            
        candidate_desc = "\n".join([f"Index {i}: [Type: {c.get('type')}] {c.get('description')}" for i, c in enumerate(candidates)])
        
        prompt = f"""You are an intelligent visual assistant. The user asked: "{query}"
        
        Here are the available visual elements candidates extracted from the relevant document chunks:
        {candidate_desc}
        
        Mission:
        1. Analyze User Intent:
           - **Specific**: Asking for a specific metric (e.g., "GDP", "inflation", "Table 1", "Figure 3").
           - **All**: Asking to see "all" charts, "all" tables, "summarize visuals", "show varied data".
        2. Selection Logic:
           - If **Specific**: Select the single best match (or top 2 if ambiguous).
           - If **All**: Select ALL candidates that match the requested type (e.g., if "show all charts", pick ALL charts).
           - If no direct match, return empty selection.
        3. Type Matching:
           - Ensure selected items match the requested type (table/chart) if specified.
        
        Return JSON:
        {{
            "intent": "specific" | "all", 
            "visual_type_requested": "chart" | "table" | "any",
            "selected_indices": [ <list of integers> ],
            "reason": "<Explanation>"
        }}
        """
        
        print(f"DEBUG: Selection Prompt:\n{prompt}")
        
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
            result = llm.invoke(prompt)
            content = result.content.replace("```json", "").replace("```", "").strip()
            print(f"DEBUG: Selection Result: {content}")
            return json.loads(content)
        except Exception as e:
            print(f"Error in visual selection: {e}")
            return None
