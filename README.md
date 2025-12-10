# Synapse: Intelligent Multi-Modal RAG System 🧠

Synapse is a text-centric and **Vision-First** Document QA system. Unlike traditional RAG pipelines that treat PDFs as plain text, Synapse uses **GPT-4o's Vision capabilities** to "see" the document, accurately detecting, cropping, and understanding visual elements like **Tables** and **Charts**.

## 🚀 Key Features

### 1. Vision-Powered Layout Analysis
Instead of relying on flaky PDF parsing libraries to find tables, Synapse renders PDF pages as high-resolution images and uses **GPT-4o Vision** to intelligently detect:
- **Tables**: Financial statements, data grids, and lists.
- **Charts**: Bar graphs, line charts, pie charts.
- **Figures**: Diagrams and technical illustrations.

### 2. Smart Visual Semantic Matching
The system understands the *intent* of your query:
- **Specific Requests**: "Show me the inflation chart" -> Returns **only** the single most relevant chart.
- **Broad Requests**: "Show me all the tables" -> Returns **all** tables available in the context.
- **Data-First**: "What is the GDP?" -> Returns the exact text answer *plus* the relevant table used to derive it.

### 3. High-Performance Ingestion
- **Parallel Processing**: Uses multi-threading to process PDF pages concurrently, significantly reducing ingestion time.
- **Rate Limit Handling**: Built-in exponential backoff to handle OpenAI API rate limits gracefully.

### 4. Data Reconstruction
- **Table-to-Markdown**: Automatically extracts raw data from detected table images and converts it into clean Markdown tables for the LLM to analyze.

---

## 🛠️ Architecture

- **`app.py`**: Streamlit frontend for file upload and chat.
- **`document_processor.py`**: 
  - Handles PDF rendering (PyMuPDF).
  - Orchestrates **Parallel Processing**.
  - Uses GPT-4o for **Vision-based Layout Detection** and **Crop Analysis**.
- **`rag_pipeline.py`**: 
  - Manages ChromaDB vector store.
  - Implements **Smart Visual Selection** logic.
  - Handles Context-Aware Retrieval.
- **`extracted_images/`**: Stores high-res crops of detected visuals.

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-your-openai-api-key
```
*Note: Synapse requires GPT-4o access.*

---

## 🖥️ Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Document**:
   - Upload any PDF (e.g., financial reports, research papers).
   - Click **"Process Document"**.
   - Watch the progress bar as it analyzes pages in parallel.

3. **Ask Questions**:
   - *General*: "Summarize the key findings."
   - *Visual-Specific*: "Show me the revenue table."
   - *Data Analysis*: "Compare the growth trends from the charts."

---

## 🔍 Troubleshooting

- **API Rate Limits**: If you see "429" errors, the system is designed to retry automatically. Just wait a moment.
- **Missing Visuals**: Ensure your PDF pages are clear. The vision model works best on standard resolution digital PDFs or high-quality scans.
