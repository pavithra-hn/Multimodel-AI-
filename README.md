# Synapse: Intelligent Multi-Modal RAG System 🧠

Project Title

Synapse – Intelligent Multi-Modal RAG System

Project Description

Synapse is an advanced multi-modal Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding and question answering. The system processes PDFs by combining textual and visual (vision-based) analysis, enabling accurate handling of complex layouts such as tables, charts, and figures that are often missed by traditional text-only parsers.

Instead of relying solely on text extraction, Synapse performs vision-based layout detection on document pages, identifies and crops visual elements, converts tables into structured Markdown, and stores enriched representations in a vector database (ChromaDB). A context-aware retrieval pipeline selects the most relevant textual and visual content, allowing users to ask both semantic questions (e.g., summaries, explanations) and visual-specific queries (e.g., “show the revenue table”).

The system is deployed through a Streamlit interface, offering an end-to-end workflow from document upload to interactive question answering, making it suitable for financial reports, research papers, and enterprise documents.

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
