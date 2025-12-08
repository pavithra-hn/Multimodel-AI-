import os
import json
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAI # Removed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

class DocumentProcessor:
    def __init__(self):
        # Initialize OpenAI Chat model which supports vision (e.g., gpt-4o)
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), max_tokens=1000)

    def process_pdf(self, file_path):
        """
        Extracts text and images from a PDF.
        Returns a list of LangChain Documents.
        """
        documents = []
        
        # Open PDF with PyMuPDF for image extraction and pypdf for text (or just PyMuPDF for both)
        # Let's use PyMuPDF (fitz) for better image handling
        doc = fitz.open(file_path)
        
        # Ensure directory for images exists
        images_dir = "extracted_images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Extract tables
            tables = page.find_tables()
            table_markdowns = []
            for table in tables:
                table_markdowns.append(table.to_markdown())

            # Extract images
            image_list = page.get_images(full=True)
            image_descriptions = []
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image to disk
                image_filename = f"image_{os.path.basename(file_path)}_{page_num}_{img_index}.png"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Get image description from OpenAI
                description = self._describe_image(image_bytes)
                if description:
                    # We store the image path in the description text or handle it via metadata
                    # Putting it in the text might help context, but let's do metadata primarily for the UI.
                    # However, since one page can have multiple images, we might need a list of images in metadata.
                    # For simplicity, we will append a special token or just rely on the fact that we have descriptions.
                    # Actually, better to create a separate document for the image or attach it to the page doc.
                    # Let's attach to the page doc's metadata as a list of images.
                    
                    image_descriptions.append(f"[Image on Page {page_num + 1}]: {description}")
                    
            # Combine text, tables, and image descriptions
            page_content = text + "\n\n" + "\n\n".join(table_markdowns) + "\n\n" + "\n".join(image_descriptions)
            
            # Collect all image paths for this page
            page_image_paths = [os.path.join(images_dir, f"image_{os.path.basename(file_path)}_{page_num}_{i}.png") for i in range(len(image_list))]
            
            documents.append(Document(
                page_content=page_content,
                metadata={
                    "source": file_path, 
                    "page": page_num + 1,
                    "image_paths": json.dumps(page_image_paths) if page_image_paths else "[]"
                }
            ))
            
        return documents

    def _describe_image(self, image_bytes):
        """
        Uses OpenAI to describe the image content (charts, tables, etc.)
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image in detail. If it's a chart or table, summarize the key data points."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.llm.invoke([message])
            return response.content
            
        except Exception as e:
            print(f"Error describing image: {e}")
            return None
