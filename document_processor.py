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
import uuid

class DocumentProcessor:
    def __init__(self):
        # Initialize OpenAI Chat model
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), max_tokens=1500)

    def process_pdf(self, file_path):
        """
        Extracts text and visual elements (Tables/Charts) using Vision-based detection.
        Renders pages to images, visual-detects bounding boxes, crops, and analyzes them.
        Uses Parallel Processing for speed.
        """
        import concurrent.futures
        
        doc = fitz.open(file_path)
        num_pages = len(doc)
        doc.close()
        
        # Base directories
        base_images_dir = "extracted_images"
        dirs = {
            "table": os.path.join(base_images_dir, "tables"),
            "chart": os.path.join(base_images_dir, "charts"),
            "figure": os.path.join(base_images_dir, "figures")
        }
        
        for d in dirs.values():
            if not os.path.exists(d):
                os.makedirs(d)

        documents = []
        
        # Parallel processing of pages
        print(f"Starting parallel processing of {num_pages} pages...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Map page numbers to futures
            future_to_page = {
                executor.submit(self._process_page, page_num, file_path, dirs): page_num 
                for page_num in range(num_pages)
            }
            
            # Collect results as they complete (but we need to reorder them later)
            results = []
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    doc_obj = future.result()
                    results.append((page_num, doc_obj))
                    print(f"Page {page_num+1} processed successfully.")
                except Exception as exc:
                    print(f"Page {page_num+1} generated an exception: {exc}")

        # Sort results by page number to maintain document order
        results.sort(key=lambda x: x[0])
        documents = [r[1] for r in results if r[1] is not None]
            
        return documents

    def _process_page(self, page_num, file_path, dirs):
        """
        Process a single page: Extract text, render image, detect layout, crop, and analyze.
        Opens its own file handle for thread safety.
        """
        try:
            # Open document locally for thread safety
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            
            # 1. Extract Text (Reliable for standard text)
            text = page.get_text()
            
            # 2. Render Page as Image for Vision (Higher DPI for better quality)
            # Matrix(3, 3) approximates ~216 DPI if base is 72, or often results in ~300 DPI effectively depending on PDF
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) 
            img_data = pix.tobytes("png")
            full_page_image = Image.open(io.BytesIO(img_data))
            
            # 3. Detect Tables/Charts/Figures via Vision
            # Note: _detect_layout calls LLM
            detected_items = self._detect_layout(img_data)
            
            page_image_metadata = []
            visual_descriptions = []
            
            for idx, item in enumerate(detected_items):
                bbox = item.get("bbox") # [ymin, xmin, ymax, xmax] (0-1000 scale)
                label = item.get("type", "figure").lower()
                
                # Convert 0-1000 scale to pixel coordinates
                w, h = full_page_image.size
                ymin, xmin, ymax, xmax = bbox
                
                # Add padding (e.g., 2% of dimension) so we don't cut off labels
                pad_x = int(w * 0.02)
                pad_y = int(h * 0.02)
                
                crop_box = (
                    max(0, int(xmin * w / 1000) - pad_x), 
                    max(0, int(ymin * h / 1000) - pad_y), 
                    min(w, int(xmax * w / 1000) + pad_x), 
                    min(h, int(ymax * h / 1000) + pad_y)
                )
                
                # Crop
                # Ensure valid box
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue
                    
                crop_img = full_page_image.crop(crop_box)
                
                # Convert crop to bytes
                buf = io.BytesIO()
                crop_img.save(buf, format="PNG")
                crop_bytes = buf.getvalue()
                
                # Generate Unique ID for this visual element to link it strongly with text
                visual_id = str(uuid.uuid4())[:8] # Short unique ID
                
                # Analyze Crop (Get detailed description + Markdown for tables)
                # We prioritize the label from detection, but allow analysis to refine description
                # If detected as table, we explicitly ask for markdown
                analysis = self._analyze_crop(crop_bytes, label)
                
                if analysis:
                    valid_type = label # Trust the detection type primarily
                    description = analysis.get("description", "")
                    table_md = analysis.get("markdown", "")
                    
                    # Normalize directory
                    target_dir = dirs["figure"] # Default
                    
                    if "table" in valid_type: 
                        target_dir = dirs["table"]
                        valid_type = "table" # Normalize name
                    elif any(x in valid_type for x in ["chart", "graph", "plot"]): 
                        target_dir = dirs["chart"]
                        valid_type = "chart" # Normalize name
                    else:
                        valid_type = "figure" # Normalize name
                    
                    # Save Crop for reference
                    image_filename = f"{valid_type}_{os.path.basename(file_path)}_{page_num}_{idx}.png"
                    image_path = os.path.join(target_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(crop_bytes)
                        
                    # Store Metadata including the Unique ID
                    page_image_metadata.append({
                        "id": visual_id,
                        "path": image_path,
                        "type": valid_type,
                        "description": description,
                        "markdown": table_md if "table" in valid_type else ""
                    })
                    
                    # Add to content with the ID explicitly
                    # This ensures that when this text chunk is retrieved, we have the ID to lookup the image
                    if "table" in valid_type and table_md:
                        visual_descriptions.append(f"\n[Detected Table ID: {visual_id}] (Page {page_num + 1}):\n{description}\n\nReconstructed Table Data:\n{table_md}\n")
                    else:
                        visual_descriptions.append(f"\n[Detected {valid_type.capitalize()} ID: {visual_id}] (Page {page_num + 1}):\n{description}\n")

            # Final Page Content
            final_content = text + "\n".join(visual_descriptions)
            
            doc.close()
            
            return Document(
                page_content=final_content,
                metadata={
                    "source": file_path, 
                    "page": page_num + 1,
                    "image_metadata": json.dumps(page_image_metadata)
                }
            )
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            return None

    def _detect_layout(self, image_bytes):
        """
        Sends full page to GPT-4o to get bounding boxes for tables and charts.
        Returns list of dicts: [{'type': 'table'|'chart'|'figure', 'bbox': [ymin, xmin, ymax, xmax]}]
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            prompt = """Look at this document page. Detect all:
            1. Tables (any grid-like data, financial statements, or list structures).
            2. Charts / Graphs (bar, line, pie, scatter, etc.).
            3. Figures (diagrams, flowcharts, technical illustrations).
            
            Return a JSON object with a key "items" containing a list of detected objects.
            For each object, provide:
            - "type": EXACTLY one of "table", "chart", or "figure". Do not use "image" or "other".
            - "bbox": [ymin, xmin, ymax, xmax] 
              where coordinates are on a 0-1000 scale (relative to image height/width). 
              0,0 is top-left. 1000,1000 is bottom-right.
            
            Example:
            { "items": [ {"type": "table", "bbox": [100, 50, 400, 950]} ] }
            
            If nothing found, return { "items": [] }
            """
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.llm.invoke([message], config={"run_name": "layout_detection"})
            content = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return data.get("items", [])
            
        except Exception as e:
            print(f"Error detecting layout: {e}")
            return []

    def _analyze_crop(self, image_bytes, detected_type):
        """
        Analyzes a specific crop. If type is table, extracts markdown.
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            type_instruction = ""
            if "table" in detected_type:
                type_instruction = "This is a table. Extract all data into a standard Markdown table format in the 'markdown' field."
            else:
                type_instruction = "Describe this visual element in detail."
                
            prompt = f"""Analyze this image crop.
            {type_instruction}
            Provide a detailed "description".
            
            Return JSON:
            {{
                "description": "...",
                "markdown": "..." (If table: strictly use standard Markdown table syntax with | separators. Do not wrap in ```markdown code blocks inside the JSON value.)
            }}
            """
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.llm.invoke([message], config={"run_name": "crop_analysis"})
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
            
        except Exception as e:
            print(f"Error analyzing crop: {e}")
            return None
