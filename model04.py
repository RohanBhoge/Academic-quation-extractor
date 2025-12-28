"""
Streamlit application for batch processing of PDF MCQs with a 'Parallel Crop' workflow. Extracts data using Gemini 2.5 Flash and launches a separate subprocess (`manual_cropper.py`) to allow real-time manual verification and cropping of diagrams alongside AI extraction.
"""
import streamlit as st
import time
import os
import json
import re
import shutil
import tempfile
import uuid
import subprocess
import sys
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Tuple, Union
import fitz  # PyMuPDF

# Note: cv2 is NOT imported here to avoid conflicts. It runs in the subprocess.

st.set_page_config(
    page_title="PDF MCQ Extractor (Parallel Crop)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Pydantic Schemas ---
class QuestionData(BaseModel):
    id: int = Field(description="Literal printed question number.")
    chapter: str = Field(default="Physical World & Measurement", description="The chapter name, default or inferred.")
    question: str
    question_latex: str
    image_url: str = Field(default="", description="Descriptive tag if diagram is present.")
    options: List[str]
    difficulty: str = Field(description="Infer the difficulty: 'easy', 'medium', or 'hard'.")
    is_fragment: bool = Field(default=False, description="True if the question is incomplete and continues onto the next page.")
    page_index: int = Field(default=0, description="The index (0-based) of the image in the provided batch where this question primarily appears.")
    diagram_bboxes: List[List[int]] = Field(default=[], description="Bounding boxes [ymin, xmin, ymax, xmax] for diagrams.")
    extracted_image_names: List[str] = Field(default=[], description="Filenames of extracted diagrams.")

class KeyData(BaseModel):
    id: int
    answer_option: str = Field(description="The single correct option letter (A, B, C, or D).")

class SolutionData(BaseModel):
    id: int
    solution: str = Field(description="The detailed solution, all math in $$. Use \\n for newlines.")
    is_fragment: bool = Field(default=False, description="True if the solution is incomplete and continues onto the next page.")

class IntermediateResult(BaseModel):
    questions: List[QuestionData] = []
    answer_keys: List[KeyData] = []
    solutions: List[SolutionData] = []

class FinalMCQ(BaseModel):
    id: int
    chapter: str
    question: str
    question_latex: str
    image_url: str
    options: List[str]
    answer: str
    solution: str
    difficulty: str
    marks: int = Field(default=4)
    extracted_image_names: List[str] = Field(default=[], description="List of filenames for extracted diagrams associated with this question.")

# --- 2. LLM System Instruction ---
SYSTEM_INSTRUCTION = """
You are an expert academic data extraction engine specializing in Physics, Chemistry, Mathematics, Biology exams. 
Your task is to analyze the provided single image of an exam page and accurately extract all distinct data fragments present: **Questions, Answer Keys, or Detailed Solutions.**

Your output MUST populate the corresponding lists (questions, answer_keys, or solutions) within the 'IntermediateResult' schema. If a section is absent from the image, return its list as an empty array ([]).

--- PRIMARY EXTRACTION RULES (ID Mapping) ---
1.  **Question/Solution ID Source:** The 'id' field MUST be the **literal printed number** (e.g., 1, 17, 30) visible in the image.
2.  **ID Continuity:** The literal ID must be used consistently across all three schemas.
3.  **Page Indexing:** For every Question, identify the `page_index` (0, 1, 2...) corresponding to the image in the input list where the question starts.

--- STRENGTHENED CONTENT EXTRACTION RULES ---
1.  **Fragment Flagging (CRITICAL):**
    * If a Question or Solution starts on this page but is clearly **truncated** or **incomplete**, set `is_fragment` to `true`.
    * If the question/solution is complete, set `is_fragment` to `false`.
2.  **Questions & Options (Content):** Extract the **full body, options, and question_latex**.
3.  **Visual Grounding (Diagrams):** 
    Role: You are a precision document parser specialized in extracting technical illustrations.
    **STRICT EXTRACTION RULES:**
    * **Identify the visual diagram** (lines, shapes, containers).
    * **Define the "Visual Envelope":** The bounding box should encapsulate all graphical strokes and immediate callouts.
    * **Strict Margin Rule:** Exclude paragraph text aligned with the diagram unless it's a label.
    * **Safety Buffer:** Provide a 1-2% "loose" fit.
    * **Format:** Output ONLY a JSON list of bounding boxes: [ymin, xmin, ymax, xmax] in normalized (0-1000) coordinates.

--- STRICT FORMATTING & QUALITY RULES ---
1.  **JSON Output:** The entire output MUST be a JSON object conforming to the 'IntermediateResult' schema.
2.  **LaTeX Requirement:** All math MUST be in LaTeX enclosed in single dollar signs ($$).
3.  **Solution Formatting:** Use '\\n' for newlines.
4.  **Image URL:** Set `image_url` to a descriptive tag if a diagram is present.
5.  **Placeholders:** Use empty string ("") for missing details.
"""

def convert_pdfs_to_images(pdf_paths: List[Path], output_dir: Path, dpi: int = 300) -> List[Path]:
    all_image_paths = []
    zoom_factor = dpi / 72
    matrix = fitz.Matrix(zoom_factor, zoom_factor)

    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                output_filename = f"{pdf_path.stem}_page_{page_num + 1}.png"
                output_path = output_dir / output_filename
                pix.save(output_path)
                all_image_paths.append(output_path)
            doc.close()
        except Exception as e:
            st.error(f"Error converting PDF {pdf_path.name}: {e}")
            continue
    return all_image_paths

def process_batch_with_ai(image_paths: List[Path], client: genai.Client, system_instruction: str, retries: int = 5, initial_delay: int = 10) -> Optional[IntermediateResult]:
    for attempt in range(retries):
        try:
            images = [Image.open(p) for p in image_paths]
            content_payload = [system_instruction] + images
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=content_payload,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=IntermediateResult,
                )
            )
            data = IntermediateResult.model_validate_json(response.text)
            return data
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = initial_delay * (2 ** attempt) 
                st.warning(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
                continue
            elif isinstance(e, ValidationError):
                st.warning(f"Validation error: {e}")
                return None 
            else:
                st.error(f"API Error: {e}")
                return None
    st.error(f"Failed after {retries} retries.")
    return None

def extract_raw_diagrams(image_paths: List[Path], page_index: int, bboxes: List[List[int]], q_id: int, output_dir: Path) -> List[str]:
    saved_files = []
    if not bboxes: return saved_files
    if page_index < 0 or page_index >= len(image_paths): page_index = 0
    target_image_path = image_paths[page_index]

    try:
        with Image.open(target_image_path) as img:
            width, height = img.size
            for i, bbox in enumerate(bboxes):
                if len(bbox) != 4: continue
                ymin, xmin, ymax, xmax = bbox
                
                # --- PADDING: 80px ---
                padding = 80
                left = (xmin * width / 1000)
                top = (ymin * height / 1000)
                right = (xmax * width / 1000)
                bottom = (ymax * height / 1000)

                left = max(0, left - padding)
                top = max(0, top - padding)
                right = min(width, right + padding)
                bottom = min(height, bottom + padding)
                
                if right > left and bottom > top:
                    filename = f"q{q_id}_diag_{i}_{uuid.uuid4().hex[:4]}.png"
                    filepath = output_dir / filename
                    img.crop((left, top, right, bottom)).save(filepath)
                    saved_files.append(filename)
    except Exception as e:
        st.warning(f"Failed to crop diagram q{q_id}: {e}")
    return saved_files

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]

def merge_data(all_q_data: List[QuestionData], all_key_data: List[KeyData], all_sol_data: List[SolutionData]) -> List[FinalMCQ]:
    question_map = {}
    key_map = {k.id: k.answer_option for k in all_key_data}
    solution_map = {s.id: s.solution for s in all_sol_data} 

    for q_data in all_q_data:
        if q_data.id not in question_map:
            question_map[q_data.id] = q_data
        else:
             existing = question_map[q_data.id]
             existing.question += " " + q_data.question
             existing.options.extend(q_data.options)
             if q_data.question_latex:
                 existing.question_latex += " " + q_data.question_latex

    final_mcqs = []
    sorted_ids = sorted(question_map.keys())
    final_id_counter = 1

    for original_id in sorted_ids:
        q_data = question_map[original_id]
        option_letter = key_map.get(original_id, "")
        answer_string = ""
        if option_letter and q_data.options:
            option_index_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            index = option_index_map.get(option_letter.upper())
            if index is not None and 0 <= index < len(q_data.options):
                answer_string = q_data.options[index]
        
        final_image_url = q_data.image_url
        if q_data.extracted_image_names:
            final_image_url = q_data.extracted_image_names[0]

        mcq = FinalMCQ(
            id=final_id_counter,
            chapter=q_data.chapter,
            question=q_data.question,
            question_latex=q_data.question_latex,
            image_url=final_image_url,
            options=q_data.options,
            answer=answer_string, 
            solution=solution_map.get(original_id, ""),
            difficulty=q_data.difficulty,
            marks=4,
            extracted_image_names=q_data.extracted_image_names
        )
        final_mcqs.append(mcq)
        final_id_counter += 1
    return final_mcqs

def main():
    st.title("📚 PDF Extractor + Parallel Manual Crop")
    st.info("The Manual Cropping window will open automatically as diagrams are found.")

    with st.sidebar:
        user_api_key = st.text_input("Gemini API Key", type="password")
        if user_api_key:
            try:
                client = genai.Client(api_key=user_api_key)
                st.success("Ready")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        else:
            st.warning("Enter API Key")
            st.stop()
            
        st.divider()
        if st.button("🛠️ Debug: Test Crop Window"):
            # Create a dummy environment to test the pop-up
            debug_dir = Path(tempfile.mkdtemp())
            debug_raw = debug_dir / "raw"
            debug_final = debug_dir / "final"
            debug_raw.mkdir()
            debug_final.mkdir()
            
            # Create a dummy image
            img = Image.new('RGB', (400, 300), color = (73, 109, 137))
            img.save(debug_raw / "test_image.png")
            
            cmd = [sys.executable, "manual_cropper.py", str(debug_raw), str(debug_final), str(debug_dir / "DONE")]
            try:
                subprocess.Popen(cmd, cwd=os.getcwd(), creationflags=subprocess.CREATE_NEW_CONSOLE)
                st.info("Launched Test Window! Check your taskbar.")
            except Exception as e:
                st.error(f"Failed to launch: {e}")

    if "stage" not in st.session_state:
        st.session_state.stage = "upload" # upload -> extracting -> done
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "final_mcqs" not in st.session_state:
        st.session_state.final_mcqs = None

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.session_state.stage == "upload":
        if st.button("🚀 Start Parallel Extraction"):
            st.session_state.stage = "extracting"
            st.rerun()

    if st.session_state.stage == "extracting":
        if not st.session_state.temp_dir:
            temp_dir = Path(tempfile.mkdtemp())
            st.session_state.temp_dir = str(temp_dir)
            
            raw_dir = temp_dir / "raw_diagrams"
            final_dir = temp_dir / "final_diagrams"
            (temp_dir / "pdfs").mkdir(exist_ok=True)
            (temp_dir / "processed_images").mkdir(exist_ok=True)
            raw_dir.mkdir(exist_ok=True)
            final_dir.mkdir(exist_ok=True)

        temp_dir = Path(st.session_state.temp_dir)
        raw_dir = temp_dir / "raw_diagrams"
        final_dir = temp_dir / "final_diagrams"
        done_signal = temp_dir / "DONE"

        # --- LAUNCH SUBPROCESS FOR CROPPING ---
        debug_log_path = temp_dir / "cropper_debug.log"
        cmd = [sys.executable, "manual_cropper.py", str(raw_dir), str(final_dir), str(done_signal)]
        
        try:
             # Use Popen to run non-blocking, redirecting output to log file
             with open(debug_log_path, "w") as log_file:
                 cropper_process = subprocess.Popen(
                     cmd, 
                     cwd=os.getcwd(), 
                     creationflags=subprocess.CREATE_NEW_CONSOLE,
                     stdout=log_file,
                     stderr=subprocess.STDOUT
                 )
             st.success("✅ Manual Cropping Window Launched! Check your taskbar.")
             st.caption(f"Debug Log: {debug_log_path}")
        except Exception as e:
             st.error(f"Failed to launch cropper: {e}")
             st.stop()

        progress_bar = st.progress(0, text="Processing...")
        
        try:
            # 1. Save PDFs
            pdf_paths = []
            for f in uploaded_files:
                p = temp_dir / "pdfs" / f.name
                with open(p, "wb") as out:
                    out.write(f.getbuffer())
                pdf_paths.append(p)
            
            # 2. Convert
            progress_bar.progress(10, "Converting...")
            images = convert_pdfs_to_images(pdf_paths, temp_dir / "processed_images")
            images = sorted(images, key=natural_sort_key)
            
            # 3. Batch AI
            BATCH_SIZE = 50
            batches = [images[i:i + BATCH_SIZE] for i in range(0, len(images), BATCH_SIZE)]
            
            all_q, all_k, all_s = [], [], []
            
            for i, batch in enumerate(batches):
                pct = 10 + int(80 * (i+1)/len(batches))
                progress_bar.progress(pct, f"AI Extracting Batch {i+1}/{len(batches)} (Populating Crop Queue)...")
                
                res = process_batch_with_ai(batch, client, SYSTEM_INSTRUCTION)
                if res:
                    # Count diagrams found in this batch
                    diag_count = sum(1 for q in res.questions if q.extracted_image_names)
                    
                    # Save RAW diagrams -> This triggers the subprocess watcher!
                    for q in res.questions:
                        q.extracted_image_names = extract_raw_diagrams(
                            batch, q.page_index, q.diagram_bboxes, q.id, raw_dir
                        )
                        # Re-calculate count after saving to verify they were written
                        
                    # Update count to reflect successful writes (since extract_raw_diagrams returns list of names)
                    saved_diag_count = sum(len(q.extracted_image_names) for q in res.questions)
                    
                    if saved_diag_count > 0:
                         st.success(f"Batch {i+1}: Found {saved_diag_count} diagrams! Check Crop Window.")
                    else:
                         st.info(f"Batch {i+1}: Processed (No diagrams detected).")

                    all_q.extend(res.questions)
                    all_k.extend(res.answer_keys)
                    all_s.extend(res.solutions)
            
            # 4. Finish
            final_mcqs = merge_data(all_q, all_k, all_s)
            st.session_state.final_mcqs = [m.model_dump() for m in final_mcqs]
            
            # Signal Done to subprocess
            with open(done_signal, "w") as f:
                f.write("done")
                
            progress_bar.progress(95, "Waiting for you to finish cropping...")
            
            # Wait for subprocess to exit
            status_text = st.empty()
            while cropper_process.poll() is None:
                status_text.warning("⏳ Waiting for Manual Cropping to complete... (Close the crop window when done)")
                time.sleep(1)
            
            status_text.success("Cropping Complete!")
            progress_bar.progress(100)
            
            st.session_state.stage = "done"
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            if cropper_process: cropper_process.terminate()

    if st.session_state.stage == "done":
        st.header("✅ Processing Complete")
        
        final_dir = Path(st.session_state.temp_dir) / "final_diagrams"
        zip_path = Path(st.session_state.temp_dir) / "extracted_diagrams"
        
        shutil.make_archive(str(zip_path), 'zip', final_dir)
        
        json_str = json.dumps(st.session_state.final_mcqs, indent=2, ensure_ascii=False)
        
        col1, col2 = st.columns(2)
        with col1:
             st.download_button("⬇️ Download JSON", json_str, "mcq_data.json", "application/json")
        with col2:
             with open(str(zip_path) + ".zip", "rb") as f:
                 st.download_button("⬇️ Download Diagrams (ZIP)", f.read(), "diagrams.zip", "application/zip")

        if st.button("Start New"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
