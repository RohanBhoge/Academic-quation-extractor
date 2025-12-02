This project is highly specialized, combining PDF processing, AI extraction, and data structuring using Streamlit and Google Gemini. Here is the professional README file, modeled after your excellent example.

-----

# Exam Data Extractor: AI-Powered MCQ Structuring Pipeline 📝

[](https://www.python.org/downloads/)
[](https://streamlit.io)
[](https://ai.google.dev/)
[](https://opensource.org/licenses/MIT)

The Exam Data Extractor is a robust **multimodal pipeline** designed to automate the conversion of educational PDF exam papers (containing questions, options, and solutions) into a clean, unified, structured **JSON array**. It leverages Google's Gemini AI for visual analysis (OCR and comprehension) and Pydantic for strict schema enforcement, making it ideal for creating high-quality, normalized question banks.

-----

## ✨ Key Features

  * **Multimodal Analysis:** Uses the Gemini model's vision capabilities to process **image representations of PDF pages**. This allows for accurate extraction of data from scanned text, diagrams, and complex layouts.
  * **LaTeX Conversion:** Automatically converts all extracted mathematical expressions, physics formulas, and dimensional analysis into standard **LaTeX format** (e.g., $\frac{\Delta g}{g} = \frac{\Delta L}{L} + 2\frac{\Delta T}{T}$).
  * **Contextual Stitching:** Implements a post-processing algorithm to solve the challenge of **fragmented data**. If a question or solution spans multiple PDF pages, the system automatically detects and **stitches** the pieces back together into a single, cohesive entry.
  * **Structured Output:** Enforces strict compliance with a defined Pydantic schema (`FinalMCQ`), ensuring the output is standardized and ready for database ingestion or application use.
  * **Component-Level Extraction:** Extracts three distinct data types in parallel—Questions, Answer Keys, and Solutions—and intelligently **merges** them based on the original question ID.
  * **Streamlit Interface:** Provides a simple, interactive web interface for uploading PDFs and downloading the final JSON file.

## 🏗️ Architecture: A Three-Phase Multimodal Pipeline

The application is structured into sequential phases managed by the Streamlit frontend.

### Phase 1: Ingestion and Pre-processing (PDF to Images)

1.  **PDF Loading:** The user uploads one or more PDF files via the Streamlit interface.
2.  **Conversion:** The code uses **PyMuPDF (`fitz`)** to reliably convert every page of the uploaded PDFs into high-resolution PNG images.
3.  **Sorting:** Image files are sorted naturally (e.g., `page_9` before `page_10`) to maintain correct page order for subsequent stitching.

### Phase 2: AI Extraction (Multimodal Analysis)

1.  **AI Invocation:** Each generated image is sent individually to the **Gemini 2.5 Flash** model.
2.  **Prompt & Schema:** The request includes a detailed `SYSTEM_INSTRUCTION` (acting as the prompt) and the `IntermediateResult` Pydantic schema.
3.  **Fragment Flagging:** The prompt instructs the model to set the **`is_fragment: true`** flag for any question or solution that appears truncated at the edge of the page.
4.  **Data Extraction:** The model performs OCR and content comprehension, returning a semi-processed JSON object for each page.

### Phase 3: Post-processing and Finalization

1.  **Data Consolidation:** All fragments (Questions, Keys, Solutions) from all pages are collected into separate lists.
2.  **Stitching (`stitch_fragments`):** The core logic iterates over the collected data:
      * It checks for items with the same `id` that appeared sequentially and were marked as fragments.
      * It concatenates text, options, and solution bodies until a complete (non-fragment) entry is formed.
3.  **Final Merge (`merge_data`):** The fully stitched Questions, Keys, and Solutions are combined based on their unique original ID. The raw option letter (e.g., 'B') from the key is mapped to the final option text.
4.  **Output:** The final data is converted into a structured JSON array (`List[FinalMCQ]`) ready for download.

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | **Streamlit** | Interactive UI for file upload and configuration. |
| **LLM/Vision** | **Google Gemini 2.5 Flash** | Multimodal analysis, OCR, comprehension, and structured JSON generation. |
| **PDF Processing** | **PyMuPDF (fitz)** | Robust PDF page-to-image conversion. |
| **Schema Validation** | **Pydantic** | Defines and enforces the strict input/output JSON structure. |
| **Orchestration** | **Python (Standard Libraries)** | Manages the file pipeline, stitching logic, and final data merging. |

## ⚙️ Setup and Local Installation

Follow these steps to run the project on your local machine.

### 1\. Prerequisites

You need **Python 3.10+** and the Google Gemini API key.

### 2\. Clone the Repository (Assuming Git)

```bash
git clone https://github.com/RohanBhoge/Academic-quation-extractor
cd exam-data-extractor
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
# Ensure you have PyMuPDF and Streamlit dependencies installed
```

*(Note: Since `fitz` (PyMuPDF) is used for PDF rendering, no external Poppler installation is required, unlike `pdf2image`.)*

### 4\. Set Up Environment Variables

Create a file named `.env` in the root of the project and add your Google API key:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 5\. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser, allowing you to upload your PDF exam files.

## 📂 Project Structure

```
.
├── .env                  # For local environment variables (API Key)
├── .gitignore
├── app.py                # The main Streamlit application script
└── requirements.txt      # Project dependencies (streamlit, google-genai, pydantic, pymupdf, etc.)
```

## 📄 License

This project is licensed under the MIT License.