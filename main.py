import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import csv
import io
import numpy as np
from transformers import pipeline
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import easyocr

# === CONFIGURATION ===
DOWNLOADS = Path.home() / "Downloads"
DOCUMENTS = Path.home() / "Documents"
LOG_FILE = DOCUMENTS / "ai_file_sort_log.csv"

CATEGORIES = ["IT" "IT/Jens", "Math", "History", "Coding", "Finacial Literacy", "College Prep", "English", "Other"]

print("Loading classification model...")
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
)
reader = easyocr.Reader(['en'], gpu=False)
print("Models ready.")

# === OCR HELPERS (No Poppler Version) ===
def ocr_pdf(file_path):
    """Convert scanned PDF pages to text via OCR using PyMuPDF (no Poppler)."""
    text_output = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)  # High-res image for OCR
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_np = np.array(img)  
            result = reader.readtext(img_np, detail=0)
            text_output.append("\n".join(result))
        return "\n\n".join(text_output)
    except Exception as e:
        print(f"[!] OCR failed for {file_path.name}: {e}")
        return ""
# === TEXT EXTRACTION ===
def extract_text_from_file(file_path):
    ext = file_path.suffix.lower()
    try:
        if ext == ".txt":
            return file_path.read_text(errors='ignore')

        elif ext == ".pdf":
            doc = fitz.open(file_path)
            text = "\n".join(page.get_text() for page in doc)
            if text.strip():
                return text
            else:
                print(f"No text found in PDF, running OCR...")
                return ocr_pdf(file_path)

        elif ext == ".docx":
            return "\n".join([p.text for p in Document(file_path).paragraphs])

        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            print(f"Running OCR on image: {file_path.name}")
            if not file_path.exists() or file_path.stat().st_size == 0:
                print(f"[!] Image file is empty or missing: {file_path}")
                return ""
            result = reader.readtext(str(file_path), detail=0)
            return "\n".join(result)

        else:
            print(f"Unsupported file type: {file_path.name}")
            return ""

    except Exception as e:
        print(f"[!] Error reading {file_path.name}: {e}")
        return ""

# === CLASSIFY TEXT ===
def classify_text(text):
    result = classifier(text[:1000], CATEGORIES)
    return result['labels'][0]

# === MOVE FILE ===
def move_file(file_path, category):
    destination = DOCUMENTS / category
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / file_path.name

    try:
        shutil.move(str(file_path), target)
        print(f"Moved '{file_path.name}' → {category}")
        return str(target)
    except Exception as e:
        print(f"[!] Error moving {file_path.name}: {e}")
        return None

# === LOG ACTION ===
def log_action(file_name, category, new_path):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), file_name, category, new_path])

# === MAIN ===
def process_downloads():
    print(f"\nScanning: {DOWNLOADS}")
    for file_path in DOWNLOADS.iterdir():
        if file_path.is_file():
            print(f"\n→ Processing: {file_path.name}")
            text = extract_text_from_file(file_path)

            category = None

            if text.strip():
                try:
                    category = classify_text(text)
                    print(f"AI classified from content: {category}")
                except Exception as e:
                    print(f"[!] Text classification failed: {e}")
            else:
                print("[-] No readable text found.")

            # Fallback to file name classification if needed
            if not category:
                print("Falling back to filename-based classification...")
                try:
                    category = classify_text(file_path.stem)
                    print(f"AI classified from filename: {category}")
                except Exception as e:
                    print(f"[!] Filename classification failed: {e}")
                    category = "Other"

            new_path = move_file(file_path, category)
            if new_path:
                log_action(file_path.name, category, new_path)

    print("\nDone. Log saved to:", LOG_FILE)


# === INIT LOG FILE ===
if not LOG_FILE.exists():
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "file_name", "category", "new_path"])

# === RUN ===
if __name__ == "__main__":
    process_downloads()

