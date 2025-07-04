import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    all_text = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            all_text.append(text.strip())
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            all_text.append(ocr_text.strip())

    return "\n".join(all_text)
