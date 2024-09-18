from fastapi import FastAPI, File, UploadFile
import pytesseract
import google.generativeai as genai
import cv2  # Import cv2 here
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import pandas as pd
import re
from io import BytesIO
import tempfile
from fastapi.responses import StreamingResponse


# Configure Google Generative AI API Key
GOOGLE_API_KEY = 'AIzaSyCOY1MkhyEDsOYQC0LEOnQzCCQjEgJhBYI'
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

app = FastAPI()

def clean_text(text):
    """Remove unnecessary characters from the text."""
    text = re.sub(r'\*+', '', text)  # Remove asterisks
    text = re.sub(r'^\d+\.\s*', '', text)  # Remove leading numbers
    text = re.sub(r'-+', '', text)  # Remove hyphens
    return text.strip()

def process_image(image: np.array):
    """Process a single image (from PDF or directly an image file)."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    ocr_result = pytesseract.image_to_string(img_bin)

    # Use Google Generative AI to process the OCR-extracted text
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Extract the following relevant details from the invoice text:
    1. Invoice Number
    2. Invoice Date
    3. Buyer Name
    4. Buyer Address
    5. Buyer GSTIN
    6. Seller Name
    7. Seller Address
    8. Seller GSTIN
    9. State Name
    10. PO Number
    11. Invoice Amount
    12. Taxable Amount
    13. Total Amount
    14. Tax Amount (CGST, SGST, IGST)
    15. Total Tax Amount
    16. TCS Amount
    17. IRN Number
    18. Receiver GSTIN
    19. Receiver Name
    20. Vendor GSTIN
    21. Vendor Name
    22. Vendor Code
    23. Remarks
    24. Other relevant financial details

    The invoice text is as follows:
    {ocr_result}
    """

    response = model.generate_content(prompt)
    structured_output = response.text

    details = []
    for line in structured_output.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = clean_text(key)
            value = clean_text(value)
            details.append([key, value])

    if details:
        df = pd.DataFrame(details, columns=["Field", "Value"])
        return df
    return None

def export_to_excel(df: pd.DataFrame):
    """Export the DataFrame to an Excel file and return the buffer."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Invoice Details')
    buffer.seek(0)
    return buffer

@app.post("/process-invoice/")
async def process_invoice(file: UploadFile = File(...)):
    """Endpoint to process an invoice and extract details."""
    try:
        contents = await file.read()

        if file.content_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(contents)
                pdf_path = tmp_file.name
            images = convert_from_path(pdf_path)
            df = process_image(images[0])  # Process first page for simplicity

        elif file.content_type.startswith('image/'):
            image = Image.open(BytesIO(contents))
            df = process_image(image)

        if df is not None:
            return {"message": "Invoice details extracted", "data": df.to_dict(orient='records')}
        else:
            return {"message": "Failed to extract details"}

    except Exception as e:
        return {"error": str(e), "message": "An error occurred while processing the invoice."}

@app.post("/download-invoice-excel/")
async def download_invoice_excel(file: UploadFile = File(...)):
    """Endpoint to process an invoice and return an Excel file."""
    try:
        contents = await file.read()

        if file.content_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(contents)
                pdf_path = tmp_file.name
            images = convert_from_path(pdf_path)
            df = process_image(images[0])

        elif file.content_type.startswith('image/'):
            image = Image.open(BytesIO(contents))
            df = process_image(image)

        if df is not None:
            excel_file = export_to_excel(df)
            return StreamingResponse(
                excel_file,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=invoice_details.xlsx"}
            )
        else:
            return {"message": "Failed to extract details"}

    except Exception as e:
        return {"error": str(e), "message": "An error occurred while processing the invoice."}
