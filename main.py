from fastapi import FastAPI, File, UploadFile
import pytesseract
import layoutparser as lp
import cv2
import google.generativeai as genai
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import pandas as pd
import re
import tempfile
import os
from io import BytesIO
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure the Google Generative AI API Key directly
GOOGLE_API_KEY = 'AIzaSyCOY1MkhyEDsOYQC0LEOnQzCCQjEgJhBYI'

# Configure the API with the API Key
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the Tesseract executable path on your system
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Invoice Processing API!"}

# Function to clean text
def clean_text(text):
    text = re.sub(r'\*+', '', text)  # Remove asterisks
    text = re.sub(r'^\d+\.\s*', '', text)  # Remove leading numbers like '1. '
    text = re.sub(r'-+', '', text)  # Remove hyphens
    return text.strip()

# Function to process image
def process_image(image: np.array):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    ocr_result = pytesseract.image_to_string(img_bin)

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

@app.post("/process-invoice/")
async def process_invoice(file: UploadFile = File(...)):
    try:
        logging.info(f"Processing file: {file.filename}, type: {file.content_type}")
        contents = await file.read()

        if file.content_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(contents)
                pdf_path = tmp_file.name
            logging.info(f"PDF saved at {pdf_path}")

            images = convert_from_path(pdf_path)
            df = process_image(images[0])  # Process first page

        elif file.content_type.startswith('image/'):
            image = Image.open(BytesIO(contents))
            logging.info("Processing image file")
            df = process_image(image)

        if df is not None:
            logging.info("Details extracted successfully")
            return {"message": "Invoice details extracted", "data": df.to_dict(orient='records')}
        else:
            logging.error("Failed to extract details")
            return {"message": "Failed to extract details"}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e), "message": "An error occurred while processing the invoice."}
