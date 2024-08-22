from io import BytesIO
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
import tempfile  # Ensure tempfile is imported

# Configure the Google Generative AI API Key
GOOGLE_API_KEY = 'AIzaSyCOY1MkhyEDsOYQC0LEOnQzCCQjEgJhBYI'
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the Tesseract executable path on your system
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

app = FastAPI()

def clean_text(text):
    """Remove unnecessary characters from the text."""
    text = re.sub(r'\*+', '', text)  # Remove asterisks
    text = re.sub(r'^\d+\.\s*', '', text)  # Remove leading numbers like '1. '
    text = re.sub(r'-+', '', text)  # Remove hyphens
    return text.strip()

def process_image(image: np.array):
    """Process a single image (from PDF or directly an image file)."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Perform OCR using Tesseract
    ocr_result = pytesseract.image_to_string(img_bin)

    # Use Google Generative AI (Gemini) to process the OCR-extracted text
    model = genai.GenerativeModel('gemini-pro')

    # Enhanced prompt providing more context and constraints
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

    # Parse the structured output into a list of key-value pairs
    details = []
    for line in structured_output.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            # Clean the key and value by removing unnecessary characters
            key = clean_text(key)
            value = clean_text(value)
            details.append([key, value])

    # Create a pandas DataFrame to return the details
    if details:
        df = pd.DataFrame(details, columns=["Field", "Value"])
        return df
    return None

@app.post("/process-invoice/")
async def process_invoice(file: UploadFile = File(...)):
    """Endpoint to process an invoice and extract details."""
    try:
        # Log file details
        print(f"Processing file: {file.filename}, type: {file.content_type}")
        
        # Read the file
        contents = await file.read()

        # Check if it's an image or a PDF
        if file.content_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(contents)
                pdf_path = tmp_file.name
            print(f"PDF saved at {pdf_path}")

            # Convert PDF to image
            images = convert_from_path(pdf_path)
            df = process_image(images[0])  # Process first page for simplicity

        elif file.content_type.startswith('image/'):
            image = Image.open(BytesIO(contents))
            print("Processing image file")
            df = process_image(image)

        if df is not None:
            print("Details extracted successfully")
            return {"message": "Invoice details extracted", "data": df.to_dict(orient='records')}
        else:
            print("Failed to extract details")
            return {"message": "Failed to extract details"}

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message
        return {"error": str(e), "message": "An error occurred while processing the invoice."}
