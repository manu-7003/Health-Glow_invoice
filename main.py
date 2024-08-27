from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
import shutil
import os

# Configure the Google Generative AI API Key directly
GOOGLE_API_KEY = 'AIzaSyCOY1MkhyEDsOYQC0LEOnQzCCQjEgJhBYI'
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the Tesseract executable path on your system
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-invoice/", response_class=HTMLResponse)
async def upload_invoice(request: Request, file: UploadFile = File(...)):
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

        # Convert DataFrame to HTML table (if details are extracted)
        if df is not None:
            extracted_details = df.to_html(classes="table table-striped")
            return templates.TemplateResponse("index.html", {"request": request, "result": extracted_details})
        else:
            return templates.TemplateResponse("index.html", {"request": request, "result": "Failed to extract details."})
    
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})

def process_image(image: np.array):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    ocr_result = pytesseract.image_to_string(img_bin)

    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Extract the following relevant details from the invoice text:
    [invoice extraction details here]
    {ocr_result}
    """
    
    response = model.generate_content(prompt)
    structured_output = response.text

    details = []
    for line in structured_output.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            details.append([key.strip(), value.strip()])

    if details:
        df = pd.DataFrame(details, columns=["Field", "Value"])
        return df
    return None
