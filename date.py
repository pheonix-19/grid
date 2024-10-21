import streamlit as st
import cv2
import easyocr
import re
import os
import subprocess
import json
from PIL import Image
import numpy as np
import tempfile

def extract_dates(text):
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # Matches dates like 12/10/2024 or 12-10-24
        r'\b\d{1,2}\s+\w+\s+\d{2,4}\b',         # Matches dates like 12 October 2024
        r'\b\w+\s+\d{1,2},\s+\d{4}\b'           # Matches dates like October 12, 2024
    ]
    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        found_dates.extend(matches)
    return found_dates

def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)
    return " ".join(result)

def process_image_in_grids(image, rows=8, cols=8):
    height, width, _ = image.shape
    grid_height = height // rows
    grid_width = width // cols
    all_dates = []
    for i in range(rows):
        for j in range(cols):
            y_start = i * grid_height
            y_end = (i + 1) * grid_height if i != rows - 1 else height
            x_start = j * grid_width
            x_end = (j + 1) * grid_width if j != cols - 1 else width
            grid = image[y_start:y_end, x_start:x_end]
            extracted_text = extract_text_from_image(grid)
            dates = extract_dates(extracted_text)
            
            if dates:
                st.write(f"Dates found in grid ({i+1}, {j+1}): {dates}")
                all_dates.extend(dates)
    return all_dates

def select_date_with_ollama(dates):
    if not dates:
        return "No dates found."
    prompt = f"""
    You are an AI assistant tasked with selecting the most appropriate date from a list of extracted dates.
    Here are the dates: {json.dumps(dates)}
    Please analyze these dates and select the one that seems most likely to be the correct or most relevant date.
    Explain your reasoning briefly, then output your selected date in the format YYYY-MM-DD.
    If no date seems valid or relevant, output 'No valid date found.'
    Your response should be in the following format:
    Explanation: [Your explanation here]
    Selected date: [YYYY-MM-DD or 'No valid date found']
    """
    try:
        process = subprocess.run(
            ["ollama", "run", "gemma:2b"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=30  # 30 second timeout
        )
        
        if process.returncode != 0:
            st.error(f"Ollama error: {process.stderr}")
            return "Ollama call failed."
            
        response = process.stdout.strip()
        st.write("Ollama output:", response)
        return response
    except subprocess.TimeoutExpired:
        return "Ollama process timed out."
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

def main():
    st.title("Image Date Extractor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process Image'):
            with st.spinner('Processing...'):
                # Convert PIL Image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                all_found_dates = process_image_in_grids(image_cv)
                
                if all_found_dates:
                    st.write("All found dates:", all_found_dates)
                    ollama_result = select_date_with_ollama(all_found_dates)
                    st.write("LLM  decision:")
                    st.write(ollama_result)
                else:
                    st.write("No dates found in the image.")

if __name__ == "__main__":
    main()