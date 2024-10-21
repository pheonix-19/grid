import streamlit as st
import cv2
import easyocr
import re
import os
import json
from PIL import Image
import numpy as np
import tempfile
import openai  # Make sure to install the OpenAI library and set your API key

# Function to extract dates from text using regex patterns
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
openai.api_key = "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"
# Function to extract text from an image using EasyOCR
def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)
    return " ".join(result)

# Function to process the image in grids and extract dates from each grid
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

# Function to select the most appropriate date using OpenAI's GPT-4o-mini model
def select_date_with_gpt(dates):
    if not dates:
        return "No dates found."
    
    # Create the prompt
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
        # Make the OpenAI API call
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Change this to the appropriate model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Adjust based on expected length
            temperature=0.5,
        )
        
        gpt_response = response.choices[0].message.content.strip()
        st.write("output:", gpt_response)
        return gpt_response
    except Exception as e:
        return f"Error calling llm: {str(e)}"

# Streamlit app
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
                    gpt_result = select_date_with_gpt(all_found_dates)
                    st.write("LLM decision:")
                    st.write(gpt_result)
                else:
                    st.write("No dates found in the image.")

if __name__ == "__main__":
    main()
