import cv2
import easyocr
import openai
import numpy as np
from pyzbar.pyzbar import decode
import streamlit as st
import matplotlib.pyplot as plt
import tempfile

# Initialize OpenAI API key
openai.api_key = "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"  # Replace with your actual OpenAI API key

# Function to extract text from an image using EasyOCR
def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image, detail=0)
    return " ".join(result)

# Function to prompt GPT-3.5 to extract product details
def get_product_details(text):
    prompt = f"""
    Extract the product details from the following text:
    Text: {text}
    
    Required details:
    - Product Name
    - Product Type
    - Any Other Relevant Information related to nutrition
    
    Please provide the details in a structured format.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5,
    )
    
    return response.choices[0].message['content'].strip()

# Function to display the current frame using matplotlib
def display_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    st.pyplot(plt)

# Function to read barcode from an image
def read_barcodes_from_image(image):
    decoded_objects = decode(image)
    barcodes = []

    for obj in decoded_objects:
        barcode_data = obj.data.decode('utf-8')
        barcode_type = obj.type
        barcodes.append((barcode_data, barcode_type))

        points = obj.polygon
        if len(points) == 4:
            cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            x, y, w, h = obj.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x, y, w, h = obj.rect
        cv2.putText(image, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, barcodes

# Function to process multiple images (frames) and accumulate text for GPT analysis
def process_video_for_product_details(video_file, frame_interval=30, num_frames_to_process=5):
    # Save the uploaded video file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    frame_count = 0
    accumulated_text = ""

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return
    
    processed_frames = 0
    
    while cap.isOpened() and processed_frames < num_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            st.write(f"Processing frame {frame_count}...")
            display_frame(frame)
            
            # Convert frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract text from the frame using EasyOCR
            extracted_text = extract_text_from_image(rgb_frame)
            st.write(f"Extracted text from frame {frame_count}: {extracted_text}")
            accumulated_text += f" {extracted_text}"
            
            # Read barcodes from the frame
            processed_image, barcodes = read_barcodes_from_image(frame)
            st.image(processed_image, caption=f'Processed Frame {frame_count} with Barcodes', channels='BGR')
            
            if barcodes:
                for data, barcode_type in barcodes:
                    st.write(f"Detected barcode: {data} of type: {barcode_type}")
            else:
                st.write("No barcodes detected in this frame.")
            
            processed_frames += 1
        
        frame_count += 1
    
    # Process accumulated text with GPT-3.5
    if accumulated_text.strip():
        st.write("Sending accumulated text to GPT-3.5 for analysis...")
        product_details = get_product_details(accumulated_text)
        st.write("Extracted Product Information from Multiple Frames:")
        st.write(product_details)
    
    cap.release()

# Streamlit app layout
st.title("Product Details Extractor and Barcode Scanner")

# Upload video file
video_file = st.file_uploader("Upload Video", type=["mp4"])
if video_file:
    st.video(video_file)
    process_video_for_product_details(video_file, frame_interval=250, num_frames_to_process=2)

# Upload image file for barcode reading
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if image_file:
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    processed_image, barcodes = read_barcodes_from_image(image)
    st.image(processed_image, caption='Processed Image with Barcodes', channels='BGR')

    if barcodes:
        for data, barcode_type in barcodes:
            st.write(f"Detected barcode: {data} of type: {barcode_type}")
    else:
        st.write("No barcodes detected.")
