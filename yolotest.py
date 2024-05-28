import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import pytesseract
import re
import numpy as np
import math

# Set up pytesseract executable path if needed
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'

# Load the pre-trained YOLO model
model = YOLO("Models/best50Epochs.pt")

# Function to perform license plate detection and OCR
def detect_and_ocr(image):
    # Convert the image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform prediction on the input image
    results = model.predict(img_cv, save=False)

    # Initialize a list to store detected license plates and their OCR results
    detected_plates = []

    # Extract the bounding boxes of detected objects
    for result in results:
        boxes = result.boxes  # Bounding boxes for detected objects
        for box in boxes:
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            height = math.fabs(y1 - y2)
            # Crop the detected region from the original image
            cropped_image_1 = img_cv[y1:y2, x1:x2]
            cropped_image_2 = img_cv[(y1+10):(y2-10), (x1+10):(x2-10)]
            if(height > 90):
                plate_text_top = pytesseract.image_to_string(cropped_image_2[:cropped_image_2.shape[0] // 2],
                                                             config='--psm 8')
                plate_text_bottom = pytesseract.image_to_string(cropped_image_2[cropped_image_2.shape[0] // 2:],
                                                                config='--psm 8')

                # Filter out non-alphanumeric characters (keep only letters and digits)
                plate_text_top = re.sub(r'[^a-zA-Z0-9]', '', plate_text_top)
                plate_text_bottom = re.sub(r'[^a-zA-Z0-9]', '', plate_text_bottom)
                plate_text = plate_text_top + plate_text_bottom
                detected_plates.append((cropped_image_1, plate_text.strip()))
            else:
            # Perform OCR using pytesseract
                plate_text = pytesseract.image_to_string(cropped_image_1, config='--psm 8')
                plate_text = re.sub(r'[^a-zA-Z0-9]', '', plate_text)

            # Append detected license plate and OCR result to the list
                detected_plates.append((cropped_image_1, plate_text.strip()))

    return detected_plates


# Streamlit UI
st.title('License Plate Detection and OCR')

# Upload image
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform license plate detection and OCR
    detected_plates = detect_and_ocr(image)
    st.markdown("<h2 style='text-align: center;'>Result of Detection", unsafe_allow_html=True)
    # Display the results
    if detected_plates:
        for i, (plate_image, plate_text) in enumerate(detected_plates):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(plate_image, caption=f"Detected License Plate {i + 1}", use_column_width=True)
            with col2:
                st.subheader("License Number: " + plate_text)
    else:
        st.write('No license plates detected.')
