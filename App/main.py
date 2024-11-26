import streamlit as st
from database_services import LockDB
from deepface import DeepFace
import tempfile
import time
import cv2
import numpy as np
import os
from PIL import Image
import io

# Function to extract feature vector with handle for no face detection
def extract_feature_vector(image_path, model_name='VGG-Face'):
    try:
        # Setting enforce_detection=False will prevent the error if no face is detected
        embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    except Exception as e:
        raise ValueError(f"Error extracting feature vector: {e}")

# Initialize database connection
db = LockDB()

# Streamlit UI Title
st.title("Look Lock")

# Section: Live Camera Face Recognition
st.header("Live Face Recognition")
camera_active = st.checkbox("Activate Camera", value=False)

if camera_active:
    st.info("Camera is active. Detecting faces automatically every 3 seconds...")
    video_capture = cv2.VideoCapture(0)  # Open default camera

    # Create a placeholder for displaying camera image
    image_placeholder = st.empty()

    while camera_active:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Error accessing the camera. Please make sure it's connected.")
            break

        # Update the image in the placeholder to replace the old one
        image_placeholder.image(frame, channels="BGR", caption="Live Camera Feed")

        # Save current frame as a temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_cam_img:
            cv2.imwrite(temp_cam_img.name, frame)
            camera_image_path = temp_cam_img.name

        try:
            # Extract feature vector from the captured image
            feature_vector = extract_feature_vector(camera_image_path)

            # Find the nearest match in the database
            lookALikePerson = db.find_most_similar_person(feature_vector)

            if lookALikePerson:
                # Match found, retrieve person details from the database
                person_id = lookALikePerson['id']
                person_name = lookALikePerson['name']
                person_distance = lookALikePerson['distance']

                # Fetch the person's image from the database
                db.cursor.execute("SELECT person_image FROM persons WHERE id = ?", (person_id,))
                person_image_data = db.cursor.fetchone()[0]

                # Convert image data to an image
                person_image = Image.open(io.BytesIO(person_image_data))

                # Display the person's info in a "card" format
                st.subheader(f"Match Found: {person_name}")
                st.image(person_image, caption=f"ID: {person_id}", width=100)  # Display small image
                st.write(f"**ID:** {person_id}")
                st.write(f"**Name:** {person_name}")
                st.write(f"**Similarity:** {person_distance:.4f}")

            else:
                st.warning("No match found in the database.")

        except ValueError as e:
            st.error(f"Error processing the image: {e}")

        # Delete the temporary image after processing
        os.remove(camera_image_path)

        # Wait 3 seconds before capturing the next image
        time.sleep(3)

    video_capture.release()  # Release the camera when done

# Section: Add a New Person
st.header("Add a New Person")
name_input = st.text_input("Enter Full Name:")
person_image = st.file_uploader("Upload Person Image (.jpg only)", type=["jpg"])

if st.button("Add Person"):
    if not name_input or not person_image:
        st.error("Please provide both name and image.")
    else:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(person_image.read())
            temp_path = temp_img.name

        try:
            # Extract feature vector from the uploaded image
            feature_vector = extract_feature_vector(temp_path)

            # Read the image as binary data
            with open(temp_path, 'rb') as img_file:
                person_image_binary = img_file.read()

            # Insert the person into the database
            db.insert_person(name=name_input, face_feature_vector=feature_vector, person_image=person_image_binary)

            st.success(f"Person '{name_input}' added successfully!")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
# Close the database when Streamlit stops
db.close_connection()
