import time

import streamlit as st
from database_services import LockDB
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import io
from threading import Thread
import queue
import tempfile
import datetime
from collections import deque


@st.cache_resource
def load_deepface_model():
    return DeepFace.build_model("VGG-Face")


deepface_model = load_deepface_model()


# Function to extract feature vector with handle for no face detection
def extract_feature_vector(image_path, model_name='VGG-Face'):
    try:
        if isinstance(image_path, str) or isinstance(image_path, np.ndarray):
            # Setting enforce_detection=False will prevent the error if no face is detected
            embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
        elif isinstance(image_path, io.BytesIO):
            image = Image.open(image_path)
            image_np = np.array(image)
            embedding = DeepFace.represent(img_path=image_np, model_name=model_name, enforce_detection=False)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_path)}")

        return np.array(embedding[0]['embedding'])

    except Exception as e:
        raise ValueError(f"Error extracting feature vector: {e}")


# Initialize database connection
db = LockDB()

# Streamlit UI Title
st.title("Look Lock")

# Section: Live Camera Face Recognition
camera_active = False

frame_queue = queue.Queue(maxsize=1)
result_data = {
    "processed": False,
    "person_name": None,
    "person_id": None,
    "person_distance": None,
    "person_image": None,
    "person_similarity": None,
    "person_permission":None
}


def process_frame():
    global frame_queue

    while camera_active:
        if not frame_queue.empty():

            frame = frame_queue.get()

            try:
                # Extract feature vector from the captured image
                frame_resized = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                feature_vector = extract_feature_vector(frame_rgb)

                # Find the nearest match in the database
                lookALikePerson = db.find_most_similar_person(feature_vector)

                if lookALikePerson:
                    # Match found, retrieve person details from the database
                    person_id = lookALikePerson['id']
                    person_name = lookALikePerson['name']
                    person_distance = lookALikePerson['distance']
                    person_similarity = lookALikePerson['similarity']
                    person_permission = lookALikePerson['permission']

                    # Fetch the person's image from the database
                    db.cursor.execute("SELECT person_image FROM persons WHERE id = ?", (person_id,))
                    person_image_data = db.cursor.fetchone()[0]

                    # Convert image data to an image
                    person_image = Image.open(io.BytesIO(person_image_data))

                    result_data.update({
                        "processed": True,
                        "person_name": person_name,
                        "person_id": person_id,
                        "person_distance": person_distance,
                        "person_image": person_image,
                        "person_similarity": person_similarity,
                        "person_permission": person_permission
                    })

                else:
                    result_data.update({
                        "processed": True,
                        "person_name": None,
                        "person_id": None,
                        "person_distance": None,
                        "person_image": None,
                        "person_similarity": None,
                        "person_permission": None
                    })

            except ValueError as e:
                st.error(f"Error processing the image: {e}")

        time.sleep(2)

def log_to_file(log_message):
    today_date = datetime.datetime.now().strftime("%d-%m-%Y")
    log_filename = f"./App/logs/{today_date}.txt"
    with open(log_filename, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_message}\n")


def CameraPage():
    st.header("Live Face Recognition")
    global camera_active

    camera_active = st.checkbox("Activate Camera", value=False)

    results = deque(maxlen=5)

    if camera_active:
        st.info("Camera is active. Detecting faces automatically every 2 seconds...")

        processing_thread = Thread(target=process_frame, daemon=True)
        processing_thread.start()

        video_capture = cv2.VideoCapture(0)  # Open default camera

        # Create a placeholder for displaying camera image
        image_placeholder = st.empty()
        placeholder = st.empty()
        content = []
        with st.container(height=500):
            while camera_active:
                ret, frame = video_capture.read()

                if not ret:
                    st.error("Error accessing the camera. Please make sure it's connected.")
                    break

                # Update the image in the placeholder to replace the old one
                image_placeholder.image(frame, channels="BGR", caption="Live Camera Feed", width=700)
                global frame_queue
                if frame_queue.empty():
                    frame_queue.put(frame)

                    if result_data["processed"]:

                        if result_data["person_name"]:
                            st.subheader(f"Match Found: {result_data['person_name']}")
                            st.image(result_data["person_image"], caption=f"ID: {result_data['person_id']}", width=100)
                            st.write(f"**ID:** {result_data['person_id']}")
                            st.write(f"**Distance:** {result_data['person_distance']:.4f}")
                            st.write(f"**Similarity:** {result_data['person_similarity']:.2f}%")

                            if result_data['person_permission']:
                                if result_data['person_permission'] == 'allowed':
                                    st.write(f"**Permission**: :green[{result_data['person_permission']}]")
                                else:
                                    st.write(f"**Permission**: :red[{result_data['person_permission']}]")

                            log_to_file(f"Match Found: {result_data['person_name']} (ID: {result_data['person_id']}, "
                                        f"Permission: {result_data['person_permission']}, "
                                        f"Similarity: {result_data['person_similarity']:.2f}%)")

                        else:
                            st.warning(f"No match found in the database.")
                            log_to_file("No match found.")

        video_capture.release()  # Release the camera when done


def AddPersonPage():
    # Section: Add a New Person
    st.header("Add a New Person")
    name_input = st.text_input("Enter Full Name:")
    person_image = st.file_uploader("Upload Person Image (.jpg only)", type=["jpg"])

    entrances = db.get_entrances()
    selected_entrance = st.selectbox(
        "Select an Entrance",
        options=entrances,
        format_func=lambda x: x['label']
    )

    permissions = db.get_permissions()
    selected_permission = st.selectbox(
        "Select a Permission",
        options=permissions,
        format_func=lambda x: x['label']
    )

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
                db.insert_person(
                    name=name_input,
                    face_feature_vector=feature_vector,
                    person_image=person_image_binary,
                    entrance_id=selected_entrance['id'],
                    permission_id=selected_permission['id']
                )

                st.success(f"Person '{name_input}' added successfully!")
            except Exception as e:
                st.error(f"Error processing the image: {e}")


def Permissions():
    st.header("Add Permission")
    label_input = st.text_input("Enter Label:")
    description_input = st.text_input("Enter Description:")
    permissions = db.get_permissions()

    if st.button("Add Permission"):
        if not label_input or not description_input:
            st.error("Please provide both label and description.")
        else:
            try:
                db.insert_permission(label_input, description_input)
                st.success(f"Permission added successfully!")
                permissions = db.get_permissions()
            except Exception as e:
                st.error(f"Error adding permission: {e}")

    selected_permission = st.selectbox(
        "Select a Permission",
        options=permissions,
        format_func=lambda x: x['label']
    )
    label_update = st.text_input("Update Label:", value=selected_permission['label'])
    description_update = st.text_input("Update Description:", value=selected_permission['description'])
    if st.button("Update Permission"):
        if not label_update and not description_update:
            st.error("To delete a permission, use the 'Delete Permission' button")
        else:
            try:
                db.update_permission(selected_permission["id"], label_update,description_update)
                st.success(f"Permission updated successfully!")
            except Exception as e:
                st.error(f"Error updating permission: {e}")


pg = st.navigation([
    st.Page(CameraPage, title="Camera Feed"),
    st.Page(AddPersonPage, title="Add Person"),
    st.Page(Permissions, title="Permissions")
])
pg.run()

db.close_connection()
