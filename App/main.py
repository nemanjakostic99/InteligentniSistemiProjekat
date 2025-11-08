import os
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
@st.cache_resource
def get_database():
    return LockDB()

db = get_database()

deepface_model = load_deepface_model()

selected_entrance_label = None

st.set_page_config(
    layout="wide"
)

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

st.markdown("#### Look Lock")

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
    global frame_queue, selected_entrance_id

    while camera_active:
        if not frame_queue.empty():

            frame = frame_queue.get()

            try:
                # Extract feature vector from the captured image
                frame_resized = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                feature_vector = extract_feature_vector(frame_rgb)

                # Find the nearest match in the database
                lookALikePerson = db.find_most_similar_person(feature_vector, selected_entrance_label)

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

    log_dir = os.path.dirname(log_filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    with open(log_filename, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_message}\n")


def CameraPage():
    # st.markdown("#### Live Face Recognition")
    global camera_active, selected_entrance_label

    # Fetch entrances from the database
    entrances = db.get_entrances()
    if not entrances:
        st.error("No entrances found in the database. Please add entrances first.")
        return

    # Let the user select an entrance
    selected_entrance = st.selectbox(
        "Select the Entrance this Camera is Guarding",
        options=entrances,
        format_func=lambda x: x['label'])

    selected_entrance_label = selected_entrance['label']

    camera_active = st.checkbox("Activate Camera", value=False)

    if 'results' not in st.session_state:
        st.session_state.results = deque(maxlen=5)

    if not selected_entrance:
        st.warning("Please select an entrance before activating the camera.")
        return

    if camera_active:
        # st.info(f"Camera is active. Guarding entrance: {selected_entrance['label']}")
        st.session_state.results = deque(maxlen=5)

        processing_thread = Thread(target=process_frame, daemon=True)
        processing_thread.start()

        video_capture = cv2.VideoCapture(0)

        # Create two columns: camera on left (60%), results on right (40%)
        col_camera, col_results = st.columns([3, 2])

        with col_camera:
            st.markdown("Live Feed")
            image_placeholder = st.empty()

        with col_results:
            st.markdown("Recent Results")
            results_placeholder = st.empty()

        while camera_active:
            ret, frame = video_capture.read()

            if not ret:
                st.error("Error accessing the camera. Please make sure it's connected.")
                break

            # Update camera feed
            image_placeholder.image(frame, channels="BGR", use_container_width=True)

            global frame_queue
            if frame_queue.empty():
                frame_queue.put(frame)

            # Check for processed results and update display
            if result_data["processed"]:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                if result_data["person_name"]:
                    result_entry = {
                        "timestamp": timestamp,
                        "type": "match",
                        "name": result_data['person_name'],
                        "id": result_data['person_id'],
                        "image": result_data["person_image"],
                        "distance": result_data['person_distance'],
                        "similarity": result_data['person_similarity'],
                        "permission": result_data['person_permission']
                    }

                    log_to_file(f"Match Found: {result_data['person_name']} (ID: {result_data['person_id']}, "
                                f"Permission: {result_data['person_permission']}, "
                                f"Similarity: {result_data['person_similarity']:.2f}%)")
                else:
                    result_entry = {
                        "timestamp": timestamp,
                        "type": "no_match"
                    }
                    log_to_file("No match found.")

                st.session_state.results.appendleft(result_entry)
                result_data["processed"] = False

            with results_placeholder.container():
                if st.session_state.results:
                    for idx, result in enumerate(st.session_state.results):

                        if result["type"] == "match":
                            col_img, col_info = st.columns([1, 2])

                            with col_img:
                                st.image(result["image"], width=100)

                            with col_info:
                                st.markdown(f"**{result['timestamp']} {result['name']}**")
                                st.caption(f"Similarity: {result['similarity']:.1f}%")

                                if result.get('permission') == 'allowed':
                                    st.success("ALLOWED")
                                elif result.get('permission'):
                                    st.error("NOT ALLOWED")
                        else:
                            st.warning(f"{result['timestamp']} No match found")
                else:
                    st.info("Waiting for recognition results...")

        video_capture.release()

# def CameraPage():
#     st.header("Live Face Recognition")
#     global camera_active, selected_entrance_id
#
#     # Fetch entrances from the database
#     entrances = db.get_entrances()
#     if not entrances:
#         st.error("No entrances found in the database. Please add entrances first.")
#         return
#
#     # Let the user select an entrance
#     selected_entrance = st.selectbox(
#         "Select the Entrance this Camera is Guarding",
#         options=entrances,
#         format_func=lambda x: x['label']  # Display a user-friendly label
#     )
#
#     selected_entrance_label = selected_entrance['label']
#
#     camera_active = st.checkbox("Activate Camera", value=False)
#
#     if not selected_entrance:
#         st.warning("Please select an entrance before activating the camera.")
#         return
#
#     if camera_active:
#         st.info(f"Camera is active. Guarding entrance: {selected_entrance['label']}")
#
#         # Start the frame processing thread
#         processing_thread = Thread(target=process_frame, daemon=True)
#         processing_thread.start()
#
#         video_capture = cv2.VideoCapture(0)  # Open the default camera
#
#         # Placeholder for displaying the camera feed
#         image_placeholder = st.empty()
#
#         with st.container():
#             while camera_active:
#                 ret, frame = video_capture.read()
#
#                 if not ret:
#                     st.error("Error accessing the camera. Please make sure it's connected.")
#                     break
#
#                 # Display the live camera feed
#                 image_placeholder.image(frame, channels="BGR", caption="Live Camera Feed", width=700)
#
#                 # Pass the frame for processing
#                 if frame_queue.empty():
#                     frame_queue.put(frame)
#
#                     if result_data["processed"]:
#                         if result_data["person_name"]:
#                             st.subheader(f"Match Found: {result_data['person_name']}")
#                             st.image(result_data["person_image"], caption=f"ID: {result_data['person_id']}", width=100)
#                             st.write(f"**ID:** {result_data['person_id']}")
#                             st.write(f"**Distance:** {result_data['person_distance']:.4f}")
#                             st.write(f"**Similarity:** {result_data['person_similarity']:.2f}%")
#
#                             if result_data['person_permission']:
#                                 if result_data['person_permission'] == 'allowed':
#                                     st.write(f"**Permission:** :green[{result_data['person_permission']}]")
#                                 else:
#                                     st.write(f"**Permission:** :red[{result_data['person_permission']}]")
#
#                             log_to_file(
#                                 f"Match Found: {result_data['person_name']} (ID: {result_data['person_id']}, "
#                                 f"Entrance: {selected_entrance['label']}, "
#                                 f"Permission: {result_data['person_permission']}, "
#                                 f"Similarity: {result_data['person_similarity']:.2f}%)"
#                             )
#                         else:
#                             st.warning("No match found in the database.")
#                             log_to_file(f"No match found at entrance: {selected_entrance['label']}.")
#
#         video_capture.release()  # Release the camera when done

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


    if st.button("Add Permission"):
        if not label_input or not description_input:
            st.error("Please provide both label and description.")
        else:
            try:
                db.insert_permission(label_input, description_input)
                st.success(f"Permission added successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding permission: {e}")

    permissions = db.get_permissions()
    if not permissions:
        st.info("No permissions available. Add one above to get started.")
        return

    st.header("Update Permission")

    selected_permission = st.selectbox(
        "Select a Permission",
        options=permissions,
        format_func=lambda x: x['label'],
        key="update_select"
    )
    if selected_permission:
        label_update = st.text_input("Update Label:", value=selected_permission['label'])
        description_update = st.text_input("Update Description:", value=selected_permission['description'])
        if st.button("Update Permission"):
            if not label_update or not description_update:
                st.error("Both label and description are required")
            else:
                try:
                    db.update_permission(selected_permission["id"], label_update, description_update)
                    st.session_state["update_success"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating permission: {e}")

    if st.session_state.get("update_success"):
        st.success("Permission updated!")
        del st.session_state["update_success"]

    st.header("Delete Permission")

    permission_to_delete = st.selectbox(
        "Select a Permission",
        options=permissions,
        format_func=lambda x: x['label'],
        key="delete_select"
    )
    if permission_to_delete:
        if st.button("Delete Permission", type="primary"):
            try:
                db.delete_permission(permission_to_delete["id"])
                st.session_state["delete_success"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting permission: {e}")

    if st.session_state.get("delete_success"):
        st.success("Permission deleted!")
        del st.session_state["delete_success"]


pg = st.navigation(pages=[
    st.Page(CameraPage, title="Camera Feed"),
    st.Page(AddPersonPage, title="Add Person"),
    st.Page(Permissions, title="Permissions")],
    position="top")
pg.run()

