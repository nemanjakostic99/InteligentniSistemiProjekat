import streamlit as st
from App.Database.database_services import LockDB
#from App.DeepFace.deepface_services import 
import cv2
import numpy as np


def main():
    st.title("LookLock")

    # Text Input for Name
    name = st.text_input("Enter name of wanted person:")

    # Camera Feed
    st.subheader("Camera Feed")
    
    # OpenCV Camera Capture (Assuming camera is at index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open camera.")
    else:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Display the camera feed in the app
        st.image(frame, channels="BGR")

    # Release the camera when the app is closed
    st.button("Close Camera", on_click=cap.release, args=())

    # Display entered name
    if name:
        st.write(f"Hello, {name}!")

if __name__ == "__main__":
    main()
