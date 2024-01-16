import sqlite3
from deepface import DeepFace
import scipy.spatial.distance
import pickle
from LockDB import LockDB
from datetime import datetime
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

lockDB = LockDB()



def add_employee(img_path, name):
    img_vector = DeepFace.represent(img_path=img_path, model_name=models[0])[0]["embedding"]
    lockDB.insert_employee(name, img_vector)

def verify_employee(img_path):
    img_vector = DeepFace.represent(img_path=img_path, model_name=models[0])[0]["embedding"]
    result = lockDB.find_nearest_employee(img_vector)
    id = result["id"]
    distance = result["distance"]
    name = result["name"]
    print(distance)
    print(name)
    if float(distance) < 0.65:
      lockDB.add_attendance(id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return id


#add_employee("face1.png", "name1")
id = verify_employee("face2.png")
print(lockDB.get_employee_attendance(id))
lockDB.close_connection()
