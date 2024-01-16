
import sqlite3
import scipy.spatial.distance
import pickle

class LockDB:
    def __init__(self, db_name='employee_database.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.init_db()

    def init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                face_feature_vector BLOB
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY,
                employee_id INTEGER,
                date TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        ''')

        self.conn.commit()

    def insert_employee(self, name, face_feature_vector):
        self.cursor.execute("INSERT INTO employees (name, face_feature_vector) VALUES (?, ?)", (name, pickle.dumps(face_feature_vector)))
        self.conn.commit()

    def add_attendance(self, employee_id, date):
        self.cursor.execute("INSERT INTO attendance (employee_id, date) VALUES (?, ?)", (employee_id, date))
        self.conn.commit()

    def get_employee_attendance(self, employee_id):
        self.cursor.execute("SELECT date FROM attendance WHERE employee_id = ?", (employee_id,))
        attendance_dates = self.cursor.fetchall()
        return [date[0] for date in attendance_dates]   
         
    def find_nearest_employee(self, query_vector):
        self.cursor.execute("SELECT id, name, face_feature_vector FROM employees")
        rows = self.cursor.fetchall()

        nearest_employee = None
        min_distance = float('inf')

        for row in rows:
            employee_id, name, stored_vector = row
            distance = scipy.spatial.distance.cosine(query_vector, pickle.loads(stored_vector))

            if distance < min_distance:
                min_distance = distance
                nearest_employee = {"id": employee_id, "name": name, "distance": distance}

        return nearest_employee

    def close_connection(self):
        self.conn.close()
