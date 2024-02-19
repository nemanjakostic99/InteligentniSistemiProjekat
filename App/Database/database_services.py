
import sqlite3
import scipy.spatial.distance
import pickle

class LockDB:
    def __init__(self, db_name='person_database.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.init_db()

    def init_db(self):
        # Tabela za osobe, sadrzi ime i prezime, sliku osobe (samo za vizualni prikaz osobe kad je potrebno, mozda i ne treba),
        # face_vector (kada se dodaje osoba, generise se face_vector na osnovi slike ili slike zabelezene od kamere i stavlja u bazu)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                person_image BLOB,
                face_feature_vector TEXT
            )
        ''')
        # Tabela za ulaze, pamti samo id ulaza, ime ulaza (tipa zadnja vrata, biblioteka itd), neki opis ako je potreban
        # permitted - da li postoje white lista za ovaj ulaz, ako postoji onda treba proveriti tabelu antrances_permissions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entrances (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT,
                permitted BOOL
            )
        ''')
        # Tabela za log fajlove o prisustvima, ko je uso? kad je uso? gde je hteo da udje? da li je bio dozvoljen? (ako je bio onda je usao)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY,
                person_id INTEGER,
                date TEXT NOT NULL,
                permitted BOOL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')
        # Tabela za spiskove dozvola za odredjene ulaze
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entrances_permissions (
                id INTEGER PRIMARY KEY,
                person_id INTEGER,
                entrance_id INTEGER,
                FOREIGN KEY (person_id) REFERENCES persons(id)
                FOREIGN KEY (entrances_id) REFERENCES entrances(id)
            )
        ''')

        self.conn.commit()

    def insert_person(self, name, face_feature_vector, person_image):
        self.cursor.execute("INSERT INTO persons (name, person_image, face_feature_vector) VALUES (?, ?, ?)", (name, person_image, face_feature_vector))
        self.conn.commit()

    def add_attendance(self, person_id, date, entrances_id):
        self.cursor.execute("INSERT INTO attendance (person_id, date, entrances_id) VALUES (?, ?, ?)", (person_id, date, entrances_id))
        self.conn.commit()

    def get_person_attendance(self, person_id):
        self.cursor.execute("SELECT date FROM attendance WHERE person_id = ?", (person_id,))
        attendance_dates = self.cursor.fetchall()
        return [date[0] for date in attendance_dates]   
    
    def get_person_attendance_entrance(self, person_id, entrance_id):
        self.cursor.execute("SELECT date FROM attendance WHERE person_id = ? AND entrance = ?", (person_id, entrance_id))
        attendance_dates = self.cursor.fetchall()
        return [date[0] for date in attendance_dates]   
         
    def find_nearest_person(self, query_vector):
        self.cursor.execute("SELECT id, name, face_feature_vector FROM persons")
        rows = self.cursor.fetchall()

        nearest_person = None
        min_distance = float('inf')

        for row in rows:
            person_id, name, stored_vector = row
            distance = scipy.spatial.distance.cosine(query_vector, pickle.loads(stored_vector))

            if distance < min_distance:
                min_distance = distance
                nearest_person = {"id": person_id, "name": name, "distance": distance}

        return nearest_person
    
    # Zamisao je da uvek u aplikaciji imas neki broj baferovanih ljudi (1000 npr) radi brzeg pronalaska
    def get_most_common_persons(self, limit=1000):
    # Execute SQL query to get the 1000 most common persons based on attendances
        query = '''
        SELECT persons.*
        FROM persons
        LEFT JOIN attendance ON persons.id = attendance.person_id
        GROUP BY persons.id
        ORDER BY COUNT(attendance.id) DESC
        LIMIT ?
        '''
        self.cursor.execute(query, (limit,))
    
        # Fetch the result
        top_attended_persons = self.cursor.fetchall()

        # Return the list of persons
        return top_attended_persons
        
    def close_connection(self):
        self.conn.close()
