import faiss
import numpy as np
import pyodbc
import pickle
import streamlit as st

class LockDB:
    def __init__(self, db_name='looklock_database', server='(localdb)\looklock_database', username='my_user', password='12345'):
        # SQL Server connection string
        self.db_name = db_name
        self.conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                                   f'SERVER={server};'
                                   f'DATABASE={db_name};'
                                   f'UID={username};'
                                   f'PWD={password}')
        self.cursor = self.conn.cursor()

        # Initialize Faiss index (for cosine similarity, we will use inner product indexing)
        self.index = None
        self.id_map = {}  # Maps Faiss IDs to person IDs
        self.feature_dimension = 4096  # Dimension of the face feature vectors (VGG-Face produces 512-dimensional vectors)
        
        # Load existing feature vectors from the database into the Faiss index
        self.load_existing_vectors()
        
        self.init_db()

    def init_db(self):
        # Creating tables if they do not exist (already implemented)
        # Create tables for persons, entrances, attendance, and permissions.
        self.cursor.execute(''' 
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='persons' AND xtype='U')
            CREATE TABLE persons (
                id INT PRIMARY KEY IDENTITY(1,1),
                name NVARCHAR(255) NOT NULL,
                person_image VARBINARY(MAX),
                face_feature_vector VARBINARY(MAX)
            )
        ''')

        self.cursor.execute(''' 
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='entrances' AND xtype='U')
            CREATE TABLE entrances (
                id INT PRIMARY KEY IDENTITY(1,1),
                label NVARCHAR(255) NOT NULL,
                description NVARCHAR(255),
                permitted BIT
            )
        ''')

        self.cursor.execute(''' 
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='attendance' AND xtype='U')
            CREATE TABLE attendance (
                id INT PRIMARY KEY IDENTITY(1,1),
                person_id INT,
                date NVARCHAR(50) NOT NULL,
                permitted BIT,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')

        self.cursor.execute(''' 
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='entrances_permissions' AND xtype='U')
            CREATE TABLE entrances_permissions (
                id INT PRIMARY KEY IDENTITY(1,1),
                person_id INT,
                entrance_id INT,
                FOREIGN KEY (person_id) REFERENCES persons(id),
                FOREIGN KEY (entrance_id) REFERENCES entrances(id)
            )
        ''')

        self.conn.commit()

    def load_existing_vectors(self):
        """
        Load all existing face feature vectors from the database into the Faiss index.
        """
        self.cursor.execute("SELECT id, face_feature_vector FROM persons")
        rows = self.cursor.fetchall()

        vectors = []
        for row in rows:
            person_id, stored_vector = row
            feature_vector = np.array(pickle.loads(stored_vector))  # Convert byte stream back to numpy array

            if feature_vector.shape[0] != self.feature_dimension:
                st.error(f"Feature vector for person ID {person_id} has incorrect dimensions: {feature_vector.shape}. Skipping.")
                continue

            vectors.append(feature_vector)
            self.id_map[len(vectors) - 1] = person_id  # Map Faiss index to person ID

        # Check if vectors list is empty
        if len(vectors) == 0:
            st.warning("No valid feature vectors found in the database. Initializing an empty Faiss index.")
            self.index = faiss.IndexFlatL2(self.feature_dimension)
            return

        # Convert list to a numpy array of shape (n, d)
        vectors = np.vstack(vectors).astype('float32')  # Stack the vectors to create a (n, d) matrix

        # Log the vector and index dimensions
        st.info(f"Feature vectors shape: {vectors.shape}")
        st.info(f"Initializing Faiss index with dimension: {self.feature_dimension}")

        # Initialize the Faiss index
        self.index = faiss.IndexFlatL2(self.feature_dimension)

        # Add the vectors to the Faiss index
        if vectors.shape[1] == self.feature_dimension:
            self.index.add(vectors)
        else:
            st.error(f"Cannot add vectors to Faiss index. Expected dimension {self.feature_dimension}, got {vectors.shape[1]}.")

    def insert_person(self, name, face_feature_vector, person_image):
        """
        Insert a new person into the database and the Faiss index.
        """
        # Save the feature vector and insert the person into the database
        self.cursor.execute('''INSERT INTO persons (name, person_image, face_feature_vector)
                               VALUES (?, ?, ?)''', (name, person_image, pickle.dumps(face_feature_vector)))
        self.conn.commit()

        # Insert the vector into the Faiss index
        faiss_index = len(self.id_map)  # Generate a new index for Faiss
        self.id_map[faiss_index] = name  # Store the person name using the Faiss index
        feature_vector = np.array(face_feature_vector).astype('float32').reshape(1, -1)  # Reshape to 2D array
        self.index.add(feature_vector)  # Add to Faiss index

    def find_most_similar_person(self, query_vector, k=1, threshold=0.6):
        """
        Find the most similar person based on facial feature similarity.
        """
        # Ensure the query_vector is in the correct shape (1, 512)
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)  # Reshape to (1, d)
        
        if query_vector.shape[1] != self.feature_dimension:
            raise ValueError(f"Query vector has incorrect dimensions: {query_vector.shape[1]}. Expected {self.feature_dimension}.")

        # Perform the search in Faiss
        distances, indices = self.index.search(query_vector, k)  # k = number of nearest neighbors to return

        if distances[0][0] > threshold:
            return None 
        
        if indices[0][0] != -1:
            person_id = self.id_map[indices[0][0]]
            return {"id": person_id, "name": self.id_map[indices[0][0]], "distance": distances[0][0]}
        else:
            return None

    def close_connection(self):
        # Close the database connection
        self.conn.close()
