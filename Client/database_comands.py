import pyodbc
from datetime import datetime
from database_comands import get_logs_by_criteria



class Log:
    def __init__(self, image_path, name, door_id, in_or_out, data_taken=None):
        self.image_path = image_path
        self.name = name
        self.data_taken = data_taken or datetime.now()
        self.door_id = door_id
        self.in_or_out = in_or_out

def insert_log_to_database(log):
    connection_string = 'DRIVER={SQL Server};SERVER=localhost;DATABASE=looklock_logs;Trusted_Connection=yes;'
    # Adjust the connection string based on your authentication method and server details.

    # Establish a connection
    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()

    # Insert log into the 'log' table
    cursor.execute('''
        INSERT INTO log (image_path, name, data_taken, door_id, in_or_out)
        VALUES (?, ?, ?, ?, ?)
    ''', log.image_path, log.name, log.data_taken, log.door_id, log.in_or_out)

    # Commit the transaction and close the connection
    connection.commit()
    connection.close()

def get_logs_by_criteria(name=None, date=None, door_id=None):
    connection_string = 'DRIVER={SQL Server};SERVER=localhost;DATABASE=looklock_logs;Trusted_Connection=yes;'
    # Adjust the connection string based on your authentication method and server details.

    # Establish a connection
    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()

    # Build the SQL query based on the provided criteria
    query = "SELECT * FROM log WHERE 1=1"
    params = []

    if name:
        query += " AND name = ?"
        params.append(name)

    if date:
        query += " AND data_taken = ?"
        params.append(date)

    if door_id:
        query += " AND door_id = ?"
        params.append(door_id)

    # Execute the query
    cursor.execute(query, *params)
    rows = cursor.fetchall()

    # Convert query results to Log objects
    logs = [Log(*row) for row in rows]

    # Close the connection
    connection.close()

    return logs
