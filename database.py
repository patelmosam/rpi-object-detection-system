import sqlite3

def Init_db(db_path):
    conn = sqlite3.connect(db_path)
    print("Opened database successfully")

    conn.execute('CREATE TABLE Detections (image_id TEXT, bbox TEXT, classes TEXT, scores TEXT, num_det TEXT)')
    print("Table created successfully")
    conn.close()


def insert_data(db_path, data):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("INSERT INTO Detections(image_id, bbox, classes, scores, num_det) VALUES (?,?,?,?,?)", data)

    conn.commit()

    print("added to database")
    conn.close()

def get_all_data(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * From Detections")

    row = c.fetchall()
    conn.close()
    return row

def get_data_id(db_path, image_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * FROM Detections WHERE image_id=?", (image_id,))
    row = c.fetchall()
    conn.close()
    return row

def delete_data(db_path, image_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DELETE FROM Detections WHERE image_id=?", (image_id,))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    DB_PATH = "./database.db"
    # Init_db(DB_PATH)
