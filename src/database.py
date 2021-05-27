import sqlite3
from pathlib import Path

def Check_db(db_path):
    dbpath = Path(db_path)

    if not dbpath.exists():
        Init_db(db_path, "Detections")
        Init_db(db_path, "auto_data")
        print('Database Created')

def Init_db(db_path, Table):
    conn = sqlite3.connect(db_path)
    print("Opened database successfully")

    conn.execute('CREATE TABLE '+Table+' (image_id TEXT, bbox TEXT, classes TEXT, scores TEXT, num_det TEXT)')
    print("Table created successfully")
    conn.close()


def insert_data(db_path, Table, data):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("INSERT INTO "+Table+"(image_id, bbox, classes, scores, num_det) VALUES (?,?,?,?,?)", data)

    conn.commit()

    print("added to database")
    conn.close()

def get_all_data(db_path, Table):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * From "+Table+"")

    row = c.fetchall()
    conn.close()
    return row

def get_data_id(db_path, Table, image_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * FROM "+Table+" WHERE image_id=?", (image_id,))
    row = c.fetchall()
    conn.close()
    return row

def delete_data(db_path, Table, image_id):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DELETE FROM "+Table+" WHERE image_id=?", (image_id,))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    DB_PATH = "./database.db"
    
