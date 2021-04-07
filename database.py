import sqlite3

def Init_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")

    conn.execute('CREATE TABLE Detections (image_id TEXT, bbox TEXT, classes TEXT, score TEXT, no_det TEXT)')
    print("Table created successfully")
    conn.close()


def insert_data(conn, image_id, bbox, classes, scores, no_det):
    c = conn.cursor()

    c.execute("INSERT INTO Detections VALUES (\""+ image_id +"\",\""+ bbox +"\",\""+ classes +"\",\""+ scores +"\",\""+ no_det +"\")")

    conn.commit()

    print("added to database")
    conn.close()

def get_all_data(conn):
    c = conn.cursor()

    c.execute("SELECT * From Detections")

    row = c.fetchall()

    return row

def get_data_id(conn, img_id):
    c = conn.cursor()

    c.execute("SELECT * FROM Detections WHERE image_id=?", (img_id,))
    row = c.fetchall()

    return row

if __name__ == '__main__':
    #Init_db()
    conn = sqlite3.connect('database.db')

    # insert_data(conn, "0", "[12,23,45,67]", "[1]", "[0.99]", "1")

    # get_all_data(conn)

    get_data_id(conn, "2021-04-07-21-46-22")
