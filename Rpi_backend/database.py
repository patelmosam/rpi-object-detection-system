import sqlite3
# import utils
import datetime

def get_date_info():
    today = datetime.date.today()
    year, month, day = str(today).split('-')
    _, week, week_day = datetime.date(int(year), int(month), int(day)).isocalendar()

    day_format = day+'-'+month+'-'+year
    month_fromat = month+'-'+year
    week_format = str(week)+'-'+year

    return day_format, month_fromat, week_format

def Init_db(db_path):
    conn = sqlite3.connect(db_path)
    print("Opened database successfully")

    ''' Monthly: Month => mm-yyyy '''
    conn.execute('CREATE TABLE Monthly (Month TEXT, Num_Images INT, Space INT)')

    ''' Weekly: Week => wk-yyyy '''
    conn.execute('CREATE TABLE Weekly (Week TEXT, Num_Images INT, Space INT)')

    ''' Daily: Day => dd-mm-yyyy '''
    conn.execute('CREATE TABLE Daily (Day TEXT, Num_Images INT, Space INT)')
    print("Table created successfully")
    conn.close()

    day, month, week = get_date_info()
    Insert_Init(db_path, 'Daily', 'Day', day)
    Insert_Init(db_path, 'Monthly', 'Month', month)
    Insert_Init(db_path, 'Weekly', 'Week', week)    
    

def Insert_Init(db_path, Table, Col, value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO "+Table+"("+Col+", Num_Images, Space) VALUES (?,?,?)", (value, 0, 0))
    conn.commit()
    conn.close()

def insert_data(db_path, img_size):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    day, month, week = get_date_info()
    try:
        day_data = get_data_id(db_path, 'Daily', "Day", day)[0]
    except IndexError:
        Insert_Init(db_path, 'Daily', 'Day', day)
        day_data = get_data_id(db_path, 'Daily', "Day", day)[0]

    try:
        month_data = get_data_id(db_path, 'Monthly', "Month", month)[0]
    except IndexError:
        Insert_Init(db_path, 'Monthly', 'Month', month)
        month_data = get_data_id(db_path, 'Monthly', "Month", month)[0]

    try:
        week_data = get_data_id(db_path, 'Weekly', "Week", week)[0]
    except IndexError:
        Insert_Init(db_path, 'Weekly', 'Week', week)
        week_data = get_data_id(db_path, 'Weekly', "Week", week)[0]

    Num_Images, Space = str(day_data[1]+1), str(day_data[2] + img_size)
    c.execute("UPDATE Daily SET Num_Images = "+Num_Images+", Space = "+Space+" WHERE Day = '"+day+"'")
    
    Num_Images, Space = str(month_data[1]+1), str(month_data[2] + img_size)
    c.execute("UPDATE Monthly SET Num_Images = "+Num_Images+", Space = "+Space+" WHERE Month = '"+month+"'")
    
    Num_Images, Space = str(week_data[1]+1), str(week_data[2] + img_size)
    c.execute("UPDATE Weekly SET Num_Images = "+Num_Images+", Space = "+Space+" WHERE Week = '"+week+"'")

    conn.commit()

    print("added to database")
    conn.close()

def get_all_data(db_path, Table):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * From "+Table)

    row = c.fetchall()
    conn.close()
    return row

def get_data_id(db_path, table, field, value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * FROM "+table+" WHERE "+field+"=?", (value,))
    row = c.fetchall()
    conn.close()
    return row

def delete_data(db_path, Table, Col, value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DELETE FROM "+Table+" WHERE "+Col+"=?", (value,))
    conn.commit()
    conn.close()

def update_data_id(db_path, table, field, value_id, num_images, space):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("UPDATE "+table+" SET Num_Images = "+num_Images+", Space = "+space+" WHERE "+field+" = '"+value_id+"'")
    conn.commit()
    conn.close()

if __name__ == '__main__':
    DB_PATH = "./database.db"
    # Init_db(DB_PATH)
    # insert_data(DB_PATH, 127000)
    
