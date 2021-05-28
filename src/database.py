import sqlite3
from pathlib import Path


def Check_db(db_path):
	""" 
	This function checks if the database is available in the specified path or not. If database
	is not available then it will create the new database.

	params:- db_path : <str>

	returns:- None
	"""
	dbpath = Path(db_path)

	if not dbpath.exists():
		Init_db(db_path, "Detections")
		Init_db(db_path, "auto_data")
		print('Database Created')


def Init_db(db_path, Table):
	""" 
	This function creates the table in database with predfined columns. It will take the table name
	as parameter and crates the table with that name with predfined columns.

	params:- db_path: <str>
			 table: <str>

	returns:- None
	"""
	conn = sqlite3.connect(db_path)
	print("Opened database successfully")

	conn.execute('CREATE TABLE '+Table+' (image_id TEXT, bbox TEXT, classes TEXT, scores TEXT, num_det TEXT)')
	print("Table created successfully")
	conn.close()


def insert_data(db_path, Table, data):
	""" 
	This function is responsible for inserting the data into database table specified in the parameters.

	params:- db_path: <str>
			 table: <str>
			 data: <tuple> or <list>

	returns:- None
	"""
	conn = sqlite3.connect(db_path)
	c = conn.cursor()

	c.execute("INSERT INTO "+Table+"(image_id, bbox, classes, scores, num_det) VALUES (?,?,?,?,?)", data)

	conn.commit()

	print("added to database")
	conn.close()


def get_all_data(db_path, Table):
	""" 
	This function fetches and returns the all data from the database table specified in the parameters.

	params:- db_path: <str>
			 table: <str>

	returns:- row: <tuple>
	"""
	conn = sqlite3.connect(db_path)
	c = conn.cursor()

	c.execute("SELECT * From "+Table+"")

	row = c.fetchall()
	conn.close()
	return row


def get_data_id(db_path, Table, image_id):
	""" 
	This function fetches and return the data row from database table that matches the value of 
	'image_id' column with the parameter 'image_id' value.

	params:- db_path: <str>
			 Table: <str>
			 image_id: <str>

	returns:- row: <tuple>
	"""
	conn = sqlite3.connect(db_path)
	c = conn.cursor()

	c.execute("SELECT * FROM "+Table+" WHERE image_id=?", (image_id,))
	row = c.fetchall()
	conn.close()
	return row


def delete_data(db_path, Table, image_id):
	""" 
	This function deletes the row in the database table(specified in the parameters) which match
	with the image_id.

	params:- db_path: <str>
			 Table: <str>
			 image_id: <str>

	returns:- None      
	"""
	conn = sqlite3.connect(db_path)
	c = conn.cursor()

	c.execute("DELETE FROM "+Table+" WHERE image_id=?", (image_id,))
	conn.commit()
	conn.close()


if __name__ == '__main__':
	DB_PATH = "./database.db"
	
