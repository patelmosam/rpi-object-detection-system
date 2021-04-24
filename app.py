from flask import Flask
from flask import request, jsonify, render_template, redirect, send_file
import numpy as np
import requests
import database as db
import json
import utils

HOST_NAME = "http://127.0.0.1:5000"
# HOST_NAME = "http://192.168.108.55:8080"

DB_PATH = "./database.db"

app = Flask(__name__, template_folder='template')
connection = False

@app.route('/', methods=['GET'])
def home():
	global connection
	try:
		r = requests.get(HOST_NAME)
	
		if r.status_code == 200:
			connection = True
			return render_template('home.html', connection=connection)
	except:
		connection = False
		return render_template('home.html', connection=connection)
	
@app.route('/capture', methods=['GET'])
def capture():
	global connection
	response = requests.get(HOST_NAME + "/api/capture")
	# print(response)
	if response.status_code != 200:
		connection = False
		return render_template('capture.html', isData=False, connection=connection)
	
	data = response.json()
	data = json.loads(data)
	id = data['image_id']

	predictions = [np.array([data['boxes']]), np.array([data['scores']]), np.array([data['classes']]), np.array([int(data['num_det'])])]
	# print(predictions)

		
	# TODO add try catch	
	img_url = HOST_NAME + "/api/img/"+str(id)
	# print(img_url)

	r = requests.get(img_url)
	
	file = open('images/'+id+'.jpg', 'wb')
	file.write(r.content)
	file.close()

	if data['num_det'] != "0":
		# print("adding to database")
		utils.make_bbox(predictions, id)
		bbox, classes, scores = utils.convert_data(data)
		classes_name = utils.get_class_names(data['classes'])

		db.insert_data(DB_PATH, (id, bbox, classes_name, scores, data['num_det']))


	data['classes'] = utils.get_class_names(data['classes'])
	data['boxes'] = np.round(data['boxes'], 3)
	data['scores'] = np.round(data['scores'], 3)
	filename = '/img/'+str(id)

	return render_template('capture.html', data=data, filename=filename, connection=connection)
	
@app.route('/stream')
def stream():
	"""Video streaming home page."""
	global connection
	video_url = HOST_NAME + "/video_feed"
	return render_template('stream.html', video_url=video_url, connection=connection)

@app.route('/history')
def get_history():
	global connection
	try:
		conn_statues = requests.get(HOST_NAME + "/") 
		if conn_statues.status_code == 200:
			connection = True
	except:
		connection = False
	try:
		response = requests.get(HOST_NAME + "/release_cam")
	except:
		pass

	data = db.get_all_data(DB_PATH)
	return render_template('history.html', data=data, connection=connection)

@app.route('/his_result/<id>')
def get_his_data(id):
	global connection
	
	data = db.get_data_id(DB_PATH, id)
	# print(type(data))
	filename = "http://127.0.0.1:8000/img/"+str(id)

	return render_template('his_result.html', data=data[0], filename=filename, connection=connection)

@app.route('/img/<id>', methods=['GET'])
def get_img(id):
	filename = './images/' + str(id) + '.jpg'
	return send_file(filename, mimetype='image/gif')


@app.route('/delete/<id>', methods=['GET'])
def delete(id):
	
	db.delete_data(DB_PATH, id)
	return redirect('/history')

if __name__ == '__main__':
    app.run(port=8000, debug=True, host='0.0.0.0')
    
