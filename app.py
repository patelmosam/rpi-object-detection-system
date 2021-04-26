from flask import Flask
from flask import request, jsonify, render_template, redirect, send_file
import numpy as np
import requests
import database as db
import json
import utils

config = utils.get_config("app.config")

app = Flask(__name__, template_folder='template')
connection = False

@app.route('/', methods=['GET'])
def home():
	global connection
	try:
		r = requests.get(config["HOST_ADDR"])
	
		if r.status_code == 200:
			connection = True
			return render_template('home.html', connection=connection)
	except:
		connection = False
		return render_template('home.html', connection=connection)
	
@app.route('/capture', methods=['GET'])
def capture():
	global connection
	response = requests.get(config["HOST_ADDR"] + "/api/capture")

	if response.status_code != 200:
		connection = False
		return render_template('capture.html', isData=False, connection=connection)
	
	data = response.json()
	data = json.loads(data)
	img_id = data['image_id']

	predictions = [np.array([data['boxes']]), np.array([data['scores']]), np.array([data['classes']]), np.array([int(data['num_det'])])]
	
	img_url = config["HOST_ADDR"] + "/api/img/"+str(img_id)
	try:
		r = requests.get(img_url)
		
		file = open('images/'+img_id+'.jpg', 'wb')
		file.write(r.content)
		file.close()
	except:
		pass

	if data['num_det'] != "0":
		utils.make_bbox(predictions, img_id)
		bbox, classes, scores = utils.convert_data(data)
		classes_name = utils.get_class_names(data['classes'])

		db.insert_data(config["DB_PATH"], (img_id, bbox, classes_name, scores, data['num_det']))


	data['classes'] = utils.get_class_names(data['classes'])
	data['boxes'] = np.round(data['boxes'], 3)
	data['scores'] = np.round(data['scores'], 3)
	filename = '/img/'+str(img_id)

	return render_template('capture.html', data=data, filename=filename, connection=connection)
	
@app.route('/stream')
def stream():
	"""Video streaming home page."""
	global connection
	video_url = config["HOST_ADDR"] + "/video_feed"
	return render_template('stream.html', video_url=video_url, connection=connection)

@app.route('/history')
def get_history():
	global connection
	try:
		conn_statues = requests.get(config["HOST_ADDR"] + "/") 
		if conn_statues.status_code == 200:
			connection = True
	except:
		connection = False
	try:
		response = requests.get(config["HOST_ADDR"] + "/release_cam")
	except:
		pass

	data = db.get_all_data(config["DB_PATH"])
	return render_template('history.html', data=data, connection=connection)

@app.route('/his_result/<img_id>')
def get_his_data(img_id):
	global connection
	
	data = db.get_data_id(config["DB_PATH"], img_id)

	filename = config['SELF_ADDR']+"/img/"+str(img_id)

	return render_template('his_result.html', data=data[0], filename=filename, connection=connection)

@app.route('/img/<img_id>', methods=['GET'])
def get_img(img_id):
	filename = './images/' + str(img_id) + '.jpg'
	return send_file(filename, mimetype='image/gif')


@app.route('/delete/<img_id>', methods=['GET'])
def delete(img_id):
	
	db.delete_data(config["DB_PATH"], img_id)
	return redirect('/history')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
	global connection

	if request.method == 'POST':
		host_add = request.form.get('host_add')
		db_path = request.form.get('db_path')
		
		config['HOST_ADDR'] = host_add
		config['DB_PATH'] = db_path
		utils.set_config("app.config", config)
		return redirect('/settings')
	else:
		return render_template('settings.html', data=config, connection=connection)

if __name__ == '__main__':
    app.run(port=8000, debug=True, host='0.0.0.0')
    
