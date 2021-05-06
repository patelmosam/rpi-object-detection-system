from flask import Flask
from flask import request, jsonify, render_template, redirect, send_file
import numpy as np
import requests
import database as db
import json
import utils
import pickle
from PIL import Image

CONFIG = utils.get_config("app.config")

app = Flask(__name__, template_folder='template')
conn = False

@app.route('/', methods=['GET'])
def home():
	global conn
	try:
		conn_statues = requests.get(CONFIG["HOST_ADDR"])
	
		if conn_statues.status_code == 200:
			conn = True
			req = requests.post(url=CONFIG['HOST_ADDR']+"/get_config", data=CONFIG)
			return render_template('home.html', conn=conn)
		else:
			conn = False
			return render_template('home.html', conn=conn)
	except:
		conn = False
		return render_template('home.html', conn=conn)
	
@app.route('/capture/<tag>', methods=['GET'])
def capture(tag):
	global conn

	if tag == "1":
		response = requests.get(CONFIG["HOST_ADDR"] + "/api/capture")

		if response.status_code != 200:
			conn = False
			return render_template('capture.html', Data=False, conn=conn)

		data = response.json()
		data = json.loads(data)
		img_id = data['image_id']

		predictions = [np.array([data['boxes']]), np.array([data['scores']]), np.array([data['classes']]), np.array([int(data['num_det'])])]
		
		img_url = CONFIG["HOST_ADDR"] + "/api/img/"+str(img_id)
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

			db.insert_data(CONFIG["DB_PATH"], 'Detections', (img_id, bbox, classes_name, scores, data['num_det']))


		data['classes'] = utils.get_class_names(data['classes'])
		data['boxes'] = np.round(data['boxes'], 3)
		data['scores'] = np.round(data['scores'], 3)
		filename = '/img/'+str(img_id)

		return render_template('capture.html', is_capture=True, data=data, filename=filename, conn=conn)
	
	else:
		return render_template('capture.html', is_capture=False, conn=conn)
	
@app.route('/stream/<tag>')
def stream(tag):
	"""Video streaming home page."""
	global conn

	if tag == "1":
		video_url = CONFIG["HOST_ADDR"] + "/video_feed"
		return render_template('stream.html', is_stream=True, video_url=video_url, conn=conn)
	else: 
		return render_template('stream.html', is_stream=False, conn=conn)

@app.route('/history')
def get_history():
	global conn
	try:
		conn_statues = requests.get(CONFIG["HOST_ADDR"] + "/") 
		if conn_statues.status_code == 200:
			conn = True
	except:
		conn = False
	try:
		response = requests.get(CONFIG["HOST_ADDR"] + "/release_cam")
	except:
		pass

	data = db.get_all_data(CONFIG["DB_PATH"], 'Detections')
	return render_template('history.html', data=data, conn=conn)

@app.route('/his_result/<img_id>')
def get_his_data(img_id):
	global conn
	
	data = db.get_data_id(CONFIG["DB_PATH"], 'Detections', img_id)

	filename = CONFIG['SELF_ADDR']+"/img/"+str(img_id)

	return render_template('his_result.html', data=data[0], filename=filename, conn=conn)

@app.route('/img/<img_id>', methods=['GET'])
def get_img(img_id):
	filename = './images/' + str(img_id) + '.jpg'
	return send_file(filename, mimetype='image/gif')


@app.route('/delete/<img_id>', methods=['GET'])
def delete(img_id):
	
	db.delete_data(CONFIG["DB_PATH"], 'Detections', img_id)
	return redirect('/history')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
	global conn

	try:
		response = requests.get(CONFIG["HOST_ADDR"] + "/release_cam")
	except:
		pass

	if request.method == 'POST':
		CONFIG['HOST_ADDR'] = request.form.get('host_add')
		CONFIG['DB_PATH'] = request.form.get('db_path')
		CONFIG['MAX_CAP'] = request.form.get('max_cap')
		CONFIG['BLUR_THRESHOLD'] = request.form.get('blur_th')

		r = requests.post(url=CONFIG['HOST_ADDR']+"/get_config", data=CONFIG)

		utils.set_config("app.config", CONFIG)
		return redirect('/settings')
	else:
		return render_template('settings.html', data=CONFIG, conn=conn)

@app.route('/config', methods=['GET'])
def send_config():
	return jsonify(CONFIG)

@app.route('/auto_cap/<tag>', methods=['GET'])
def auto(tag):
	
	if tag == "on":
		res = requests.get(CONFIG['HOST_ADDR']+'/auto_capture/1')
	elif tag == 'off':
		res = requests.get(CONFIG['HOST_ADDR']+'/auto_capture/0')
	
	return render_template('auto_cap.html', conn=conn)

@app.route('/get_auto', methods=['POST'])
def get_auto():
	results = request.form.get('data')
	file = request.files['image']
	
	data = json.loads(results)
	print(data)
	file.save('images/auto_'+data['image_id']+'.jpg')
	predictions = [np.array([data['boxes']]), np.array([data['scores']]), np.array([data['classes']]), np.array([int(data['num_det'])])]
	utils.make_bbox(predictions, "auto_"+data['image_id'])
	
	bbox, classes, scores = utils.convert_data(data)
	data = (data['image_id'], bbox, classes, scores, data['num_det'])
	db.insert_data(CONFIG['DB_PATH'], 'auto_data', data)
	
	print('done')
		
	return "ok"


if __name__ == '__main__':
    app.run(port=8000, debug=True, host='0.0.0.0')
    