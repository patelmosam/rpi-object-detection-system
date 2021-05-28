from flask import Flask
from flask import request, jsonify, render_template, redirect, send_file
import numpy as np
import requests
import database as db
import json
import utils
from PIL import Image
import os

CONFIG = utils.get_config("app.config")
db.Check_db('./database.db')

app = Flask(__name__, template_folder='template')
conn = False


@app.route('/', methods=['GET'])
def home():
	""" 
	This function handels the request for home page. It checks the connection status with RPI and
	renders the home page.

	returns:- Http Response
	"""
	global conn
	try:
		conn_statues = requests.get(CONFIG["HOST_ADDR"]+'/api')
	
		if conn_statues.status_code == 200:
			conn = True
			req = requests.post(url=CONFIG['HOST_ADDR']+"/api/get_config", data=CONFIG)
			return render_template('home.html', conn=conn)
		else:
			conn = False
			return render_template('home.html', conn=conn)
	except:
		conn = False
		return render_template('home.html', conn=conn)

	
@app.route('/capture/<tag>', methods=['GET'])
def capture(tag):
	""" 
	This function handels capture page request. If the tag in url is equal to "0" then it will render
	the page with button to start live stream. If the tag in url is equal to "stream" then it will 
	render the live stream page. And if the tag is equal to "1" then it will render the capture page.

	returns:- Http Response
	"""
	global conn
	video_url = CONFIG["HOST_ADDR"] + "/api/video_feed"
	try:
		if tag=="1":
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

			return render_template('capture.html', is_stream=False, data=data, filename=filename, conn=conn, video_url=video_url)
		
		elif tag == 'stream':
			return render_template('capture.html', is_stream=True, home=False, conn=conn, video_url=video_url)

		else:
			try:
				response = requests.get(CONFIG["HOST_ADDR"] + "/api/release_cam")
			except:
				pass
			return render_template('capture.html', is_stream=True, home=True, conn=conn)
	except:
		error_msg = "capture error"
		return render_template('error.html', error=error_msg, conn=conn)


@app.route('/history')
def get_history():
	""" 
	This function handels the request for history page. It will fetch the data from the database and
	renders to history page.

	returns:- Http Response
	"""
	global conn
	try:
		try:
			conn_statues = requests.get(CONFIG["HOST_ADDR"] + "/api") 
			if conn_statues.status_code == 200:
				conn = True
		except:
			conn = False
		try:
			response = requests.get(CONFIG["HOST_ADDR"] + "/api/release_cam")
		except:
			pass

		data = db.get_all_data(CONFIG["DB_PATH"], 'Detections')
		auto_data = db.get_all_data(CONFIG["DB_PATH"], 'auto_data')

		return render_template('history.html', data=data, auto_data=auto_data, conn=conn)
	except:
		error_msg = "history fetch error"
		return render_template('error.html', error=error_msg, conn=conn)


@app.route('/his_result/<tag>')
def get_his_data(tag):
	""" 
	This function handels the request for history results page for specific image_id. It will fetch the 
	data form the database with the image_id and renders to the data to the his_result page. 
	
	returns:- Http Response
	"""
	global conn
	try: 
		table, img_id = tag.split('+')
		print(img_id, table)
		data = db.get_data_id(CONFIG["DB_PATH"], table, img_id)

		filename = CONFIG['SELF_ADDR']+"/img/"+str(img_id)

		return render_template('his_result.html', data=data[0], filename=filename, conn=conn)
	except:
		error_msg = 'error in retriving history data'
		return render_template('error.html', error=error_msg, conn=conn)


@app.route('/img/<img_id>', methods=['GET'])
def get_img(img_id):
	""" 
	This function handels the request for the image. It will return the image with the image_id provided 
	in the url form the "./images/" folder.

	returns:- Http Response
	"""
	filename = './images/' + str(img_id) + '.jpg'
	return send_file(filename, mimetype='image/gif')


@app.route('/delete/<tag>', methods=['GET'])
def delete(tag):
	""" 
	This function handels the request for the delete. It will delete the data from database and "./images/"
	folder with image_id. 
	
	returns:- Http Response
	"""
	table, img_id = tag.split('+')
	db.delete_data(CONFIG["DB_PATH"], table, img_id)
	try:
		os.remove('./images/'+img_id+'.jpg')
	except:
		print('image not found: ', img_id)
	return redirect('/history')


@app.route('/settings', methods=['GET', 'POST'])
def settings():
	""" 
	This function handels the request for settings page. If the request is GET then it will render the
	settings page. And if the request is POST then it will accepts the data form the settings page. 
	
	returns:- Http Response
	"""
	global conn
	try:
		try:
			response = requests.get(CONFIG["HOST_ADDR"] + "/api/release_cam")
		except:
			pass

		if request.method == 'POST':
			CONFIG['HOST_ADDR'] = request.form.get('host_add')
			CONFIG['DB_PATH'] = request.form.get('db_path')
			CONFIG['MAX_CAP'] = request.form.get('max_cap')
			CONFIG['BLUR_THRESHOLD'] = request.form.get('blur_th')
			CONFIG['MOTION_THRESHOLD'] = request.form.get('motion_th')

			r = requests.post(url=CONFIG['HOST_ADDR']+"/api/get_config", data=CONFIG)

			utils.set_config("app.config", CONFIG)
			return redirect('/settings')
		else:
			return render_template('settings.html', data=CONFIG, conn=conn)
	except:
		error_msg = "settings page error"
		return render_template('error.html', error=error_msg, conn=conn)


@app.route('/config', methods=['GET'])
def send_config():
	"""	
	This function handels the request for the config file. It will return the config file.

	returns:- Http Response
	"""
	return jsonify(CONFIG)


@app.route('/auto_cap/<tag>', methods=['GET'])
def auto(tag):
	""" 
	This function handels the request for auto_capture. If the tag in url is equal to "on" then it 
	will send the GET request for to start the auto-capture process on RPI. And if the tag is equal to 
	"off" the it will send GET request for to stop auto-capture on RPI.

	returns:- Http Response
	"""
	try:
		if tag == "on":
			res = requests.get(CONFIG['HOST_ADDR']+'/api/auto_capture/1')
		elif tag == 'off':
			res = requests.get(CONFIG['HOST_ADDR']+'/api/auto_capture/0')
		if res.status_code == 200:
			conn = True
		return render_template('auto_cap.html', conn=conn)
	except:
		error_msg = "auto_cap error"
		return render_template('error.html', error=error_msg, conn=conn)


@app.route('/get_auto', methods=['POST'])
def get_auto():
	""" 
	This function handels the request for get the data from auto-capture process. It will accepts the
	data(results and image) and  from POST request and stores into database and './images/' folder.

	returns:- Http Response
	"""
	results = request.form.get('data')
	file = request.files['image']
	
	data = json.loads(results)
	print(data)
	file.save('images/auto_'+data['image_id']+'.jpg')
	predictions = [np.array([data['boxes']]), np.array([data['scores']]), np.array([data['classes']]), np.array([int(data['num_det'])])]
	utils.make_bbox(predictions, "auto_"+data['image_id'])
	
	bbox, classes, scores = utils.convert_data(data)
	classes_name = utils.get_class_names(data['classes'])
	data = ('auto_'+data['image_id'], bbox, classes_name, scores, data['num_det'])
	db.insert_data(CONFIG['DB_PATH'], 'auto_data', data)
	
	print('done')
		
	return "ok"


@app.route('/manage_rpi', methods=['GET'])
def manage_rpi():
	"""	
	This function handels the request for manage_rpi page and rendera the manage_rpi page.

	returns:- Http Response
	"""
	global conn
	try:
		try:
			response = requests.get(CONFIG["HOST_ADDR"] + "/api/release_cam")
		except:
			pass
		space_data = requests.get(CONFIG['HOST_ADDR']+'/api/get_data')

		return render_template('manage_rpi.html', conn=conn, data=space_data.json())
	except:
		error_msg = "manage_rpi page error"
		return render_template('error.html', error=error_msg, conn=conn)


@app.route('/delete_imgs', methods=['POST'])
def delete_imgs():
	"""	
	This function handels the request for the delete_imgs. It will accepts the POST request and get the
	data (num_images) from it. And send the GET request to RPI on '/delete_img' endpoint with num_images
	specified in the tag.

	returns:- Http Response
	"""
	if request.method == 'POST':
		tag = request.form.get('num_imgs')
		req = requests.get(CONFIG['HOST_ADDR']+'/api/delete_imgs/'+tag)

	return redirect('/manage_rpi')


if __name__ == '__main__':
	app.run(port=8000, debug=True, host='0.0.0.0')
	