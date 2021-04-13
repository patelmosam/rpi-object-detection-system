from flask import Flask
from flask import request, jsonify, render_template, redirect, send_file
import requests
from database import *
import json
from utils import *
# from Rpi_backend import 

HOST_NAME = "http://127.0.0.1:5000"
# HOST_NAME = "http://192.168.108.55:8080"

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def home():
	try:
		r = requests.get(HOST_NAME)
	
		if r.status_code == 200:
			return render_template('home.html', connection = True)
	except:
		return render_template('home.html', connection=False)
	
@app.route('/capture', methods=['GET'])
def capture():

	response = requests.get(HOST_NAME + "/api/capture")
	print(response)
	if response.status_code != 200:
		return "<h2>server is not responding</h2>"
	
	data = response.json()
	data = json.loads(data)
	id = data['image_id']

	predictions = [np.array([data['boxes']]), np.array([data['scores']]), np.array([data['classes']]), np.array([int(data['no_det'])])]
	print(predictions)
	# try:
	# 	id = response.json()['image_id']
	# 	data = response.text
	# except:
	# 	error = response.json()['error']
	# 	msg = response.json()['msg']
	# 	return "<h2> " + msg + "</h2>"
		
	img_url = HOST_NAME + "/api/img/"+str(id)
	print(img_url)

	r = requests.get(img_url)
	
	file = open('images/'+id+'.jpg', 'wb')
	file.write(r.content)
	file.close()

	if data['no_det'] != "0":
		print("adding to database")
		make_bbox(predictions, id)
		bbox, classes, scores = convert_data(data)
		classes_name = get_class_names(data['classes'])

		conn = sqlite3.connect('database.db')
		insert_data(conn, (id, bbox, classes_name, scores, data['no_det']))
		# insert_data(conn, data)

	data['classes'] = get_class_names(data['classes'])
	data['boxes'] = np.round(data['boxes'], 3)
	data['scores'] = np.round(data['scores'], 3)
	filename = '/img/'+str(id)

	return render_template('index.html', data=data, filename=filename, isData=True)
	
@app.route('/stream')
def stream():
	"""Video streaming home page."""
	video_url = HOST_NAME + "/video_feed"
	return render_template('stream.html', video_url=video_url)

@app.route('/history')
def get_history():
	try:
		response = requests.get(HOST_NAME + "/release_cam")
	except:
		pass
	conn = sqlite3.connect('database.db')
	data = get_all_data(conn)

	return render_template('history.html', data=data)

@app.route('/his_result/<id>')
def get_his_data(id):
	conn = sqlite3.connect('database.db')
	data = get_data_id(conn, id)
	print(type(data))
	filename = "http://127.0.0.1:8000/img/"+str(id)

	return render_template('his_result.html', data=data[0], filename=filename)

@app.route('/img/<id>', methods=['GET'])
def get_img(id):
	filename = './images/' + str(id) + '.jpg'
	return send_file(filename, mimetype='image/gif')


@app.route('/delete/<id>', methods=['GET'])
def delete(id):
	conn = sqlite3.connect('database.db')
	delete_data(conn, id)
	return redirect('/history')

if __name__ == '__main__':
    app.run(port=8000, debug=True, host='0.0.0.0')
    
