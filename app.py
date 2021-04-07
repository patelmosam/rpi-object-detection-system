from flask import Flask
from flask import request, jsonify, render_template, redirect
import requests
from database import *
import json
from utils import *

HOST_NAME = "http://127.0.0.1:5000"

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def home():
	r = requests.get(HOST_NAME)
	if r.status_code == 200:
		return render_template('index.html', data=None, filename=None, isData=False)
	else:
		return "<h2>server is not responding</h2>"
	
@app.route('/capture', methods=['GET'])
def capture():

	response = requests.get(HOST_NAME + "/api/capture")
	print(response)
	if response.status_code != 200:
		return "<h2>server is not responding</h2>"

	try:
		id = response.json()['image_id']
		data = response.text
	except:
		error = response.json()['error']
		msg = response.json()['msg']
		return "<h2> " + msg + "</h2>"
		
	filename = HOST_NAME + "/api/img/"+str(id)
	print(filename)

	jdata = json.loads(data)

	r = requests.get(filename)
	file = open('images/'+jdata['image_id']+'.png', 'wb')
	file.write(r.content)
	file.close()

	if jdata['no_det'] != "0":
		conn = sqlite3.connect('database.db')
		insert_data(conn, jdata['image_id'], jdata['boxes'], jdata['classes'], jdata['scores'], jdata['no_det'])

	return render_template('index.html', data=jdata, filename=filename, isData=True)
	
@app.route('/stream')
def stream():
	"""Video streaming home page."""
	video_url = HOST_NAME + "/video_feed"
	return render_template('stream.html', video_url=video_url)

@app.route('/history')
def get_history():
	response = requests.get(HOST_NAME + "/release_cam")
	conn = sqlite3.connect('database.db')
	data = get_all_data(conn)

	return render_template('history.html', data=data)

if __name__ == '__main__':
    app.run(port=8000, debug=True, host='0.0.0.0')
    
