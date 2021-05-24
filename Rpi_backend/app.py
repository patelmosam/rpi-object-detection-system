import flask
from flask import request, jsonify, render_template, url_for, send_file, Response
import backend 
import datetime
import utils
import cv2
import requests
from threading import Thread
import database
import os
import numpy as np

MODEL_PATH = './weights/yolov4-416-tiny.tflite'
input_size = 416
MODEL_SIZE = (input_size, input_size)
CONFIG = None
camera = None
AutoDetect = False

app = flask.Flask(__name__)
app.config["DEBUG"] = True

interpreter, input_details, output_details = backend.initilize(MODEL_PATH)

def auto_detect_img(MODEL_SIZE, interpreter, input_details, output_details):
	global AutoDetect
	frame = None
	camera = cv2.VideoCapture(0)
	
	ret, frame1 = camera.read()
	ret, frame2 = camera.read()

	while AutoDetect and camera.isOpened():
		frame = backend.detect_motion(frame1, frame2)

		if frame is not None:
			input_array = utils.prep_image(frame, MODEL_SIZE)
			input_array = np.float32(input_array)
			image_id = backend.get_image_id()
		
			predictions = backend.predict(interpreter, input_details, output_details, input_array, MODEL_SIZE[0])
			cv2.imwrite("static/auto.jpg", frame)

			if predictions[3][0] > 0:
				results = backend.process_predictions(predictions, image_id)
			
				img = open('static/auto.jpg', 'rb')
			
				data = {'data':results}
				img = {'image': img}
				r = requests.post(url="http://127.0.0.1:8000/get_auto", files=img, data=data)
				print(r.status_code)
		
		frame1 = frame2
		ret, frame2 = camera.read()
			
	camera.release()
	return None

@app.route('/', methods=['GET'])
def home():
	global CONFIG
	global camera
	if camera is not None:
		camera.release()
		camera = None

	data = {"statues": "connected"}
	return Response(data, content_type='application/json')


@app.route('/api/capture', methods=['GET'])
def capture_img():
	global camera
	global CONFIG
	results = None
	if camera is not None:
		camera.release()
		camera = None
	camera = cv2.VideoCapture(0)
	
	if CONFIG is None:
		r = requests.get('http://127.0.0.1:8000/config')
		CONFIG = r.json()

	image_id = backend.get_image_id()
	img, original_img = backend.capture_perfect(camera, 
												MODEL_SIZE, 
												int(CONFIG["MAX_CAP"]), 
												int(CONFIG["BLUR_THRESHOLD"]))

	if img is not None:
		predictions = backend.predict(interpreter, input_details, output_details, img, input_size)
	
		# make_bbox(original_img, predictions, image_id)
		cv2.imwrite("static/"+str(image_id)+".jpg", original_img)
		results = backend.process_predictions(predictions, image_id)

		img_size = os.path.getsize('static/'+str(image_id)+'.jpg')
		database.insert_data('database.db', img_size)

	if results is not None:
		return jsonify(results)
		# return Response(response=results, mimetype='application/json')
	else:
		results = { 'error': "500",
					'msg' : 'Captured the blur Image , Try Again..'
					}
		return jsonify(results)

@app.route('/api/img/<img_id>', methods=['GET'])
def get_img(img_id):
	filename = './static/' + str(img_id) + '.jpg'
	return send_file(filename, mimetype='image/gif')

@app.route('/video_feed')
def video_feed():
	global camera
	if camera is not None:
		camera.release()
		camera = None

	camera = cv2.VideoCapture(0)
	return Response(backend.gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/release_cam')
def release_cam():
	global camera
	if camera is not None:
		camera.release()
		camera = None
	data = {"release": "True"}
	return Response(data, content_type='application/json')

@app.route('/get_config', methods=['GET', 'POST'])
def get_config():
	global CONFIG
	if request.method == 'POST':
		CONFIG = request.form.to_dict()
		return "True"


@app.route('/auto_capture/<tag>', methods=['GET'])
def auto_capture(tag):
	global camera
	global AutoDetect
	if camera is not None:
		camera.release()
		camera = None
	if tag == "1":
		AutoDetect = True
		t1 = Thread(target=auto_detect_img, args=[MODEL_SIZE, interpreter, input_details, output_details])
		t1.start()
	elif tag == "0":
		AutoDetect = False
	
	return "OK"

@app.route('/get_data', methods=['GET'])
def get_data():
	space_stat = backend.get_space_info('./static')
	 
	daily_data = database.get_all_data('database.db', 'Daily')
	monthly_data = database.get_all_data('database.db', 'Monthly')
	weekly_data = database.get_all_data('database.db', 'Weekly')

	daily_dict = backend.convert_to_dict(daily_data)
	monthly_dict = backend.convert_to_dict(monthly_data)
	weekly_dict = backend.convert_to_dict(weekly_data)

	result_dict = {
		'space_stat': {
			'totel_space': space_stat[0]/1000000000,
			'used_space': space_stat[1]/1000000000,
			'free_space': space_stat[2]/1000000000
			},
		'daily_data': daily_dict,
		'monthly_data': monthly_dict,
		'weekly_data': weekly_dict
	}
	return jsonify(result_dict)

@app.route('/clean_space/<tag>', methods=['GET'])
def clean_space(tag):
	type_, value = tag.split('_') 
	print(type_, value)
	delete_list = backend.get_delete_imglist('./static/', value, type_)
	backend.delete_imgs('./static/', delete_list)

	if type_ == 'day':
		database.delete_data('database.db', 'Daily', 'Day', value)
	elif type_ == 'month':
		database.delete_data('database.db', 'Monthly', 'Month', value)
	elif type_ == 'week':
		database.delete_data('database.db', 'Weekly', 'Week', value)

	data = {"clean": "True"}	
	return Response(data, content_type='application/json')

@app.route('/delete_last_imgs/<tag>', methods=['GET'])
def delete_last(tag):
	img_list = os.listdir('./static/')
	img_list.sort()
	backend.delete_imgs('./static/', img_list)

	data = {"clean": "True"}	
	return Response(data, content_type='application/json')



if __name__ == "__main__":
    app.run(debug=True)
    #app.run(debug=True, port=8080, host='0.0.0.0')
