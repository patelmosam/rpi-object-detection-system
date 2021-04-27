import flask
from flask import request, jsonify, render_template, url_for, send_file, Response
import backend 
import datetime
from utils import prep_image
import cv2
import requests

MODEL_PATH = './weights/yolov4-416-tiny.tflite'
input_size = 416
MODEL_SIZE = (input_size, input_size)
CONFIG = None
camera = None

app = flask.Flask(__name__)
app.config["DEBUG"] = True

interpreter, input_details, output_details = backend.initilize(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
	global CONFIG
	global camera
	if camera is not None:
		camera.release()
		camera = None
	# r = requests.get('http://127.0.0.1:8000/config')
	# CONFIG = r.json()

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
		cv2.imwrite("static/capture_"+str(image_id)+".jpg", original_img)
		results = backend.process_predictions(predictions, image_id)

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
	filename = './static/capture_' + str(img_id) + '.jpg'
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

@app.route('/auto_capture')
def auto_capture():
	global camera
	if camera is not None:
		camera.release()
		camera = None
	camera = cv2.VideoCapture(0)
	img, original_img = backend.auto_detect_img(camera, MODEL_SIZE)
	image_id = backend.get_image_id()
	if img is not None:
		predictions = backend.predict(interpreter, input_details, output_details, img, MODEL_SIZE[0])
		cv2.imwrite("static/capture_"+str(image_id)+".jpg", original_img)
		results = backend.process_predictions(predictions, image_id)
		print(results)

	if results is not None:
		return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(debug=True, port=8080, host='0.0.0.0')
