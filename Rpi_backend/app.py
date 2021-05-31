import flask
from flask import request, jsonify, send_file, Response
import backend 
import utils
import cv2
import requests
from threading import Thread
import os
import numpy as np

MODEL_PATH = './weights/yolov4-416-tiny.tflite'
input_size = 416
MODEL_SIZE = (416, 416)
CONFIG = None
camera = None
AutoDetect = False

app = flask.Flask(__name__)
app.config["DEBUG"] = True

interpreter, input_details, output_details = backend.initialize(MODEL_PATH)


def auto_detect_img(MODEL_SIZE, interpreter, input_details, output_details, motion_thresh):
	""" 
	This function executes the auto-detection process. It runs a while loop in which the motion detection 
	is performed between two captured frames. If motion detection is true then only object detection is 
	performed otherwise skipped. Every time when object detection is performed, the results and image are 
	sent to server via post request. 

	params:- MODEL_SIZE <tuple> : eg. (416, 416)
			 interpreter : tflite interperter object
			 input_details <tf.tensor>
			 output_details <tf.tensor>
			 motion_thresh <int>

	returns:- None
    """
	global AutoDetect
	frame = None
	camera = cv2.VideoCapture(0)
	
	ret, frame1 = camera.read()
	ret, frame2 = camera.read()

	while AutoDetect and camera.isOpened():
		frame = backend.detect_motion(frame1, frame2, motion_thresh)

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


@app.route('/api', methods=['GET'])
def home():
	""" 
	This function handles the request for '/api' endpoint.

	returns:- Response 
	"""
	global CONFIG
	global camera
	if camera is not None:
		camera.release()
		camera = None

	data = {"statues": "connected"}
	return Response(data, content_type='application/json')


@app.route('/api/capture', methods=['GET'])
def capture_img():
	""" 
	This function handles the capture & detection request. It captures the image, preprocess it, 
	performs object detection and returns the result bake to the server. 

	returns:- JSON Response => { 'image_id': <str>,
								'bbox': <List>,
								'classes: <List>,
								'scores': <List>,
								'num_det': <int> }
	"""
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
	
		cv2.imwrite("static/"+str(image_id)+".jpg", original_img)
		results = backend.process_predictions(predictions, image_id)

	if results is not None:
		return jsonify(results)
	else:
		results = { 'error': "500",
					'msg' : 'Captured the blur Image , Try Again..'
					}
		return jsonify(results)


@app.route('/api/img/<img_id>', methods=['GET'])
def get_img(img_id):
	""" 
	This function send the image requested by 'img_id' from "./static/" folder.

	params:- img_id: <str>

	returns:- Response : image
	"""
	filename = './static/' + str(img_id) + '.jpg'
	return send_file(filename, mimetype='image/gif')


@app.route('/api/video_feed')
def video_feed():
	""" 
	This function is responsible for continuously capturing and sending the frame. It continuesly returns
	the frames as response.
	"""
	global camera
	if camera is not None:
		camera.release()
		camera = None

	camera = cv2.VideoCapture(0)
	return Response(backend.gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/release_cam')
def release_cam():
	""" 
	This function is used for releasing / turning off camera. 

	returns:- Response
	"""
	global camera
	if camera is not None:
		camera.release()
		camera = None
	data = {"release": "True"}
	return Response(data, content_type='application/json')


@app.route('/api/get_config', methods=['POST'])
def get_config():
	""" 
	This function accepts the POST request containing config file data.

	returns:- Response
	"""
	global CONFIG
	if request.method == 'POST':
		CONFIG = request.form.to_dict()
		return "True"


@app.route('/api/auto_capture/<tag>', methods=['GET'])
def auto_capture(tag):
	""" 
	This function handels the auto-capture request. If the tag in url is equal to "1", then it sets 
	the global variable "AutoDetect" to True and starts a seprate thread which will execute the 
	auto-capture process. And if the tag in url is equal to "0", then is set the global variable 
	"AutoDetect" to False which will break/stop the auto-capture process.

	returns:- Response
	"""
	global camera
	global AutoDetect
	global CONFIG
	if camera is not None:
		camera.release()
		camera = None

	if CONFIG is None:
		r = requests.get('http://127.0.0.1:8000/config')
		CONFIG = r.json()
	
	if tag == "1":
		AutoDetect = True
		t1 = Thread(target=auto_detect_img, 
					args=[MODEL_SIZE, interpreter, input_details,	 
						output_details, int(CONFIG['MOTION_THRESHOLD'])])
		t1.start()
	elif tag == "0":
		AutoDetect = False
	
	return "OK"


@app.route('/api/delete_imgs/<tag>', methods=['GET'])
def delete_last(tag):
	""" 
	This function handles the delete_images request. It deletes the oldest images from "./static" 
	folder. The number of images to be deleted is provided in url tag.

	returns:- Response
	"""
	img_list = os.listdir('./static/')
	img_list.sort()
	backend.delete_imgs('./static/', img_list[:int(tag)])

	data = {"clean": "True"}	
	return Response(data, content_type='application/json')


@app.route('/api/get_data', methods=['GET'])
def get_data():
	""" 
	This function handles the get_data request. It calculated the space occupied by './static/' 
	folder and the number of images stored in that folder and sends that info as response.

	returns:- Response 
	"""
	space_stat = backend.get_space_info('./static')

	num_images = backend.get_available_img('./static')

	result_dict = {
		'space_stat': {
			'total_space': space_stat[0]/1000000000,
			'used_space': space_stat[1]/1000000000,
			'free_space': space_stat[2]/1000000000
			},
		'num_images': num_images
	}
	return jsonify(result_dict)


if __name__ == "__main__":
	app.run(debug=True)
