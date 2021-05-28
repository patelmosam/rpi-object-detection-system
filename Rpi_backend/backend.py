import tensorflow as tf
import cv2
import numpy as np
import json
from json import JSONEncoder
from utils import *
import datetime
import requests
import time
import os
import shutil

def initilize(model_path):
	''' This function loads the model and initialize the tf.lite interpreter.

		params:- model_path: <str>

		returns:- interpreter: <tflite object>
				  input_details: <tf.tensor>
				  output_details: <tf.tensor>
	'''
	interpreter = tf.lite.Interpreter(model_path=model_path)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	return interpreter, input_details, output_details

def capture(cap, MODEL_SIZE, blur_threshold=100):
	''' This function captures frame from the camera and check for blurness. If the captured frame is
	 	blur then return None otherwise returns the frame after processing it.

		params:- cap: <cv2 camera object>
				 MODEL_SIZE: <tuple>
				 blur_threshold: <int>

		returns:- input_array: <numpy.ndarray>
				  frame: <numpy.ndarray>
	'''
	assert cap.isOpened(), 'Cannot capture source'
	if cap.isOpened():
		ret, frame = cap.read()
		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			fm = variance_of_laplacian(gray)
			print("blur = ",fm)
			if fm < blur_threshold:
				return None, None
			
			input_array = prep_image(frame, MODEL_SIZE)
			input_array = np.float32(input_array)
			cap.release()
			return input_array, frame
	return None, None

def predict(interpreter, input_details, output_details, input_array, input_size):
	''' This function perfoms object detections on given image data with tensorflow lite 
		interpreter. After prediction it filers the result and applys the NMS(non maximum suppression)
		on results. It returns the prediction results after converting them to numpy array.

		params:- interpreter: <tflite object>
				 input_details: <tf.tensor>
				 output_details: <tf.tensor>
				 input_array: <numpy.ndarray>
				 input_size: <int>

		returns:- boxes: <numpy.ndarray>
				  scores: <numpy.ndarray>
				  classes: <numpy.ndarray>
				  valid_detections: <numpy.ndarray>
	'''

	interpreter.set_tensor(input_details[0]['index'], input_array)
	interpreter.invoke()

	pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

	try:
		boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
		

		boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
				boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
				scores=tf.reshape(
					pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
				max_output_size_per_class=50,
				max_total_size=50,
				iou_threshold=0.45,
				score_threshold=0.25
			)
		return boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()
	except:
		return None


def process_predictions(predictions, image_id):
	''' This function converts the predictions (numpy array) to list and stores into the dict.
		It returns the resultent dict after serializing it with 'json.dumps' method.

		params:- predictions: <numpy.ndarray>
				image_id: <str>

		returns:- encodedData: <bytes>
	'''

	num_det = predictions[3][0]
	boxes = predictions[0][0][:num_det].tolist()
	scores = predictions[1][0][:num_det].tolist()
	classes = predictions[2][0][:num_det].tolist()

	class_names = get_class_names(predictions[2][0][:num_det])

	results = { "image_id": image_id,
				"boxes": boxes,
				"scores": scores,
				"classes": classes,
				"num_det": str(num_det)
				}
	
	encodedData = json.dumps(results)

	return encodedData

def get_class_names(classes):
	''' This function takes list containing classes_id(int) and converts into list containing the 
        classes name coresponing the class_id using 'read_class_names' function.

        params:- classes: <list>

        returns:- names: <list>
    '''
	labels = read_class_names('coco.names')
	names = '['
	for cls in classes:
		names += labels[cls] + ','
	
	names += ']'
	return names
	
def variance_of_laplacian(image):
	''' This function calculate the variance of the image array.

		params:- image: <numpy.ndarray>

		returns:- <int>
	'''
	return cv2.Laplacian(image, cv2.CV_64F).var()
	
def is_blured(image, threshold=100):
	''' This function checks if the given image is blur of not. If blur the return True otherwise Flase.

		params:- image: <numpy.ndarray>
				 threshold: <int>

		returns:- <bool>
	'''
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	print(fm)
	if fm < threshold:
		return True
	return False
	
def capture_perfect(camera, MODEL_SIZE, max_capture, blur_threshold):
	''' This function captures the image using helper function "capture". If the capture function returns
		the image the it will return it oterwise it will try again at most max_capture(int) times.

		params:- camera: <cv2 camera object>
				 MODEL_SIZE: <tuple>
				 max_capture: <int>
				 blur_threshold: <int>

		returns:- img: <numpy.ndarray>
                  original_img: <numpy.ndarray>
	'''
	for i in range(max_capture):
		print("capture: ", i)
		img, original_img = capture(camera, MODEL_SIZE, blur_threshold)
		if img is not None:
			return img, original_img
	return None, None

def get_image_id():
	''' This function returns the string which contains date and time info.

		returns:- image_id: <str>
	'''
	cdate = datetime.date.today()
	now = datetime.datetime.now()
	ctime = now.strftime("%H-%M-%S")
	image_id = str(cdate)+"("+ctime+")"
	return image_id

def gen_frames(camera): 
	''' This function continuesly captures and yields the frames after encoding it.

		params:- camera: <cv2 camera object>

		returns:- <byte>
	'''
	assert camera.isOpened(), 'Cannot capture source'
	while True:
		success, frame = camera.read()  
		if not success:
			break
		else:
			frame = cv2.resize(frame, (420,320))
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


	
def capture_and_predict(MODEL_SIZE, MAX_CAP, BLUR_THRESHOLD, interpreter, input_details, output_details):
	''' This function captures the image using helper function "capture_perfect", performs the object 
		detection using helper function "predict" and returns the result after precessing it.

		params:- MODEL_SIZE: <tuple>
				 MAX_CAP: <int>
				 BLUR_THRESHOLD: <int>
				 interpreter: <tflite object>
				 input_details: <tf.tensor>
				 output_details: <tf.tensor>
		
		returns:- results: <bytes>
	'''
	image_id = get_image_id()
	img, original_img = capture_perfect(MODEL_SIZE, MAX_CAP, BLUR_THRESHOLD)

	if img is not None:
		predictions = predict(interpreter, input_details, output_details, img, MODEL_SIZE[0])
		# make_bbox(original_img, predictions, image_id)
		cv2.imwrite("static/capture_"+str(image_id)+".jpg", original_img)
		results = process_predictions(predictions, image_id)
		return results
	return None

def detect_motion(frame1, frame2, motion_thresh):
	''' This function detects the motion between two frames using background subtraction method.

		params:- frame1: <numpy.ndarray>
				 frame2: <numpy.ndarray>
				 motion_thresh: <int>
		
		returns: frame2: <numpy.ndarray>
	'''
	diff = cv2.absdiff(frame1, frame2)
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
	
	dilated = cv2.dilate(thresh, None, iterations=3)
	
	contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	for contour in contours:
		if cv2.contourArea(contour) > motion_thresh:
			return frame2

	return None

def get_space_info(path):
	''' This function returns the storage space info of the path folder.

		params:- path: <str>

		returns:- <tuple>
	'''
	return shutil.disk_usage(path)

def convert_to_dict(data_list):
	''' This function converts the list data into dict.

		params:- data_list: <list>
				 
		returns:- result_dict: <dict>
	'''
	result_dict = {}
	for data in data_list:
		result_dict[data[0]] = {'num_images': data[1],
								'total_space': data[2]} 
	return result_dict

def delete_imgs(im_dir, img_list):
	''' This function delets the image provided in img_list form the im_dir path.

		params:- im_dir: <str>
				 img_list: <list>

		returns:- None 
	'''
	for img in img_list:
		os.remove(im_dir+img)

def get_available_img(image_dir):
	''' This function returns the number of image availble at the image_dir

		params:- image_dir: <str>

		returns:- number: <int>
	'''
	img_list = os.listdir(image_dir)
	return len(img_list)

