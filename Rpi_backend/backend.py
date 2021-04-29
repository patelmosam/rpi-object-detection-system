import tensorflow as tf
import cv2
import numpy as np
import json
from json import JSONEncoder
from utils import *
import datetime
import os
import shutil

def initilize(model_path):

	interpreter = tf.lite.Interpreter(model_path=model_path)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	return interpreter, input_details, output_details

def capture(cap, MODEL_SIZE, blur_threshold=100):
	# cap = cv2.VideoCapture(0)
	assert cap.isOpened(), 'Cannot capture source'
	if cap.isOpened():
		ret, frame = cap.read()
		if ret:
			#cv2.imwrite("capture/capture_"+str(CNT)+".jpg", frame)
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
	interpreter.set_tensor(input_details[0]['index'], input_array)
	interpreter.invoke()

	pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

	try:
		boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
		# print(boxes, pred_conf)

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


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def process_predictions(predictions, image_id):
	num_det = predictions[3][0]
 
	  # TODO check .tolist()             
	boxes = predictions[0][0][:num_det]
	scores = predictions[1][0][:num_det]
	classes = predictions[2][0][:num_det]

	class_names = get_class_names(predictions[2][0][:num_det])

	results = { "image_id": image_id,
				"boxes": boxes,
				"scores": scores,
				"classes": classes,
				"num_det": str(num_det)
				}
	
	encodedData = json.dumps(results, cls=NumpyArrayEncoder)

	return encodedData

def get_class_names(classes):
	labels = read_class_names('coco.names')
	names = '['
	for cls in classes:
		names += labels[cls] + ','
	
	names += ']'
	return names

def make_bbox(original_img, pred_bbox, cnt):
	labels = read_class_names('coco.names')
	image = draw_bbox(original_img, pred_bbox, labels)
	#print(image.shape)
	# cv2.imwrite("static/capture__"+str(cnt)+".jpg", original_img)
	cv2.imwrite("static/capture_"+str(cnt)+".jpg", image)
	
def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
	
def is_blured(image, threshold=100):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	print(fm)
	if fm < threshold:
		return True
	return False
	
def capture_perfect(camera, MODEL_SIZE, max_capture, blur_threshold):
	for i in range(max_capture):
		print("capture: ", i)
		img, original_img = capture(camera, MODEL_SIZE, blur_threshold)
		if img is not None:
			return img, original_img
	return None, None

def get_image_id():
	cdate = datetime.date.today()
	now = datetime.datetime.now()
	ctime = now.strftime("%H-%M-%S")
	image_id = str(cdate)+"("+ctime+")"
	return image_id

def gen_frames(camera): 
	
	assert camera.isOpened(), 'Cannot capture source'
	while True:
		success, frame = camera.read()  
		if not success:
			break
		else:
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


	
def capture_and_predict(MODEL_SIZE, MAX_CAP, BLUR_THRESHOLD, interpreter, input_details, output_details):
	image_id = get_image_id()
	img, original_img = capture_perfect(MODEL_SIZE, MAX_CAP, BLUR_THRESHOLD)

	if img is not None:
		predictions = predict(interpreter, input_details, output_details, img, MODEL_SIZE[0])
		# print(predictions)
		# make_bbox(original_img, predictions, image_id)
		cv2.imwrite("static/capture_"+str(image_id)+".jpg", original_img)
		results = process_predictions(predictions, image_id)
		return results
	return None

def get_delete_imglist(image_dir, value, type_):
	img_list = os.listdir(image_dir)
	val = value.split('-')

	result_list = []
	for img in img_list:
		date = img.split('(')[0]
		y, m, d = date.split('-')

		if type_ == 'day':
			if d==val[0] and m==val[1] and y==val[2]:
				result_list.append(img)
		elif type_ == 'month':
			if m==val[0] and y==val[1]:
				result_list.append(img)
		elif type_ == 'week':
			_, w, _ = datetime.date(int(y), int(m), int(d)).isocalendar()
			if y==val[1] and str(w)==val[0]:
				result_list.append(img)
	return result_list

def get_space_info(path):
	return shutil.disk_usage(path)

def convert_to_dict(data_list):
	result_dict = {}
	for data in data_list:
		result_dict[data[0]] = {'num_images': data[1],
								'total_space': data[2]} 
	return result_dict

def delete_imgs(im_dir, img_list):

	for img in img_list:
		os.remove(im_dir+img)

# def update_database():
	
# 	img_list = os.listdir('./static/')
# 	data_dict = {'Daily': {},
# 				 'Monthly': {},
# 				 'Weekly': {}} 
# 	for img in img_list:
# 		date = img.split('(')[0]
# 		y, m, d = date.split('-')
# 		_, w, _ = datetime.date(int(y), int(m), int(d)).isocalendar()
# 		d_f = d+'-'+m+'-'+y
# 		data_dict['Daily'][d_f] = {'num_images': }

# 	get_delete_imglist('./static/', )
	
def update_database(type_, value):

	if type_ == 'day':
		d_data = database.get_data_id('database.db', 'Daily', 'Day', value)
		database.delete_data('database.db', 'Daily', 'Day', value)
		d, m, y = value.split('-')
		m_data = database.get_data_id('database.db', 'Monthly', 'Month', m+'-'+y)
		m_imgs, m_space = m_data[1]-d_data[1], m_data[2]-d_data[2]
		database.update_data_id('database.db', 'Monthly', 'Month', m+'-'+y, m_imgs, m_space)

		_, w, _ = datetime.date(int(y), int(m), int(d)).isocalendar()
		w_data = database.get_data_id('database.db', 'Monthly', 'Month', w+'-'+y)
		w_imgs, w_space = w_data[1]-d_data[1], w_data[2]-d_data[2]
		database.update_data_id('database.db', 'Weekly', 'Week', w+'-'+y, w_imgs, w_space)

	elif type_ == 'month':
		database.delete_data('database.db', 'Monthly', 'Month', value)
		for i in range(1,32):
			database.delete_data('database.db', 'Daily', 'Day', i+'-'+value)
		