import tensorflow as tf
import cv2
import numpy as np
import json
from json import JSONEncoder
from utils import *
import datetime

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
	#print(no_det)
	# boxes = np.array2string(predictions[0][0][:no_det], precision=2, separator=',',
    #                   suppress_small=True)
	# scores = np.array2string(predictions[1][0][:no_det], precision=2, separator=',',
    #                   suppress_small=True)
	# classes = np.array2string(predictions[2][0][:no_det], precision=2, separator=',',
    #                   suppress_small=True)
 
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
	# print(type(ctime), type(cdate))
	image_id = str(cdate)+"-"+ctime
	return image_id

def gen_frames(camera): 

	# print(camera.isOpened())
	
	assert camera.isOpened(), 'Cannot capture source'
	while True:
		#Capture frame-by-frame
		success, frame = camera.read()  # read the camera frame
		if not success:
			break
		else:
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


	
def capture_and_predict(MODEL_SIZE, MAX_CAP, BLUR_THRESHOLD, interpreter, input_details, output_details):
	image_id = get_image_id()
	# print(image_id)
	# input_size = (MODEL_SIZE, MODEL_SIZE)
	img, original_img = capture_perfect(MODEL_SIZE, MAX_CAP, BLUR_THRESHOLD)
	# print(img.shape)
	if img is not None:
		predictions = predict(interpreter, input_details, output_details, img, MODEL_SIZE[0])
		# print(predictions)
		# make_bbox(original_img, predictions, image_id)
		cv2.imwrite("static/capture_"+str(image_id)+".jpg", original_img)
		results = process_predictions(predictions, image_id)
		return results
	return None

# def detect_motion(camera):
# 	ret, frame1 = camera.read()
# 	ret, frame2 = camera.read()
# 	print(frame1.shape)
# 	while camera.isOpened():
# 		diff = cv2.absdiff(frame1, frame2)
# 		gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# 		blur = cv2.GaussianBlur(gray, (5,5), 0)
# 		_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
# 		# print(thresh.shape)
# 		dilated = cv2.dilate(thresh, None, iterations=3)
# 		# print(dilated.shape)
# 		contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 		print(len(contours))
# 		for contour in contours:
# 			if cv2.contourArea(contour) > 900:
# 				camera.release()
# 				return frame2
# 	return None

# def auto_detect_img(camera, MODEL_SIZE):
# 	frame = None
# 	while frame is None:
# 		frame = detect_motion(camera)

# 	input_array = prep_image(frame, MODEL_SIZE)
# 	input_array = np.float32(input_array)
	
# 	return input_array, frame


	
