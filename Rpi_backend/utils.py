import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json


def load_image(image_path, model_size):
	""" 
	This function loads the image from image_path (param) and process it. Processing includes converting
	image to RGB from BGR, resizing to model_size (param), normalizing with value 255 and expanding 
	dimantions. 

	params:- image_path: <str>
			 model_size: <tuple>

	returns:- image_data: <numpy.ndarray>
			  original_image: <numpy.ndarray>
	"""
	original_image = cv2.imread(image_path)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

	image_data = cv2.resize(original_image, model_size)
	image_data = image_data / 255.0
	image_data = np.expand_dims(image_data, axis=0)

	return image_data, original_image


def prep_image(img, model_size):
	""" 
	This function process the image by resizing it to model_size (param), then normalizing it with
	value 255 and expanding dimantions of it.

	params:- img: <numpy.ndarray>
			 model_size: <tuple>

	returns:- image_data: <numpy.ndarray>
	"""
	image_data = cv2.resize(img, model_size)
	image_data = image_data / 255.0
	image_data = np.expand_dims(image_data, axis=0)

	return image_data


def load_labels(labels_file):
	""" 
	This function opens the label file and returns the text data.

	params: lables_file: <str>

	returns:- labels: <str>
	"""
	with open(labels_file) as jfile:
		labels = json.load(jfile)
	return labels


def read_class_names(class_file_name):
	""" 
	This function reads the class names from txt file and loads into dict.

	params: class_file_name: <str>

	returns:- names: <dict>
	"""
	names = {}
	with open(class_file_name, 'r') as data:
		for ID, name in enumerate(data):
			names[ID] = name.strip('\n')
	return names


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
	""" 
	This function filters the bounding boxes and only includes the bounding boxes which has the score
	greater then or equal to the score_threshold.

	params:- box_xywh: <tf.tensor>
			 scores: <tf.tensor>
			 score_threshold: <float>
			 input_shape: <tf.constant>

	returns:- boxes: <tf.tensor>
			  pred_conf: <tf.tensor>
	"""
	scores_max = tf.math.reduce_max(scores, axis=-1)

	mask = scores_max >= score_threshold
	class_boxes = tf.boolean_mask(box_xywh, mask)
	pred_conf = tf.boolean_mask(scores, mask)
	class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
	pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

	box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

	input_shape = tf.cast(input_shape, dtype=tf.float32)

	box_yx = box_xy[..., ::-1]
	box_hw = box_wh[..., ::-1]

	box_mins = (box_yx - (box_hw / 2.)) / input_shape
	box_maxes = (box_yx + (box_hw / 2.)) / input_shape
	boxes = tf.concat([
		box_mins[..., 0:1],  # y_min
		box_mins[..., 1:2],  # x_min
		box_maxes[..., 0:1],  # y_max
		box_maxes[..., 1:2]  # x_max
	], axis=-1)
	return (boxes, pred_conf)


def process_bbox(boxes, width, hight):
	""" 
	This function scales the bounding boxes x_min and x_max values with width and y_min and
	y_max values with hight.

	params:- boxes: <numpy.ndarray>
			 width: <int>
			 hight: <int>

	returns:- boxes: <numpy.ndarray>
	"""
	for box in boxes:
		box[0] = int(box[0]*hight)
		box[1] = int(box[1]*width)
		box[2] = int(box[2]*hight)
		box[3] = int(box[3]*width)
	return boxes 


def get_config(config_file):
	""" 
	This function opens the json file from the path given as parameter and return the data after loading
	in the form of python dictionary.

	params:- config_file: <str>

	returns:- data: <dict>
	"""
	with open(config_file) as cfg:
		data = json.load(cfg)
	return data


def set_config(config_file, data):
	""" 
	This function accepts the python dict and writes the dict data to json file on specified path.

	params:- config_file: <str>
			 data: <dict>

	returns:- None
	"""
	with open(config_file, 'w') as cfg:
		json.dump(data, cfg, indent="")

