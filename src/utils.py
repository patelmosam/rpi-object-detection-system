import json
import numpy as np
import cv2
import colorsys
import random


def get_config(config_file):
    ''' This function opens the json file from the path given as parameter and return the data after loading
        in the form of python dictionary.

        params:- config_file: <str>

        returns:- data: <dict>
    '''
    with open(config_file) as cfg:
        data = json.load(cfg)
    return data

def set_config(config_file, data):
    ''' This function accepts the python dict and writes the dict data to json file on specified path.

        params:- config_file: <str>
                data: <dict>

        returns:- None
    '''
    with open(config_file, 'w') as cfg:
        json.dump(data, cfg, indent="")
 
def make_bbox(pred_bbox, id):
    ''' This function opens the image form './images' folder which has name==id and makes the 
        bounding boxes on top of it with coordinates values (pred_bbox) using helper function "draw_bbox"
        and stores back to the same folder with same name.  

        params:- pred_bbox: <numpy.ndarray>
                id: <str>

        returns:- None
    '''
    original_img = cv2.imread('./images/'+id+".jpg")
    print(original_img.shape)
    labels = read_class_names('coco.names')
    image = draw_bbox(original_img, pred_bbox, labels)
    cv2.imwrite("images/"+str(id)+".jpg", image)

def draw_bbox(image, bboxes, classes, show_label=True):
    ''' This is the helper function to draw the bounding box on the image.

        params:- image: <numpy.ndarray>
                 bboxes: <numpy.ndarray>
                 classes: <numpy.ndarray>
                 show_label: <bool>

        returns:- image: <numpy.ndarray>
    '''
    num_classes = len(classes)
    image_h, image_w, _ = image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
  
        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
       
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def convert_data(data):
    ''' This function converts the numpy array data to string

        params:- data: <numpy.ndarray>

        returns:- bbox: <str>
                  classes: <str>
                  scores: <str>
    '''
    bbox =  np.array2string(np.array(data['boxes']), precision=2, separator=',', suppress_small=True)           
    classes =  np.array2string(np.array(data['classes']), precision=2, separator=',', suppress_small=True) 
    scores =  np.array2string(np.array(data['scores']), precision=2, separator=',', suppress_small=True) 

    return bbox, classes, scores

def read_class_names(class_file_name):
    ''' This function opens the classes name file and reads the names from it one by one and stores 
        into dict.

        params:- class_file_name: <str>

        returns: names: <dict>
    '''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def get_class_names(classes):
    ''' This function takes list containing classes_id(int) and converts into list containing the 
        classes name coresponing the class_id using 'read_class_names' function.

        params:- classes: <list>

        returns:- names: <list>
    '''

    labels = read_class_names('coco.names')
    names = ""
    for c in classes:
        names += labels[c] + ", "
    names = names[:-2]
    return names

