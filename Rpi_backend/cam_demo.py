
import numpy as np
# from PIL import Image, ImageDraw, ImageFont
import cv2


cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot capture source'



frames = 0
inp_dim = 640

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        
        
            
        cv2.imshow('image', frame)
        cv2.imwrite('image.jpg', frame)
           
       
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1

    else:
        break
        
