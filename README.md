# Raspberry PI Object Detection System

## APIs

### frontend APIs
1. To goto home page: "/"
2. To goto capture&stream page: "/capture/0"
3. To see only live video feed from RPI: "/capture/stream" 
4. To trigger capture and see the detection result: "/capture/1"
5. To goto auto-capture page or turn off auto-capture: "/auto_cap/off"
6. To turn on auto-capture: "auto_cap/on"
7. To goto history page (Displays all detections): "/history"
7. To see particular detection result: "/his_result/<image_id>"
8. To goto settings page: "/settings"
9. To goto manage RPI page: "/manage_rpi"
10. To get image by ID: "/img/<image_id>"
11. To delete image by ID: "/delete/<image_id>"
12. To get config file: "/config"
13. To delete images on RPI: "/delete_imgs/<no_of_imgs>"


### backend APIs
1. To check connection statues: "/api"
2. To capture and get detected objects : "api/capture"
3. To get image by id: "/api/img/<image_id>"
4. To get live video feed: "/api/video_feed"
5. To release/off camera: "/api/release_cam"
6. To get config file from server: "/api/get_config"
7. To on auto-capture: "/api/auto_capture/1"
8. To off auto-capture: "/api/auto_capture/0"
9. To delete bunch of old images: "/api/delete_imgs/<number>
10. To get storage space data: "/api/get_data"


## interactions
### from frontend

1.  '/':
        |--> '/api' :(GET)  To check connection statues
        |--> '/api/get_config' :(POST) To send config file to RPI

2.  'capture/0':
        |--> '/api/release_cam' :(GET) To release camera
    
    'capture/1':
        |--> 'api/capture/' :(GET) To initiate capture and detection process
        |--> 'api/img/<img_id> :(GET) To get image by ID from RPI
        |--> 'api/video_feed' :(GET) To get continues frames

    'capture/stream':
        |--> 'api/video_feed' :(GET) To get continues frames

3.  '/history':
        |--> '/api' :(GET)  To check connection statues
        |--> '/api/release_cam' :(GET) To release camera
        |--> fetch all data from database

4.  '/his_result/<img_id>:
        |--> '/img/<img_id>' :(GET) To get image by ID
        |--> fetch row which has image_id=img_id from database
        
5.  '/delete/<img_id>:
        |--> delete row in database which has image_id=img_id

6.  '/settings' (GET):
        |--> '/api/release_cam' :(GET) To release camera
    '/settings' (POST):  
        |--> accepts form data from settings page

7.  'auto_cap/0:
        |--> '/api/auto_capture/0' :(GET) To turn off auto-capture
    'auto_cap/1:
        |--> '/api/auto_capture/1' :(GET) To turn on auto-capture

8.  '/get_auto' (POST):
        |--> accepts detection data and image
        |--> stores detections into database and image into './images' folder

9.  '/manage_rpi':
        |--> '/api/release_cam' :(GET) To release camera
        |--> '/api/get_data' :(GET) TO get storage info data

10. '/delete_imgs' (POST):
        |--> accepts data about num_images from manage RPI page
        |--> '/api/delete_imgs/<num_imgs> :(GET) To delete num_images from RPI

### from backend

1.  '/api/capture':
        |--> '/config': (GET) To get config file

2.  '/api/auto_capture/0':
        |--> '/config': (GET) To get config file
    '/api/auto_capture/1':
        |--> '/config': (GET) To get config file   
        |--> '/get_auto': (POST) To send detection results and image


## Database schema

=> Image Id: image_id (Text)
=> Bounding Box: bbox (Text)
=> Detection Classes: classes (Text)
=> Detection Scores: scores (Text)
=> Number of Detection: num_det (Text)