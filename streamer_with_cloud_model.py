import cv2
import time
import json
import base64
import urllib
import socketio
import datetime
import sagemaker
import numpy as np

## Function to handle boundaries values when cropping image
def handleBoundaries(val, maxval):
    return 0 if val < 0 else int(maxval) if val > maxval else val

## Sagemaker endpoint predictor instance
sagemaker_session = sagemaker.Session()
predictor = sagemaker.predictor.RealTimePredictor(endpoint='model3-endpoint-deployment', 
                                                  sagemaker_session=sagemaker_session,
                                                  content_type='image/png')

## Read classes from text file 
synset_path = '/home/rock/Desktop/classes.txt'
with open(synset_path, 'r') as f:
    lines = f.readlines()
classes = [l[:-1] for l in lines]

## Variables definition
resume = True
frame_skipping = 2
printOnce = True
current_time = time.time()

## Socket connection
sio = socketio.Client()

@sio.on('response')
def response(data):
    print(data)
    global resume
    resume = True

sio.connect('http://localhost:8000')

## Main script
while True:    
    
    ## Waiting for signal to resume detection
    if not resume:
        if printOnce:
            print('Waiting for signal...')
            printOnce = False
        continue
    
    ## Detection happened here for every 2 seconds
    if resume and (time.time() - current_time > frame_skipping):
        current_time = time.time() 
        
        ## Read image from streaming url
        resp = urllib.request.urlopen('http://127.0.0.1:8080/?action=snapshot&ignored.mjpg')
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        height, width, _ = frame.shape
        
        ## Perform prediction using sagemaker endpoint
        retval, buffer = cv2.imencode('.png', frame)
        b = bytearray(buffer)
        results = predictor.predict(b)
        prob = json.loads(results)['prediction']

        ## Handle response from endpoint
        predictions = list()
        for det in prob:
            (klass, score, x1, y1, x2, y2) = det
            if klass == -1:
                continue
            if score < 0.5:
                break
            class_name = classes[int(klass)]
            
            xmin = handleBoundaries(int(x1*width)-30, width) 
            ymin = handleBoundaries(int(y1*height)-30, height)
            xmax = handleBoundaries(int(x2*width)+30, width) 
            ymax = handleBoundaries(int(y2*height)+30, height)
            
            detected_frame = frame[ymin:ymax, xmin:xmax]
            retval, buffer = cv2.imencode('.jpg', detected_frame)
            jpg_as_text = base64.b64encode(buffer)
            predictions.append([class_name, score])  
            sio.emit('data', data={'weapon': class_name, 'percentage': str(int(round(score*100))),
                                      'img':jpg_as_text.decode()})
            resume = False
        printOnce = True
        print('Results - {} '.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print(predictions)
      
sio.disconnect()