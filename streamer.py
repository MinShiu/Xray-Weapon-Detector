## SocketIO is optional as it acts as a bridge for my detector and webapp to communicate
## Variables to be modified:
## synset_path (line30) - classes.txt that contains all the string of classes on every line
## network_prefix (line43) - path and predix to your model
## resp (line68) - streaming address, note that this is not stream as a video but image snapshot. Use VideoCapture to load vid.

import cv2
import time
import json
import base64
import pickle
import urllib
import socketio
import datetime
import mxnet as mx
import numpy as np
from collections import namedtuple

def handleBoundaries(val, maxval):
    return 0 if val < 0 else int(maxval) if val > maxval else val

# standard Python
sio = socketio.Client()

@sio.on('response')
def response(data):
    print(data)
    global resume
    resume = True

sio.connect('http://localhost:8000')
synset_path = '/home/rock/Desktop/classes.txt'
with open(synset_path, 'r') as f:
    lines = f.readlines()
classes = [l[:-1] for l in lines]

## Variables definition
resume = True
frame_skipping = 2
printOnce = True
current_time = time.time()

## model shd be at deployment mode
network_prefix = '/home/rock/Desktop/detector3'
label_names = ['label']

# Load the network parameters from default epoch 0
sym, arg_params, aux_params = mx.model.load_checkpoint(network_prefix, 0)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, label_names=label_names, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,512,512))])
mod.set_params(arg_params, aux_params)
Batch = namedtuple('Batch', ['data'])

while True:

    ## Waiting for signal to resume detection
    if not resume:
        if printOnce:
            print('Waiting for signal...')
            printOnce = False
        continue
    
    if time.time() - current_time > frame_skipping:
        current_time = time.time() 
        
        resp = urllib.request.urlopen('http://127.0.0.1:8080/?action=snapshot&ignored.mjpg')
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        height, width, _ = frame.shape
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        print('Results:')
        print('Predicting...')
        mod.forward(Batch([mx.nd.array(img)]))
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        
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
            sio.emit('message', data={'weapon': class_name, 'percentage': str(int(round(score*100))),
                                      'img':jpg_as_text.decode()})
            resume = False
        printOnce = True
        print('Results - {} '.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print(predictions)
        
sio.disconnect()
