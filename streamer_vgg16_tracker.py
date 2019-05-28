import cv2
import time
import json
import math
import base64
import socketio
import datetime
import mxnet as mx
import numpy as np
from collections import namedtuple

tracker = cv2.TrackerKCF_create()

def handleBoundaries(val, maxval):
    return 0 if val < 0 else int(maxval) if val > maxval else val

sio = socketio.Client()

@sio.on('response')
def response(data):
    print('Received message: ' + data)
    global resume
    if data == 'start':
        resume = True
    if data == 'stop':
        resume = False

#sio.connect('http://172.20.10.13:7000')
synset_path = '/home/minshiu/aws/lauretta-sagemaker-XRAY/classes.txt'
with open(synset_path, 'r') as f:
    lines = f.readlines()
classes = [l[:-1] for l in lines]

## Variables definition
resume = True
sleep_5_secs = False
frame_skipping = 1
printOnce = True
current_time = time.time()
desire_output = ['knife', 'guns', 'scissors', 'other_weapon', 'metal_pipes', 'catridges', 'rifles', 'other']

# **model shd be at deployment mode
network_prefix = '/home/minshiu/aws-lauretta-sagemaker-XRAY/model/detector3'
label_names = ['label']

# Load the network parameters from default epoch 0
sym, arg_params, aux_params = mx.model.load_checkpoint(network_prefix, 0)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, label_names=label_names, context=mx.gpu(0))
mod.bind(for_training=False, data_shapes=[('data', (1,3,512,512))])
arg_params['prob_label'] = mx.nd.array([0])
mod.set_params(arg_params, aux_params)
Batch = namedtuple('Batch', ['data'])

#vid = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream&ignored.mjpg')
vid = cv2.VideoCapture('/home/minshiu/Downloads/video.mp4')
current_time = time.time()
bbox = None

while True:

    ## Waiting for signal to resume detection
    if not resume:
        if printOnce:
            print('Waiting for signal...')
            printOnce = False
        continue

    ret, frame = vid.read()
    if not ret:
        break
    frame = cv2.resize(frame, (512, 512))[:, 55:]
    height, width, _ = frame.shape
    
    (success, box) = tracker.update(frame)
    frame_cp = frame
    if success:
        xm = int(box[0])
        ym = int(box[1])
        w = int(box[2])
        h = int(box[3])
        p1 = (xm, ym)
        p2 = (handleBoundaries(xm+w, width), handleBoundaries(ym+h, height))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        #frame_cp = frame[:, :p1[0]]
        frame_cp = frame[:, p2[0]:]
        #if p1[0] <= 0:
        if p2[0] >= width:
            #frame_cp = frame[:, :p1[0]+20]
            frame_cp = frame[:, p2[0]-20:]
    
    cv2.imshow('test frame', frame_cp)
    
    if resume and (time.time() - current_time > frame_skipping):
        print('Success: ' + str(success))
        current_time = time.time()
        img = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        print('Results:')

        mod.forward(Batch([mx.nd.array(img)]))
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        print(prob)
        predictions = list()
        prob_list = []

        weapon_list = []
        percentage_list = []
        image_list = []

        for det in prob:

            (klass, score, x1, y1, x2, y2) = det
            if klass == -1:
                continue

            if score < 0.3:
                break
            class_name = classes[int(klass)]

            xmin = handleBoundaries(int(x1*width)-30, width) 
            ymin = handleBoundaries(int(y1*height)-30, height)
            xmax = handleBoundaries(int(x2*width)+30, width) 
            ymax = handleBoundaries(int(y2*height)+30, height)
            prob_list.append(tuple([class_name, xmin, ymin, xmax-xmin, ymax-ymin]))

            detected_frame = frame[ymin:ymax, xmin:xmax]
            retval, buffer = cv2.imencode('.jpg', detected_frame)
            jpg_as_text = base64.b64encode(buffer)
            weapon_list.append(class_name)
            percentage_list.append(str(int(round(score*100))))
            image_list.append(jpg_as_text.decode())

        if len(prob_list) > 0:
            #bbox = min(prob_list, key=lambda x:x[1])[1:]
            bbox = max(prob_list, key=lambda x:x[1])[1:]
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bbox)

        print('Results - {} '.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print(predictions)

        if len(weapon_list) == 0:
            continue
        elif len(weapon_list) == 1:
            weapon_str = weapon_list[0]
            percentage_str = percentage_list[0]
            image_str = image_list[0]
        else:
            weapon_str = '._____.'.join(weapon_list)
            percentage_str = '._____.'.join(percentage_list)
            image_str = '._____.'.join(image_list)
        
        #sio.emit('data', {'weapon': weapon_str, 'percentage': percentage_str, 'img': image_str})
        print('Emitted')
        #resume = False
        printOnce = True

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == 27:
        break 
    
sio.disconnect()
vid.release()
cv2.destroyAllWindows()
