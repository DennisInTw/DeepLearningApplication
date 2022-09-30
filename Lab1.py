import threading
import logging
import json
import os
import sys
import cv2
import paho.mqtt.client as mqtt
import pickle
import base64
import configparser
import numpy as np
import time

# MQTT subscriber thread
class Client(threading.Thread):
    def __init__(self, broker):
        threading.Thread.__init__(self)

        self.killswitch = threading.Event()
        self.broker = broker
        print("[DBG] Sub init")

        # TODO2: setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client

        # TODO3: set topic you want to subscribe
        self.topic = "server_response_7"    #"test/topic/topic"
        self.topic_2 = "secret_photo_7"

    def on_connect(self, client, userdata, flag, rc):
        print(f"Connected with result code: {rc}")
        self.client.subscribe(self.topic)
        self.client.subscribe(self.topic_2)

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)

        if 'text' in cmds:
            print(cmds['text'])

        if 'data' in cmds:
            os.makedirs('./q2', exist_ok=True)
            print("[DBG1]")
            for i in range(len(cmds['data'])):
                print("[DBG] ")
                # TODO4: save iamge to ./q2, so you have to decode msg
                image = coverToCV2(cmds['data'][i])
                filename = 'photo_' + str(i) + '.png'
                cv2.imwrite(os.path.join('./q2', filename), image)

        if 'dist_data' in cmds:
            print("[DBG] dist_data")
#            # TODO5: you have to decode msg
            image = coverToCV2(cmds['dist_data'])
            cv2.imwrite('dist_image.png', image)

    def stop(self):
        self.killswitch.set()

    def run(self):
        try:
            print("[DBG] run")
            self.client.loop_start()
            print("[DBG] wait")
            self.killswitch.wait()
            print("[DBG] ready")
        finally:
            self.client.loop_stop()

# use this function to decode msg from server
def coverToCV2(data):
    imdata = base64.b64decode(data)
    buf = pickle.loads(imdata)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image

# save config to './config'
def saveConfig(cfg_file, config):
    try:
        with open(cfg_file, 'w') as configfile:
            config.write(configfile)
    except IOError as e:
        logging.error(e)
        sys.exit()

# load config from './config'
def loadConfig(cfg_file):
    try:
        config = configparser.ConfigParser()
        config.optionxform = str
        with open(cfg_file) as f:
            config.read_file(f)
    except IOError as e:
        logging.error(e)
        sys.exit()
    return config

def calculateIntrinsic():
    config_file = './config'
    cameraCfg = loadConfig(config_file)

    # calculate mtx, dist, newcameramtx here
    # ================================

    # TODO11: set corner number
    corner_x = 7
    corner_y = 7

    objp = np.zeros((corner_x*corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    images = []
    for filename in os.listdir('./q2'):
        fullpath = os.path.join('./q2', filename)
        if filename.endswith('.png'):
            images.append(fullpath)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)

        # TODO12: convert image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO13: find image point by opencv
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])

    # TODO14: get camera matrix and dist matrix by opencv
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # TODO15: get optimal new camera matrix, alpha = 0
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, 0, img_size)
    # ================================

    cameraCfg['Intrinsic']['ks'] = str(mtx.tolist())
    cameraCfg['Intrinsic']['dist'] = str(dist.tolist())
    cameraCfg['Intrinsic']['newcameramtx'] = str(newcameramtx.tolist())
    saveConfig(config_file, cameraCfg)

def undistImage():
    config_file = './config'
    cameraCfg = loadConfig(config_file)

    # undistortion image here
    # ================================
    image = 'dist_image.png'
    ks = np.array(json.loads(cameraCfg['Intrinsic']['ks']))
    dist = np.array(json.loads(cameraCfg['Intrinsic']['dist']))
    newcameramtx = np.array(json.loads(cameraCfg['Intrinsic']['newcameramtx']))
    src = cv2.imread(image)

    # TODO16: undistortion image by opencv
    dst = cv2.undistort(src, ks, dist, None, newcameramtx)

    output_name = 'D_' + image
    cv2.imwrite(output_name, dst)
    # ================================

def sendMsg(broker):
    # TODO6: setup MQTT client
    client = mqtt.Client()
    client.connect(broker)
    topic = "deep_learning_lecture_7"      #"test/topic/topic"

    # TODO7: send msg to get echo
    payload = {'text': 'test hello'}
    client.publish(topic, json.dumps(payload))

    # TODO8: send msg to get hint
    payload = {'request': 'photo'}
    client.publish(topic, json.dumps(payload))

    # TODO9: send msg to get 10 photos
    payload = {'request': 'EC234_NOL'}
    client.publish(topic, json.dumps(payload))

    # TODO10: send msg to get 1 photo to undistort
    payload = {'request': 'dist_photo'}
    client.publish(topic, json.dumps(payload))

def main():
    # start mqtt subscribe thread
    # ================================
    # TODO1: set broker ip to connect
    broker_ip = "140.113.208.103"#"test.mosquitto.org"
    mainThread = Client(broker_ip)   # Create a new thread
    mainThread.start()   # Start running thread
    # ================================

    # send msg to server
    # ================================
    sendMsg(broker_ip)
    # ================================

    time.sleep(3)

    # calculate intrinsic matrix and save in config file
    # ================================
    calculateIntrinsic()
    # ================================

    # undistortion image
    # ================================
    undistImage()
    # ================================

    mainThread.join()

if __name__ == '__main__':
    main()
