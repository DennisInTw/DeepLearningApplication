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

##################################################################################
# 2022/10/11
# Lab1 description :
# 1. Create another thread, subscriber thread, to receive the data from publisher
# 2. Receive 10 images from broker to determine intrinsic parameters
# 3. Remove distortion effect from a testing image
##################################################################################


# MQTT subscriber thread
class Client(threading.Thread):
    def __init__(self, broker):
        threading.Thread.__init__(self)

        self.killswitch = threading.Event()
        self.broker = broker
        logging.info("[DBG] Client object initializes MQTT client setting")

        # TODO2: setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client

        # TODO3: set topic you want to subscribe
        self.topic = "server_response_7"
        self.topic_2 = "secret_photo_7"

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"Connected with result code: {rc}")
        self.client.subscribe(self.topic)
        self.client.subscribe(self.topic_2)

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)

        if 'text' in cmds:
            print(cmds['text'])

        if 'data' in cmds:
            os.makedirs('./q2', exist_ok=True)
            logging.info(f"[DBG] begin receiving 10 images... ...")
            for i in range(len(cmds['data'])):
                # TODO4: save iamge to ./q2, so you have to decode msg
                image = coverToCV2(cmds['data'][i])
                filename = 'photo_' + str(i) + '.png'
                cv2.imwrite(os.path.join('./q2', filename), image)
                logging.info(f"[DBG] R/W {filename}")

        if 'dist_data' in cmds:
            logging.info("[DBG] begin receiving a testing distorted image... ...")
            # TODO5: you have to decode msg
            image = coverToCV2(cmds['dist_data'])
            cv2.imwrite('dist_image.png', image)

    def stop(self):
        self.killswitch.set()

    def run(self):
        logging.info("[DBG] Subscriber thread starts to run")
        try:
            self.client.loop_start()
            logging.info("[DBG] Subscriber thread is blocking for event...")
            self.killswitch.wait()
        finally:
            self.client.loop_stop()


"""
use this function to decode msg from server
[Description]
   This function will convert the received string (i.e. data) to bytes by base64.b64decode()
   Then, use pickle’s load method to convert from a bytes-array type back to the original object
   Last, read an image from a buffer in memory
"""
def coverToCV2(data):
    imdata = base64.b64decode(data)
    buf = pickle.loads(imdata)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image


"""
save config to './config'
"""
def saveConfig(cfg_file, config):
    try:
        with open(cfg_file, 'w') as configfile:
            config.write(configfile)
    except IOError as e:
        logging.error(e)
        sys.exit()


"""
load config from './config'
[Description]
   This function will read ./config as dict type by configparser.ConfigParser()
"""
def loadConfig(cfg_file):
    try:
        config = configparser.ConfigParser()
        config.optionxform = str      # This will let [Intrinsic] be case-sensitive, but it seems always case-sensitive
        with open(cfg_file) as f:
            config.read_file(f)
    except IOError as e:
        logging.error(e)
        sys.exit()
    return config

"""
This function will determine intrinsic parameters of camera
[Description]
   First, define the real world coordinates of each corners
   Second, use cv2.findChessboardCorners() to get coordinates of corners in the images
   Third, determine intrinsic parameters of camera
   Fourth, set alpha=0 to get an undistorted image with minimum unwanted pixels, and also get new intrinsic parameters for that new image
"""
def calculateIntrinsic():
    config_file = './config'
    cameraCfg = loadConfig(config_file)

    # calculate mtx, dist, newcameramtx here
    # ================================

    # threhold
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # TODO11: set corner number
    corner_x = 7
    corner_y = 7

    objp = np.zeros((corner_x * corner_y, 3), np.float32)                # A 2-D float-32 type array [49, 3] with initialized values all 0
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)      # Here we define the real world coordinates for corners (Z coordinate is 0)
                                                                         # 2x7x7 ==> Transpose ==> 7x7x2 ==> reshape ==> 49x2

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = []  # Store file names of 10 images (photo_0.png, etc...)
    for filename in os.listdir('./q2'):
        fullpath = os.path.join('./q2', filename)
        if filename.endswith('.png'):
            images.append(fullpath)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)

        # TODO12: convert image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO13: find image point by opencv
        # If find desired number of corners, then ret will be True; otherwise, will be false
        # In this lab case, it should be 7x7 = 49
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        if ret == True:
            ## @TODO:for more accurate
            # winSize = (5, 5) for the search window
            # zeroZone = (-1, -1)
            # We change winSize to get a bit of different result
            refined_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            ##
            objpoints.append(objp)
            #imgpoints.append(corners)
            # @TODO: for more accurate
            imgpoints.append(refined_corners)


    # width  => img.shape[1]
    # height => img.shape[0]
    img_size = (img.shape[1], img.shape[0])

    # TODO14: get camera matrix and dist matrix by opencv
    # Camera is fixed, so get intrinsic matrix, lens distortion coef. and rotation/translation vectors
    # Actually, I have no idea why we don't need to use rvecs and tvecs
    # The official tutorial doesn't describe more, either
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # TODO15: get optimal new camera matrix, alpha = 0
    # Here we use the new image size as the same size of img_size
    # Also, we set alpha=0 to have an undistorted image with minimum unwanted pixels
    # BTW, if we set alpha=1, then we can crop the image from roi information
    new_img_size = img_size
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, 0, new_img_size)
    # ================================

    cameraCfg['Intrinsic']['ks'] = str(mtx.tolist())
    cameraCfg['Intrinsic']['dist'] = str(dist.tolist())
    cameraCfg['Intrinsic']['newcameramtx'] = str(newcameramtx.tolist())
    saveConfig(config_file, cameraCfg)
    return objpoints, imgpoints, img_size


"""
This function removes distort effect
"""
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
    topic = "deep_learning_lecture_7"

    # TODO7: send msg to get echo
    payload = {'text': 'test hello'}
    logging.info(f'[DBG] To get echo => sendMsg:{payload}')
    client.publish(topic, json.dumps(payload))

    # TODO8: send msg to get hint
    payload = {'request': 'photo'}
    client.publish(topic, json.dumps(payload))
    logging.info(f'[DBG] To get hint => sendMsg:{payload}')
    # receive the hint message
    # "subscribe topic [secret_photo], send msg [request] for [EC234_NOL], and receive [data] to get 10 photo"

    # TODO9: send msg to get 10 photos
    payload = {'request': 'EC234_NOL'}
    client.publish(topic, json.dumps(payload))
    logging.info(f'[DBG] To get 10 images => sendMsg:{payload}')

    # TODO10: send msg to get 1 photo to undistort
    payload = {'request': 'dist_photo'}
    client.publish(topic, json.dumps(payload))
    logging.info(f'[DBG] To get 1 testing distorted image => sendMsg:{payload}')


"""
This function calculate the average error for two coordinate of images
[Description]
   objpoints  : 3D coordinates of images
   imgpoints  : 2D coordinates of images from cv2.cornerSubPix
   imgpoints2 : 2D coordinates of images from cv2.projectPoints
"""
def calculateTotalError(objpoints, imgpoints, img_size):
    # get camera matrix and dist matrix by opencv
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    total_error = 0
    for i in range(len(objpoints)):
        # Project 3D coordinate to 2D coordinate
        # Because z axis is always 0, we can do this
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # Clculate the mean square root of sum of squares
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        
        total_error += error
    print("Avg error:", total_error / len(objpoints))


def main():
    # start mqtt subscribe thread
    # ================================
    # TODO1: set broker ip to connect
    broker_ip = "140.113.208.103"
    mainThread = Client(broker_ip)  # Create a new thread
    mainThread.start()              # Start running thread
    # ================================

    # send msg to server
    # ================================
    sendMsg(broker_ip)              # Another thread will be Publisher to send the data to broker
    # ================================

    time.sleep(3)

    # calculate intrinsic matrix and save in config file
    # ================================
    objpoints, imgpoints, img_size = calculateIntrinsic()
    # ================================

    # undistortion image
    # ================================
    undistImage()
    # ================================

    calculateTotalError(objpoints, imgpoints, img_size)

    mainThread.join()


if __name__ == '__main__':
    # Set debug level
    # Print logging.info by level=logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

    main()
