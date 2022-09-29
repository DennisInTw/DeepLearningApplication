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

    # ================================

def on_connect(client, userdata, flag, rc):
    global state

    if rc != 0:
        state = "ConnFail"
        print("Connection fail!!! , ret = ", rc)
    else:
        state = "ConnOK"
        print("Connection OK!")

def on_message(client, userdata, msg):
    print("[Test] ", msg.payload)

    cmds = json.loads(msg.payload)
    print("[Sub] Received data :", cmds['text'])

state = 'NonInit'
client = mqtt.Client()
cond = threading.Condition()

def subscriberCallback(Arg):
    global client, state, cond
    client.loop_start()

    try:
        while True:
            time.sleep(1)
            cond.acquire()
            if state == "ConnFail":
                break
            else:
                state = "ConnInitialized"
                cond.notify()
                client.subscribe(Arg)
            cond.release()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    state = "ConnFail"
    client.loop_stop()

def publisherCallback(Arg):
    global client, state, cond

    try:
        while True:
            time.sleep(1)
            cond.acquire()
            if state == "ConnFail":
                break
            elif state != "ConnInitialized":
                print("[DBG] Connection not ready, publisher needs to wait")
                cond.wait()
                print("[DBG] Publisher wakes up")
            payload = {'text': 'Atest hello.'}
            client.publish(Arg, json.dumps(payload))
            cond.release()

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    client.loop_stop()

def main():
    # start mqtt subscribe thread
    # ================================
    BROKER = "test.mosquitto.org"
    SUB_TOPIC = "test/topic/topic"

    # 準備好client設定
    client.on_connect = on_connect
    client.on_message = on_message
    print("connecting to broker...", BROKER)
    client.connect(BROKER)

    # 建立一個thread給subscriber使用
    subThread = threading.Thread(target=subscriberCallback, args=(SUB_TOPIC,))


    # ================================
    # send msg to server
    # ================================
    PUB_TOPIC = "test/topic/topic"
    pubThread = threading.Thread(target=publisherCallback,args=(PUB_TOPIC,))
    pubThread.start()
    subThread.start()

    subThread.join()
    pubThread.join()


    print("all done")

    # ================================

    # calculate intrinsic matrix and save in config file
    # ================================

    # ================================

    # undistortion image
    # ================================

    # ================================

if __name__ == '__main__':
    main()
    
