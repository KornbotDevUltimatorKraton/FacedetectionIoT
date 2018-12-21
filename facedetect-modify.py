#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import math 
# local modules
from video import create_capture
from common import clock, draw_str
import microgear.client as microgear
import time
import logging
appid = "Kornbot"
gearkey = "wBHqON1EtNqlTzu"
gearsecret =  "nt0utSlDrPEOiYOFFfHYJDbEw"


microgear.create(gearkey,gearsecret,appid,{'debugmode': True})

deg = 0   # Setting the degree calculate path 
centre_x = 0 
centre_y = 0  

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
def connection():
   logging.info("Now I am connected with netpie")

def subscription(topic,message):
   logging.info(topic+" "+message)
def disconnect():
    
    logging.debug("disconnect is work")

microgear.setalias("VisualStudio")

microgear.on_connect = connection

microgear.on_message = subscription

microgear.on_disconnect = disconnect

microgear.subscribe("/Topic")

microgear.connect(False) 
 
if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cascade_fn)
    nested = cv.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                print("Facedetcted")
                print(int(x1),int(y1))
                print(int(x2),int(y2))
                centre_x = int(x1) + 340 
                centre_y = int(y1) + 240 
                deg = 2*math.degrees(math.acos( (int(y1)+240)/(math.hypot(centre_x,centre_y)))) 
                print("Degree:")
                print(deg)
                microgear.chat("VisualStudio",deg)
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
