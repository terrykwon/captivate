# for Django channels

import os
import json
from threading import Thread, Event
from datetime import datetime
from channels.generic.websocket import WebsocketConsumer
from django.contrib.auth.models import User

import cv2
import numpy as np
import redis

from .models import ContextRecord


class FrameProcessThread(Thread):
    ''' This thread connects to the RTMP video stream and
        save frames to the redis cache.
    '''
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

        self.db = redis.Redis(host='redis', port=6379, db=0)

    def run(self):
        # self.rtmp_addr = "rtmp://192.168.219.101/live/terry"
        self.rtmp_addr = "rtmp://video:1935/live/terry"
        self.video_capture = cv2.VideoCapture(self.rtmp_addr)

        if self.video_capture is None or not self.video_capture.isOpened():
            print('RTMP connection not available!')
            return

        while not self.stopped.wait(1.0):
            # print('ping!')

            success, frame = self.video_capture.read()
            if not success:
                print('frame not read')
                continue

            # self.db.rpush(os.environ.get("IMAGE_QUEUE", "image_queue"), "hello")
            self.db.rpush(os.environ.get("IMAGE_QUEUE"), frame.tobytes())
            print('frame read!', frame.shape)


class FrameProcessConsumer(WebsocketConsumer):

    def connect(self):
        username = self.scope['url_route']['kwargs']['username']
        self.user = User.objects.get(username=username)

        self.stop_flag = Event()

        self.accept()

    def disconnect(self, close_code):
        self.connected = False
        self.stop_flag.set() # stop the thread
        print('FrameProcessConsumer disconnected with code', close_code)

    def receive(self, text_data):
        print(text_data)
        data = json.loads(text_data)
        if data['connect'] == True:
            # Trigger async "producer" process that fetches
            # frames from the RTMP stream.
            self.connected = True
            self.stop_flag.clear() # unset the flag

            self.process_frames()
        elif data['connect'] == False:
            self.stop_flag.set() # this will stop the thread?

    def process_frames(self):
        self.thread = FrameProcessThread(self.stop_flag)
        self.thread.start()
        # This will cause a crash!!
        # while(1):
        #     print(datetime.time())


# Echo consumer (synchronous)
class EchoConsumer(WebsocketConsumer):
    def connect(self):
        # very verbose... 
        self.username = self.scope['url_route']['kwargs']['username']
        self.user = User.objects.get(username=self.username)
        self.accept()

    def disconnect(self, close_code):
        print('EchoConsumer disconnected')

    # the parameter name `text_data` cannot be changed
    # because receive is called with keyword arguments
    def receive(self, text_data):
        print(text_data)
        data = json.loads(text_data)
        context = data['context']
        time = data['time'] # UTC timestamp in millis

        time = datetime.utcfromtimestamp(time / 1000.0)
        print(time)
        print(type(time))

        self.send(text_data=json.dumps({
            'context': context
        }))

        record = ContextRecord(
            interest=context,
            time=time,
            user=self.user # must be a User instance 
        )

        record.save()
