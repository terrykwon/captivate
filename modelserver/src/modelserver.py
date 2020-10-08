import multiprocessing as mp
from multiprocessing import Process, Lock

from unimodal import audial, visual
from visualization.visualizer import Visualizer

import numpy as np


class ModelServer:
    ''' The "server" is a main process that starts two additional processes: the 'visual' process
        and the 'audial' process. Each process runs streaming inference on the decoded outputs of
        a video stream, which is done by starting decoder subprocesses.

        The inference outputs of the two processes are put into a Queue, where they are consumed
        by the main process.

        TODO:
          - Separate initialization of models, run, and graceful shutdown. Currently one has to
            re-run the whole pipeline to start inference on a new video, which is very annoying.
            This can be done with some form of locking? Since the models have to be initialized in
            their respective thread.

          - Running the pipeline on a local video file and saving the output, for demo purposes.
        
    '''

    def __init__(self, stream_url):
        self.queue = mp.Queue() # thread-safe
        self.stream_url = stream_url
        # self.lock = Lock()

        self.visual_process = Process(target=visual.run, args=(self.stream_url, self.queue,))
        self.audial_process = Process(target=audial.run, args=(self.stream_url, self.queue,))

        self.visualizer = Visualizer()

 
    def get_attended_objects(self, focus, object_bboxes, classes):
        ''' Returns the object at the `focus` coordinate.

            Epsilon widens the bbox by a fixed amount to accomodate slight errors in the gaze
            coordinates.

        '''
        attended_objects = []
        x, y = focus
        epsilon = 10 # widens the bounding box, pixels

        for i, bbox in enumerate(object_bboxes):
            # exclude person!
            if classes[i] == 'person':
                continue

            left, top, right, bottom = bbox
            if x > left-epsilon and x < right+epsilon and y < bottom+epsilon and y > top-epsilon:
                attended_objects.append(classes[i])

        return attended_objects


    def run(self, visualize=False):
        ''' Visualize only works in Jupyter.
        '''
        self.visual_process.start() # start processes
        self.audial_process.start()

        transcript = ''
        image = None
        object_bboxes = []
        object_confidences = []
        object_classnames = []
        face_bboxes = []
        gaze_targets = []

        while (1):
            # check if something new appeared in queue
            if self.queue.empty():
                # sleep here?
                continue

            result = self.queue.get() # is this blocking? nope

            if result['from'] == 'image':                    
                image = result['image']
                object_bboxes = result['object_bboxes']
                object_confidences = result['object_confidences']
                object_classnames = result['object_classnames']
                face_bboxes = result['face_bboxes']
                gaze_targets = result['gaze_targets']

                attended_objects = [] # includes both parent & child for now
                for target in gaze_targets:
                    attended_objects.extend(self.get_attended_objects(target, object_bboxes, object_classnames))

                if visualize:
                    self.visualizer.clear()
                    self.visualizer.draw_objects(image, object_bboxes, object_classnames, object_confidences)
                    self.visualizer.draw_face_bboxes(image, face_bboxes)
                    for i, face_bbox in enumerate(face_bboxes):
                        self.visualizer.draw_gaze(image, face_bbox, gaze_targets[i])

                    image = self.visualizer.add_captions(image, transcript, attended_objects)
                    self.visualizer.imshow(image)

            elif result['from'] == 'audio':
                # print('confidence {}: {}'.format(result['confidence'], result['transcript']))
                # print('audio')
                transcript = result['transcript']


if __name__ == '__main__':  
    server = ModelServer()
    server.run()
