# from multiprocessing import Process, Lock, Barrier, Queue
from torch import multiprocessing
from torch.multiprocessing import Process, Lock, Barrier, Queue

from unimodal import audial, visual
from visualization.visualizer import Visualizer

import time
import math

import numpy as np
import heapq


class ModelServer:
    ''' The "server" is a main process that starts two additional processes: the
        'visual' process and the 'audial' process. Each process runs streaming
        inference on the decoded outputs of a video stream, which is done by
        starting decoder subprocesses.

        The inference outputs of the two processes are put into a `Queue`, where
        they are consumed by the main process.

        TODO:
          - Graceful shutdown. Currently one has to re-run the whole pipeline to
            start inference on a new video, which is very annoying. This can be
            done with some form of locking? Since the models have to be
            initialized in their respective thread.

          - Running the pipeline on a local video file and saving the output,
            for demo purposes.
    '''

    def __init__(self, stream_url):
        # multiprocessing.set_start_method('spawn') # CUDA doesn't support fork

        self.queue = Queue() # thread-safe
        self.stream_url = stream_url
        # self.lock = Lock()

        self.barrier = Barrier(2) # barrier that waits for 2 parties
        self.visual_process = Process(target=visual.run, 
                args=(self.stream_url, self.queue, self.barrier))
        self.audial_process = Process(target=audial.run, 
                args=(self.stream_url, self.queue, self.barrier))

        self.visual_process = Process(target=visual.run, 
                args=(self.stream_url, self.queue, None))
        self.audial_process = Process(target=audial.run, 
                args=(self.stream_url, self.queue, None))

        self.visualizer = Visualizer()

        print('Init modelserver done')

 
    def get_attended_objects(self, focus, object_bboxes, classes):
        ''' Returns the object at the `focus` coordinate.

            Epsilon widens the bbox by a fixed amount to accomodate slight
            errors in the gaze coordinates.

        '''
        attended_objects = []
        x, y = focus
        epsilon = 10 # widens the bounding box, pixels

        for i, bbox in enumerate(object_bboxes):
            # exclude person!
            if classes[i] == 'person':
                continue

            left, top, right, bottom = bbox
            if (x > left-epsilon and x < right+epsilon and 
               y < bottom+epsilon and y > top-epsilon):
                attended_objects.append(classes[i])

        return attended_objects
        

    def update_context(self, modality, targets):
        ''' Adds weight to the detected objects: by `alpha` for a single visual
            frame and by `beta` for an audial utterance.
        '''
        if modality not in {'visual', 'audial'}:
            raise ValueError('modality must be one of visual or audial')

        alpha = 0.1
        beta = 0.2
        
        if modality == 'visual':
            denom = 1.0 + len(targets) * alpha
        else:
            denom = 1.0 + len(targets) * beta

        for o in self.objects:
            if o in targets:
                if modality == 'visual':
                    self.context[o] += alpha
                elif modality == 'audial':
                    self.context[o] += beta 

            self.context[o] /= denom

        # sort context in descending order (most significant first)
        self.context = {k:v for k,v in sorted(self.context.items(), 
                key=lambda item: -item[1])}


    def get_recommendations(self):
        ''' Returns a list of recommended words.

            The proportion of words are determined by the context weights.

            A max heap-like structure would be a lot more convenient than
            recalculating weights and sorting every time...
        '''
        N = 12 # Total number of words to recommend

        recommendations = [] # list because ordered

        for obj in self.context:
            weight = self.context[obj]
            # number of targets to recommend for this word
            n = math.ceil(weight * N) 
            top_candidates = heapq.nsmallest(n, self.candidates[obj])

            recommendations.extend([c.word for c in top_candidates])

        return recommendations[:N]

    
    def on_spoken(self, word):
        ''' Action for when a target word is spoken. TODO

            * This isn't the name of the object! It's the candidate.

            The word's relevance should be decreased a bit so that the parent
            diversifies words.
        '''
        self.spoken[word] += 1
        gamma = 1 # amount to decrement the relevance by 

        for obj in self.candidates:
            for cand in self.candidates[obj]:
                if cand.word == word:
                    # decreasing relevance since it is negative
                    cand.relevance += gamma 
                    print(cand.word, cand.relevance)

                    # update order... so convoluted though...
                    heapq.heapify(self.candidates[obj])
        

    def run(self, visualize=False):
        ''' Visualize only works in Jupyter.
        '''
        self.visual_process.start() # start processes
        self.audial_process.start()

        # self.barrier.wait() # wait for all processes to be initialized

        transcript = ''
        image = None
        object_bboxes = []
        object_confidences = []
        object_classnames = []
        face_bboxes = []
        gaze_targets = []

        while (1):
            # This is unnecesssary since queue.get() can be blocking
            # check if something new appeared in queue
            # if self.queue.empty(): 
            #     time.sleep(0.005)
            #     continue

            # This blocks until an item is available
            result = self.queue.get(block=True, timeout=None) 

            if result['from'] == 'image':                    
                image = result['image']
                object_bboxes = result['object_bboxes']
                object_confidences = result['object_confidences']
                object_classnames = result['object_classnames']
                face_bboxes = result['face_bboxes']
                gaze_targets = result['gaze_targets']

                attended_objects = [] # includes both parent & child for now
                for target in gaze_targets:
                    attended_objects.extend(self.get_attended_objects(
                            target, object_bboxes, object_classnames))

                self.update_context('visual', attended_objects)

                if visualize:
                    self.visualizer.clear()
                    self.visualizer.draw_objects(image, object_bboxes, 
                            object_classnames, object_confidences)
                    self.visualizer.draw_face_bboxes(image, face_bboxes)
                    for i, face_bbox in enumerate(face_bboxes):
                        self.visualizer.draw_gaze(image, face_bbox, 
                                gaze_targets[i])

                    image = self.visualizer.add_captions(image, transcript, 
                            attended_objects)
                    self.visualizer.imshow(image)

            elif result['from'] == 'audio':
                # print('confidence {}: {}'.format(result['confidence'], result['transcript']))
                # print('audio') 
                transcript = result['transcript']

                if result['is_final'] == True: # is this a string or boolean?
                    # TODO: pass this to the semantic component which will parse
                    # the sentence, and match the spoken words against the
                    # recommendations 
                    # finally, self.update_context('audial',
                    # spoken_objects)
                    pass


if __name__ == '__main__':  
    server = ModelServer()
    server.run()
