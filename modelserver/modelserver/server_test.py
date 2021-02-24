# from multiprocessing import Process, Lock, Barrier, Queue
from torch import multiprocessing
from torch.multiprocessing import Process, Lock, Barrier, Queue

import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modelserver.unimodal import audial_infinite, visual
from modelserver.visualization.visualizer import Visualizer

import time
import math

import numpy as np
import heapq

from khaiii import KhaiiiApi

from collections import defaultdict
import re


import asyncio
import websockets
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/modelserver/default-demo-app-c4bc3-b67061a9c4b1.json"

url = 'rtmp://video:1935/captivate/test'
# url ='/workspace/modelserver/test.flv'

data_queue = Queue()



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

    def __init__(self, stream_url, result_queue):
        # this method must be only called once
        # multiprocessing.set_start_method('spawn') # CUDA doesn't support fork
        # but doesn't matter if CUDA is initialized on a single process
        self.result_queue = result_queue
        
        self.stream_url = stream_url
        self.queue = Queue() # thread-safe


        self.visual_process = Process(target=visual.run, 
                args=(self.stream_url, self.queue, None))
        self.audial_process = Process(target=audial_infinite.run, 
                args=(self.stream_url, self.queue, None))

        self.visualizer = Visualizer()

        self.Khaiii_api = KhaiiiApi()


        self.objects = ['빵빵','아빠','밀가루'] 
        self.context = {obj : 1/len(self.objects) for obj in self.objects}
        self.candidates = {
            '빵빵' : {'어디': 0.1, '가다' : 0.1, '자동차' : 0.1,'발':0.1, '신호등':0.1, '버스':0.1, '앉다':0.1, '비키다':0.1, '길':0.1, '손':0.1},
            '아빠' : {'손': 0.1,'빨리': 0.1,'엄마': 0.1, '밀가루':0.1, '빵빵':0.1,'아기':0.1,'인형':0.1,'어디':0.1,'남자':0.1,'빠빠이':0.1},
            '밀가루' : {'꼭': 0.1 , '동그라미' : 0.1, '누르다': 0.1,'숟가락':0.1, '주다':0.1,'날리다':0.1,'식빵':0.1,'음식':0.1,'하얗다':0.1,'손':0.1}
        }


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
        

        ## get distribution
        target_length = 0

        target_dist = defaultdict(float)
        for t in targets:
            if t in self.objects:
                target_dist[t] += 1
                target_length += 1
        
        alpha = 0.1
        beta = 0.1

        if target_length != 0:
            for o in self.objects:
                if modality == 'visual':
                    self.context[o] = self.context[o] * (1-alpha) + target_dist[o] * alpha / target_length
                if modality == 'audial':
                    self.context[o] = self.context[o] * (1-beta) + target_dist[o] * beta / target_length
        
        recommendations = self.get_recommendations()
        print(self.context)
        print(self.candidates)

        return recommendations

    def get_recommendations(self):
        ''' Returns a list of recommended words.

            The proportion of words are determined by the context weights.

            A max heap-like structure would be a lot more convenient than
            recalculating weights and sorting every time...
        '''
        N = 12 # Total number of words to recommend
        N_h = N/2
        count = N_h
        
        recommendations = [] # list to order


        for item in sorted(self.context.items(), key = lambda x: x[1], reverse=True):
            obj = item[0]
            weight = item[1]
            
            # number of targets to recommend for this word
            n = math.ceil(weight * N_h)
            if count == 0:
                break
            elif count - n < 0:
                n = count
            count -= n
            
            heap_candidates = heapq.nlargest(int(n*2), self.candidates[obj].items(), key = lambda x : x[1])
            top_candidates = [ c[0] for c in heap_candidates]
            recommendations.append({'object' : obj, 'target_words' : top_candidates})
        
        recommendations = sorted(recommendations, key = lambda x: len(x['target_words']), reverse=True)
        
        recommendation_to_queue = {
            'tag' : 'recommendation',
            'recommendation' : recommendations
        }
        
        if self.result_queue:
            self.result_queue.put(recommendation_to_queue)
            
        return recommendations

    
    def on_spoken(self, words):
        ''' Action for when a target word is spoken. TODO

            * This isn't the name of the object! It's the candidate.

            The word's relevance should be decreased a bit so that the parent
            diversifies words.
        '''
        gamma = 0.1 # amount to decrement the relevance by 
        
        target_spoken = []

        for word in words:
            for obj in self.candidates:
                for cand in self.candidates[obj]:
                    if cand == word:
                        self.candidates[obj][cand] = round(self.candidates[obj][cand] - gamma,1)
                        target_spoken.append(word)
            
        if len(target_spoken) > 0:
            
            target_to_queue = {
                'tag':'target_words',
                'words':target_spoken
            }

            if self.result_queue:
                self.result_queue.put(target_to_queue)
        return target_spoken

    def run(self, visualize=False):
        ''' Main loop.
            Visualize only works in Jupyter.
        '''
        # These processes should be joined on error, interrupt, etc.
        self.visual_process.start() # start processes
        self.audial_process.start()
        print('process start')
        # This is unnecessary because the queue.get() below is blocking anyways
        # self.barrier.wait()

        transcript = ''
        spoken_words_prev = []
        spoken_words_update = []
        target_spoken = []

        recommendations = []

        image = None
        object_bboxes = []
        object_confidences = []
        object_classnames = []
        face_bboxes = []
        gaze_targets = []


        while (1):
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
                
                target_objects = []
                for o in attended_objects:
                    if o in self.objects:
                        target_objects.append(o)
                
                # update if there's objects
                if len(target_objects) != 0:
                    recommendations = self.update_context('visual', target_objects)

                if visualize:
                    self.visualizer.clear()
                    self.visualizer.draw_objects(image, object_bboxes, 
                            object_classnames, object_confidences)
                    self.visualizer.draw_face_bboxes(image, face_bboxes)
                    for i, face_bbox in enumerate(face_bboxes):
                        self.visualizer.draw_gaze(image, face_bbox, 
                                gaze_targets[i])

                    image = self.visualizer.add_captions_recommend(image, transcript, target_spoken, recommendations)
                    # self.visualizer.imshow(image)
                    self.visualizer.visave(image)
                target_spoken.clear()

            elif result['from'] == 'audio':
                transcript = result['transcript']
                
                spoken_words = self.morph_analyze(transcript)
                
                spoken_words_update = spoken_words.copy()
                
                for word in spoken_words_prev:
                    if word in spoken_words_update:
                        spoken_words_update.remove(word)
                
                # update spoken & target word weight
                spoken = self.on_spoken(spoken_words_update)
                if len(spoken) != 0:
                    target_spoken = spoken


                # if transcript is final
                if result['is_final']: 
                    spoken_objects = []

                    # update spoken objects list
                    for word in spoken_words:
                        if word in self.objects:
                            spoken_objects.append(word)

                    # update object context
                    if len(spoken_objects) != 0 :
                        recommendations = self.update_context('audial', spoken_objects)
                    
                    spoken_words.clear()
                        
                spoken_words_prev = spoken_words
                        
    def morph_analyze(self, transcript):
        spoken_words = []
        
        try:
            line_pos = self.Khaiii_api.analyze(transcript)

            for w in line_pos:
                for m in w.morphs:
                    if m.tag in ['NNG', 'NR', 'MAG']:
                        spoken_words.append(m.lex)
                    elif m.tag in ['VV', 'VA']:
                        spoken_words.append(m.lex + '다')
                    elif m.tag in ['XR']:
                        spoken_words.append(m.lex+'하다')
                    else :
                        spoken_words.append(m.lex)
        except:
            print('morph')
        
        return spoken_words


def start(url, queue, is_visualize):
    print('server start')
    server = ModelServer(url, queue)

    server.run(visualize=is_visualize)

async def pop_and_send(websocket, path):
    print("websocket start!")

    while(1):
        data = data_queue.get(block=True)
        data_json = json.dumps(data,ensure_ascii=False)

        await websocket.send(data_json)
        print('data_send!')
        print(data_json)
        

if __name__ == '__main__': 
    
    server_process = Process(target=start, 
                args=(url, data_queue, True))
    
    server_process.start()

    # while (1):
    #     result = data_queue.get(block=True)
        # print(result)
        # print('\n')
    websocket_server = websockets.serve(pop_and_send, '0.0.0.0', 8888, ping_interval = None)

    asyncio.get_event_loop().run_until_complete(websocket_server)
    asyncio.get_event_loop().run_forever()
    
    
    
