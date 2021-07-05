import os

# from multiprocessing import Process, Lock, Barrier, Queue
from torch import multiprocessing
from torch.multiprocessing import Process, Queue

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modelserver.unimodal import audial_infinite, visual
from modelserver.visualization.visualizer import Visualizer
from modelserver.guidance.guidance import Guidance

import time
import math

import numpy as np
import heapq

from khaiii import KhaiiiApi

from collections import defaultdict
import re

import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/modelserver/default-demo-app-c4bc3-b67061a9c4b1.json"


guide_file_path = '/workspace/modelserver/modelserver/guidance/demo_9_add_id.csv'

def start(url, queue, is_visualize):
    print('server start')
    server = ModelServer(url, queue)

    server.run(visualize=is_visualize)


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

        ## for target weight decay (ms)
        self.time_decay = int(round(time.time() * 1000)) 

        
        self.visual_process = [ Process(target=visual.run, 
                args=(self.stream_url+str(camera_id), self.queue, None), daemon=True) for camera_id in range(1, 3) ]

        self.audial_process = Process(target=audial_infinite.run, 
                args=('rtmp://video:1935/captivate/test_audio', self.queue, None), daemon=True)

        # ## sync test
        # self.visual_process = [ Process(target=visual.run, args=('rtmp://video:1935/captivate/test', self.queue, None), daemon=True) ]
        # self.audial_process = Process(target=audial_infinite.run, args=('rtmp://video:1935/captivate/test', self.queue, None), daemon=True)
        

        self.visualizer = [ Visualizer(camera_id) for camera_id in range(1, 3) ]

        self.Khaiii_api = KhaiiiApi()

        self.guidance = Guidance(guide_file_path)

        self.objects = self.guidance.get_object_names()

        self.visual_classes = {
            'ball' : '공',
            'dog' : '강아지',
            'cat' : '고양이',
            'shoe' : '신발',
            'spoon' : '숟가락',
            'bowl' : '그릇',
            'fork' : '포크',
            'bus' : '버스',
            'bear' : '곰돌이',
            'bicycle' : '자전거',
            'fish' : '물고기',
            'mirror' : '거울',
            'toothbrush' : '칫솔',
            'sock' : '양말',
            'rabbit' : '토끼',
            'flower' : '꽃'
        }
    
        self.context = self.guidance.get_object_context()

        self.candidates = self.guidance.get_candidates()

        ## init recommendation (first send)
        self.get_recommendations()
        
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
        
        alpha = 0.017 / 4
        beta = 0.05

        if target_length != 0:
            for o in self.objects:
                if modality == 'visual':
                    self.context[o] = self.context[o] * (1-alpha) + target_dist[o] * alpha / target_length
                if modality == 'audial':
                    self.context[o] = self.context[o] * (1-beta) + target_dist[o] * beta / target_length
        
        recommendations = self.get_recommendations()

        return recommendations

    def get_recommendations(self):
        ''' Returns a list of recommended words.

            The proportion of words are determined by the context weights.

            A max heap-like structure would be a lot more convenient than
            recalculating weights and sorting every time...
        '''
        N = 6 # Total number of words to recommend
        # N_h = N/2
        # count = N_h
        count = N
        
        recommendations = [] # list to order


        for item in sorted(self.context.items(), key = lambda x: x[1], reverse=True):
            obj = item[0]
            weight = item[1]
            
            # number of targets to recommend for this word
            n = math.ceil(weight * N)
            if count == 0:
                break
            elif count - n < 0:
                n = count
            count -= n
            
            heap_candidates = heapq.nlargest(int(n), self.candidates[obj].items(), key = lambda x : round(x[1]['weight']))
            
            # top_candidates = [ {c[0] : c[1]['sentence']} for c in heap_candidates]
            for c in heap_candidates:
                recommendations.append(
                    {
                        'object' : obj,
                        'target_word' : c[0],
                        'target_sentence' : c[1]['sentence'],
                        'highlight' : c[1]['highlight'],
                        'id' : c[1]['id'],
                        'color' : c[1]['color']
                    }
                )
        
        # recommendations = sorted(recommendations, key = lambda x: len(x['target_words']), reverse=True)
        
        recommendation_to_queue = {
            'tag' : 'recommendation',
            'recommendations' : recommendations
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
        gamma = 0.5 # amount to decrement the relevance by //0.01
        
        target_spoken = []
        is_spoken_word = 0

        for word in words:
            for obj in self.candidates:
                for cand in self.candidates[obj]:
                    if cand == word:
                        self.candidates[obj][cand]['weight'] = self.candidates[obj][cand]['weight'] - gamma
                        is_spoken_word = 1
            if is_spoken_word:
                target_spoken.append(word)
                is_spoken_word = 0
            
        if len(target_spoken) > 0:
            
            target_to_queue = {
                'tag':'target_words',
                'words':target_spoken
            }

            if self.result_queue:
                self.result_queue.put(target_to_queue)
        return target_spoken

    def decay_target_weights(self, recommendation):

        for target_recommend in recommendation:
            target_word = target_recommend['target_word']

            for obj in self.candidates:
                for cand in self.candidates[obj]:
                    if cand == target_word:
                        self.candidates[obj][cand]['weight'] = self.candidates[obj][cand]['weight'] - 0.1

    def run(self, visualize=False):
        ''' Main loop.
        '''
        # These processes should be joined on error, interrupt, etc.
        [ vp.start() for vp in self.visual_process ]

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

        ## test for audio-video sync
        audio_time = ''
        video_time = ''

        


        while True:
            try:

                ## restart audio process when there is no signal
                if not self.audial_process.is_alive():
                    self.audial_process.start()

                # This blocks until an item is available
                result = self.queue.get(block=True, timeout=None) 

                if result['from'] == 'image':                    
                    image = result['image']
                    object_bboxes = result['object_bboxes']
                    object_confidences = result['object_confidences']
                    object_classnames = result['object_classnames']
                    face_bboxes = result['face_bboxes']
                    gaze_targets = result['gaze_targets']
                    camera_id = result['camera_id']-1
                    frame_num = result['frame_num']
                    video_time = result['video_time']


                    attended_objects = [] # includes both parent & child for now
                    for target in gaze_targets:
                        attended_objects.extend(self.get_attended_objects(
                                target, object_bboxes, object_classnames))
                    
                    target_objects = []
                    for o in attended_objects:
                        if o in self.visual_classes.keys():
                            object_korean = self.visual_classes[o]
                            target_objects.append(object_korean)
                    
                    # update if there's objects
                    if len(target_objects) != 0:
                        recommendations = self.update_context('visual', target_objects)

                    curr_time = int(round(time.time() * 1000))
                    if curr_time - self.time_decay > 6000 :
                        self.decay_target_weights(recommendations)
                        self.time_decay = curr_time

                    if visualize:
                        visualizer_curr = self.visualizer[camera_id]
                        visualizer_curr.draw_objects(image, object_bboxes, 
                                object_classnames, object_confidences)
                        visualizer_curr.draw_face_bboxes(image, face_bboxes)
                        for i, face_bbox in enumerate(face_bboxes):
                            visualizer_curr.draw_gaze(image, face_bbox, 
                                    gaze_targets[i])

                        #test for sync
                        # transcript_sync = video_time+ " "+ transcript
                        image = visualizer_curr.add_captions_recommend(image,transcript,target_spoken)
                        visualizer_curr.visave(image, frame_num)
                    target_spoken.clear()


                elif result['from'] == 'audio':

                    transcript = result['transcript']

                    audio_time = result['audio_time']

                    print(transcript)
                    
                    spoken_words = self.morph_analyze(transcript)
                    
                    spoken_words_update = spoken_words.copy()
                    
                    # print("spoken_words_prev")
                    # print(spoken_words_prev)
                    # print("spoken_words")
                    # print(spoken_words)
                    
                    for word in spoken_words_prev:
                        if word in spoken_words_update:
                            spoken_words_update.remove(word)
                    
                    # print("spoken_words_update")
                    # print(spoken_words_update)
                    
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
                        spoken_words_prev.clear()
                    
                        
                    if not len(spoken_words) < len(spoken_words_prev):
                        spoken_words_prev = spoken_words
            except:
                ## close processes
                self.audial_process.terminate()
                [vp.terminate() for vp in self.visual_process]
                print("exit server run")        

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
