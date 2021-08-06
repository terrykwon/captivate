from modelserver.base.base_predictor import BasePredictor
import insightface
import numpy as np


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class FaceDetector(BasePredictor):

    def __init__(self, gpu_id):
        self.model = insightface.model_zoo.get_model('retinaface_r50_v1')

        # ctx_id: gpu number, non-max suppression threshold, no idea
        gpu_id -= 1
        self.model.prepare(ctx_id=gpu_id, nms=0.4)


    def predict(self, image):
        ''' Gazefollow requires the bounding box
            Recognition requires an affine-transformed square crop (112x112)
            based on landmark detection

            Maybe just detect + recognize in single class
        '''
        predictions, landmarks = self.model.detect(image, threshold=0.5, scale=1.0)
        bboxes = []

        if predictions.shape[0] == 0:
            # no face detected
            return []

        for p in predictions:
            bbox = p[:4]
            confidence = p[4]

            bbox = self._enlarge_bbox(bbox, 0.2)
            bboxes.append(bbox)

        bboxes = sorted(bboxes, key= lambda x : x[1]-x[3])

        return np.array(bboxes[:2])


    def _enlarge_bbox(self, bbox, ratio):
        left, top, right, bottom = bbox
        d = ratio / 2
        
        w = right - left
        h = bottom - top
        left -= d * w
        right += d * w
        top -= d * h
        bottom += d * h

        return [left, top, right, bottom]