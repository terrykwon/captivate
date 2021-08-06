from retinaface.pre_trained_models import get_model
from base.base_predictor import BasePredictor



class FaceDetector(BasePredictor):

    def __init__(self):
        self.face_detector = get_model("resnet50_2020-07-20", max_size=2048)
        self.face_detector.eval()


    def predict(self, image):
        predictions = self.face_detector.predict_jsons(image)
        bboxes = []

        # enlarge bounding boxes by 40%
        d = 0.2 # half of expand ratio
        for p in predictions:
            left, top, right, bottom = p['bbox']
            
            w = right - left
            h = bottom - top
            left -= d * w
            right += d * w
            top -= d * h
            bottom += d * h
            
            bboxes.append((left, top, right, bottom))

        return bboxes