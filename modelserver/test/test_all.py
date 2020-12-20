import unittest
import numpy as np
import torch
from modelserver.base.face_detector2 import FaceDetector
from modelserver.base.object_detector import ObjectDetector
from modelserver.base.gaze_follower import GazeFollower


class Test(unittest.TestCase):


    def test_sanity(self):
        self.assertEqual(1, 1)


    def test_face_detector(self):
        d = FaceDetector()
        frame = torch.zeros(224, 224, 3)
        p = d.predict(frame)
        self.assertIsNotNone(p)


    def test_object_detector(self):
        d = ObjectDetector() 
        frame = np.zeros((720, 1080, 3), dtype='uint8')
        objects = d.predict(frame)
        instances = objects['instances']
        self.assertIsNotNone(instances)
        fields = instances.get_fields()
        object_bboxes = fields['pred_boxes'].tensor.cpu().numpy()
        self.assertIsNotNone(object_bboxes)


    def test_google_speech_api(self):
        """Is there a way to just ping the endpoint instead of sending an actual
        request?
        """
        pass


    def test_gaze_follower(self):
        d = GazeFollower()
        self.assertIsNotNone(d)


if __name__ == '__main__':
    unittest.main()
