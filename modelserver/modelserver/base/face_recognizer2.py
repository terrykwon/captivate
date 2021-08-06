import insightface
from insightface.utils import face_align
import numpy as np



class FaceRecognizer(): 
    ''' does detection + recognition.
        returns bounding box + identity?
    '''

    def __init__(self, target_face_image):
        detector_name = 'retinaface_r50_v1'
        recognizer_name = 'arcface_r100_v1'
        self.detector = insightface.model_zoo.get_model(detector_name)
        self.recognizer = insightface.model_zoo.get_model(recognizer_name)
        self.detector.prepare(ctx_id=1, nms=0.4)
        self.recognizer.prepare(ctx_id=1)

        self.target_embedding = self._get_target_embedding(target_face_image)

    
    def _get_target_embedding(self, target_face_image):
        faces = self.process_faces(target_face_image)

        # ideally there should only be one face detected,
        # but if not--just take the first face
        # TODO: change because this is very bad
        face = faces[0]

        return face['embedding']


    def predict(self, image):
        threshold = 0.3
        faces = self.process_faces(image)

        for face in faces:
            candidate_embedding = face['embedding']
            if np.dot(candidate_embedding, self.target_embedding) > threshold:
                return face

        return [] # target face not found


    def process_faces(self, image):
        ''' Finds all faces in an images and returns their bounding boxes
            as well as their embeddings (length 512 vector).
        '''
        threshold = 0.3
        scale = 1.0
        bboxes, landmarks = self.detector.detect(image, threshold=threshold, scale=scale)

        if bboxes.shape[0]==0: # no faces found
            return []

        if bboxes.shape[0]:
            area = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
            image_center = image.shape[0]//2, image.shape[1]//2
            offsets = np.vstack([ (bboxes[:,0]+bboxes[:,2])/2-image_center[1], 
                    (bboxes[:,1]+bboxes[:,3])/2-image_center[0] ])
            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
            values = area-offset_dist_squared*2.0 # some extra weight on the centering
            bindex = np.argsort(values)[::-1] # some extra weight on the centering
            # bindex = bindex[0:max_num]
            bboxes = bboxes[bindex, :]
            landmarks = landmarks[bindex, :]

        results = []
        for i in range(bboxes.shape[0]): # loop over all faces
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i,4]
            landmark = landmarks[i]
            face_crop = face_align.norm_crop(image, landmark = landmark)
            embedding = None
            embedding_norm = None
            normed_embedding = None

            embedding = self.recognizer.get_embedding(face_crop).flatten()
            embedding_norm = np.linalg.norm(embedding)
            normed_embedding = embedding / embedding_norm

            # face = Face(bbox = bbox, landmark = landmark, det_score = det_score, embedding = embedding, gender = gender, age = age
            #         , normed_embedding=normed_embedding, embedding_norm = embedding_norm)

            face = {
                'bbox': bbox,
                'embedding': normed_embedding
            }

            results.append(face)

        return results
