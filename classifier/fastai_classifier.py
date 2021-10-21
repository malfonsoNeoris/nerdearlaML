# from fastai.vision.widgets import *
# from fastai.vision.all import *

from fastai.vision.all import load_learner
from pathlib import Path


class FastAiClassifier:
    def __init__(self, filename):
        self.learner = load_learner(Path()/filename)
        #self.classes = self.learner.data.classes

    
    def predict(self, image):
        pred, pred_idx, probs = self.learner.predict(image)
        probs = {n:round(float(prob),4) for n,prob in enumerate(list(probs))}

        return {'class_id':int(pred_idx), 'class':pred,'score':float(probs[int(pred_idx)]), 'scores':probs}


if __name__ == '__main__':
    import cv2
    import time
    filename = 'models/truck_classifier/trucks.pkl'
    props = FastAiClassifier(filename)
    image = cv2.imread('72de4e2d-b8ec-4bfd-9be1-90f0412089ca_97.jpg')
    start = time.time()
    for i in range(100):
    
        data = props.predict(image)
    end = time.time()
    print(data)
    print(f"time: {(end-start) / 100}")
