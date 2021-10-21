# from fastai.vision.widgets import *
# from fastai.vision.all import *

from fastai.vision.all import load_learner
from pathlib import Path
import pathlib
import time
from datetime import datetime


from contextlib import contextmanager
from tqdm.auto import tqdm
# from fastcore.basics import true
# from torch._C import T
@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

class FastAiClassifier:
    def __init__(self, filename, use_gpu=False):
        t=time.perf_counter()
        self.learner = load_learner(filename, cpu=not use_gpu)
        # self.learner = load_learner(Path()/filename, cpu=not use_gpu)
        dt = time.perf_counter()-t
        print('classifier load time: ',dt)
        #self.classes = self.learner.data.classes

    
    def predict(self, image, roi=None):
        if roi:
            x1,y1,w1,h1 = roi
            h,w = image.shape[:2]
            x1 = min(max(0,x1),w)
            x2 = max(min(w,x1+w1),0)
            y1 = min(max(0,y1),h)
            y2 = max(min(h,y1+h1),0)

            img_roi = image[y1:y2,x1:x2]
        else:
            img_roi=image
        pred, pred_idx, probs = self.learner.predict(img_roi)
        probs = {n:round(float(prob),4) for n,prob in enumerate(list(probs))}

        return {'class_id':int(pred_idx), 'class':pred,'score':float(probs[int(pred_idx)]), 'scores':probs}


if __name__ == '__main__':
    import cv2
    import time
    filename = r'./models/truck_classifier/trucks.pkl'#D:\Source\cemex\src\cxus_src\one_api\

    def simple(filename):
        try:
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            filename = Path(filename)
            props = FastAiClassifier(filename, use_gpu=True)
            image = cv2.imread(r'tests\imgs\class_img_test.png')
            data = props.predict(image)
            print(data)
        except Exception as ex:
            print(ex)
        finally:
            pathlib.PosixPath = temp
            pass

    def multiple(filename,use_gpu=False):
        try:
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            filename = Path(filename)
            props = FastAiClassifier(filename, use_gpu=use_gpu)
            image = cv2.imread(r'tests\imgs\class_img_test.png')
            t=time.perf_counter()
            for _ in range(100):                
                data = props.predict(image)
            dt = time.perf_counter() -t
            print('Time taken: ',dt)
            print(data)
        except Exception as ex:
            print(ex)
        finally:
            pathlib.PosixPath = temp
            pass

    print(datetime.now())
    multiple(filename,use_gpu=True)
    print(datetime.now())