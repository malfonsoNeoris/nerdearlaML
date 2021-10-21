import cv2
import time
import random
from numpy.lib.type_check import imag

class YoloDetector:
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    CONFIDENCE_THRESHOLD = 0.7
    NMS_THRESHOLD = .5

    @classmethod
    def createColors(cls, N):
        def randomColor():
            rgbl=[255,0,0]
            random.shuffle(rgbl)
            return tuple(rgbl)

        colors = [randomColor() for r in range(N)]
        return colors
    
    @classmethod
    def scaleBox(cls, box, from_size=None, to_size=None):
        if from_size is not None:
            W,H = from_size
        box = [box[0]/W,box[1]/H,box[2]/W,box[3]/H]

        if to_size is not None:
            W,H = to_size
            box = [box[0]*W,box[1]*H,box[2]*W,box[3]*H]
        return box

    def __init__(self, class_file, weights_file,cfg_file, input_size = (416, 416),swapRB=True,  PreferableBackend = None, PreferableTarget=None):
        class_names = []
        with open(class_file, "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]
        self.class_names = class_names
        self.cfg_file = cfg_file
        self.cfg_file = weights_file
        self.net = cv2.dnn.readNet(weights_file, cfg_file)

        self.PreferableBackend=PreferableBackend
        self.PreferableTarget=PreferableTarget

        if PreferableBackend is not None:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        if PreferableTarget is not None:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        model = cv2.dnn_DetectionModel(self.net)
        model.setInputParams(size=input_size, scale=1/255, swapRB=swapRB)
        self.model = model
        self.input_size=input_size

    def DetectOnVideoFile(self, video_file,conf=CONFIDENCE_THRESHOLD, nms_tresh = NMS_THRESHOLD, draw=False, show=False, wait_delay=10):
        vc = cv2.VideoCapture(video_file)

        detections=[]
        grabbed, frame = vc.read()    
        frame_count=-1    
        while grabbed:
            frame_count+=1

            start = time.time()
            classes, scores, boxes = self.model.detect(frame, conf, nms_tresh, draw)
            end = time.time()

            start_drawing = time.time()
            for (classid, score, box) in zip(classes, scores, boxes):
                row = {'frame':frame_count, 'class':self.class_names[classid[0]], 'score':score, 'box':box}
                detections.append(row)

                if draw:
                    color = self.COLORS[int(classid) % len(self.COLORS)]
                    label = "%s : %f" % (self.class_names[classid[0]], score)

                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            end_drawing = time.time()

            if draw:
                fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
                cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            if show:
                cv2.imshow("detections", frame)
                k=cv2.waitKey(wait_delay)
                if k==ord('q'):
                    break


            (grabbed, frame) = vc.read()
        
        vc.release()
        return detections

    def DetectOnImageFile(self, img_file:str):
        img = cv2.imread(img_file)
        return self.DetectOnImage(img)
        
    def DetectOnImage(self, img, conf=CONFIDENCE_THRESHOLD, nms_tresh = NMS_THRESHOLD): 

        if img is None:
            return

        start = time.time()
        classes, scores, boxes = self.model.detect(img, conf, nms_tresh)
        end = time.time()

        if len(classes)==0:
            print("nothing here")
            return

        classes, scores, boxes = zip(*sorted(zip(classes, scores, boxes), key = lambda k:k[2][0]))
        detections = []
        for (classid, score, box) in zip(classes, scores, boxes):
            #box = self.scaleBox(box, from_size=self.input_size)
            bbox = [int(b) for b in box]
            row = {'class':self.class_names[classid[0]], 'classId':int(classid), 'score':round(float(score),4), 'box':bbox}
            detections.append(row)
            
        return detections        

    def DrawDetections(self, img, detections, font_size=0.5):
        start_drawing = time.time()

        colors = self.createColors( len(self.class_names) )
        for detection in detections:
            #box = self.scaleBox(box, from_size=self.input_size)

            color = colors[ detection['classId'] ]
            label = "%s : %.1f" % (detection['class'], detection['score'])
            box = detection['box']
            #box = self.scaleBox(box, to_size=imag.shape[:2])
            cv2.rectangle(img, box, color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)        
        
        end_drawing = time.time()
        return end_drawing-start_drawing



if __name__ =="__main__":

    weights_file = "/home/user/pmptests/models/plate_recognition/y4t-letters_final.weights"
    names_file = "/home/user/pmptests/models/plate_recognition/obj.names"
    cfg_file = "/home/user/pmptests/models/plate_recognition/y4t-letters.cfg"
    detector = YoloDetector(names_file,weights_file,cfg_file)

    img = cv2.imread(r"warped_extreme.jpg")
    detections = detector.DetectOnImage(img)
    detector.DrawDetections(img, detections)