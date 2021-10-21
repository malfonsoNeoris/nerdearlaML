import cv2
import time
import random
import imutils
import json
import os
import numpy as np
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

    def __init__(self, class_file, weights_file,cfg_file, input_size = (416, 416),swapRB=True,  use_gpu=True):
        class_names = []
        with open(class_file, "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]
        self.class_names = class_names
        self.colors = self.createColors(len(class_names))
        self.cfg_file = cfg_file
        self.cfg_file = weights_file
        self.net = cv2.dnn.readNet(weights_file, cfg_file)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            pass

        model = cv2.dnn_DetectionModel(self.net)
        model.setInputParams(size=input_size, scale=1/255, swapRB=swapRB)
        self.model = model
        self.input_size=input_size

    def DetectOnVideoFile(self, video_file,conf=CONFIDENCE_THRESHOLD,
                            nms_tresh = NMS_THRESHOLD, draw=False,
                            show=False, wait_delay=10,
                            frame_skip=1, 
                            font_size=1,
                            write_output=False,

                            ):
        vc = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
        if "rtsp" not in video_file:
            basename  =os.path.basename(video_file)
        else: basename =video_file


        writer=None
        grabbed, frame = vc.read()    
        if write_output:
            name = video_file
            name = name.replace(".mp4", "_procesed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h,w = frame.shape[:2]
            writer =  cv2.VideoWriter(name, fourcc, 5, (w,h))
            

        frame_count=-1    
        while grabbed:
            (grabbed, frame) = vc.read()
            if not grabbed:
                break

            detections=[]

            frame_count+=1
            if frame_count%frame_skip!=0:
                continue

            start = time.time()
            
            ##-----------------------------------
            #H,W = frame.shape[:2]
            #small_frame = cv2.resize(frame, (W//4, H//4))
            #classes, scores, boxes = self.model.detect(small_frame, conf, nms_tresh)
            #boxes = [self.scaleBox(box, from_size=(W//4, H//4), to_size=(W,H)) for box in boxes]

            ##-----------------------------------
            classes, scores, boxes = self.model.detect(frame, conf, nms_tresh)
            ##--------------

            for (classid, score, box) in zip(classes, scores, boxes):
                bbox = [int(b) for b in box]
                row = {'class':self.class_names[classid[0]], 'classId':int(classid), 'score':round(float(score),4), 'box':bbox}            
                row['frame'] =frame_count
                detections.append(row)
            end = time.time()
            
            
            if draw:
                dt = self.DrawDetections(frame, detections, font_size=font_size)
                fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (frame_skip / (end - start), (dt) * 1000)
                cv2.putText(frame, fps_label, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                name_label = f'File: {basename} \t\t\tShape({frame.shape[0]},{frame.shape[1]})'
                cv2.putText(frame, name_label, (20, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2*font_size)
            if write_output and writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("detections", frame)
                k=cv2.waitKey(wait_delay)
                if k==ord('q'):
                    break


            
        if writer:
            writer.release()
        vc.release()
        return

    def DetectOnImageFile(self, img_file:str, conf=CONFIDENCE_THRESHOLD, nms_tresh = NMS_THRESHOLD):
        img = cv2.imread(img_file)
        return self.DetectOnImage(img, conf=conf, nms_tresh = nms_tresh)


    @classmethod
    def getroi(img, roi):
        if roi is None:
            return img
        x1,y1,w1,h1 = roi
        h,w = img.shape[:2]
        x1 = min(max(0,x1),w)
        x2 = max(min(w,x1+w1),0)
        y1 = min(max(0,y1),h)
        y2 = max(min(h,y1+h1),0)

        return img[y1:y2,x1:x2]

    def DetectOnImageBase(self, img, conf=CONFIDENCE_THRESHOLD, nms_tresh = NMS_THRESHOLD, roi=None): 

        if img is None:
            return None,None,None

        if roi:
            x1,y1,w1,h1 = roi
            h,w = img.shape[:2]
            x1 = min(max(0,x1),w)
            x2 = max(min(w,x1+w1),0)
            y1 = min(max(0,y1),h)
            y2 = max(min(h,y1+h1),0)

            img_roi = img[y1:y2,x1:x2]
        else:
            img_roi=img
        start = time.time()
        classes, scores, boxes = self.model.detect(img_roi, conf, nms_tresh)
        end = time.time()
        if roi:
            for box in boxes:
                box[0]+=x1
                box[1]+=y1

        if len(classes)==0:
            return None,None,None

        classes, scores, boxes = zip(*sorted(zip(classes, scores, boxes), key = lambda k:k[2][0]))
        
        return classes, scores, boxes

    def DetectOnImage(self, img, conf=CONFIDENCE_THRESHOLD, nms_tresh = NMS_THRESHOLD,roi=None): 

        if img is None:
            return []

        classes, scores, boxes = self.DetectOnImageBase(img, conf=conf, nms_tresh = nms_tresh, roi=roi)
        if classes is None:
            return []
        detections = []
        for (classid, score, box) in zip(classes, scores, boxes):
            #box = self.scaleBox(box, from_size=self.input_size)
            bbox = [int(b) for b in box]
            row = {'class':self.class_names[classid[0]], 'classId':int(classid), 'score':round(float(score),4), 'box':bbox}
            detections.append(row)
            
        return detections        

    @classmethod
    def DrawDetections(cls, img, detections, colors=None,font_size=0.5, thickness=2):
        start_drawing = time.time()

        for detection in detections:
            #box = self.scaleBox(box, from_size=self.input_size)
            if colors is None:
                colors = cls.COLORS
            color = colors[ detection['classId'] ]
            label = "%s : %.1f" % (detection['class'], detection['score'])
            box = detection['box']
            #box = self.scaleBox(box, to_size=imag.shape[:2])
            cv2.rectangle(img, box, color, thickness)
            ts,baseline = cv2.getTextSize("pepe", cv2.FONT_HERSHEY_SIMPLEX, font_size,thickness)
            space  = 5
            where = (box[0]+space, box[1] + space+ts[1])
            if box[1]>ts[1]+5:
                where = (box[0]+space, box[1] - space)
            elif box[1]+box[3] + space+ ts[1] < img.shape[0]:
                where = (box[0]+space, box[1]+box[3] + space+ ts[1])

            cv2.putText(img, label, where, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        
        end_drawing = time.time()
        return end_drawing-start_drawing

    @classmethod
    def tupleNumpy2List(cls, tpl):
        if not isinstance(tpl, list):
            return np.asarray(tpl).flatten().tolist()
        return tpl

    @classmethod
    def _encode_info(cls,classes, scores, boxes):
        clss = cls.tupleNumpy2List(classes)
        confs = cls.tupleNumpy2List(scores)
        boxes = np.asarray(boxes).tolist()

        json_str_boxes = json.dumps(boxes)
        json_str_confs = json.dumps(confs)
        json_str_clss = json.dumps(clss)

        # Return json
        return {
                "pred_boxes":json_str_boxes,
                "confs":json_str_confs,
                "classes":json_str_clss
                }

if __name__ =="__main__":
    from tqdm.auto import tqdm
    vehicle = {'cfg':r'models\vehicle_detection\yolov4-tiny-416.cfg',
                'weights':r'models\vehicle_detection\yolov4-tiny-416.weights',
                'names':r'models\vehicle_detection\coco.names'}
    props = {'cfg':r'models\plate_and_others\yolo4-second.cfg',
                'weights':r'models\plate_and_others\yolo4-second.weights',
                'names':r'models\plate_and_others\obj.names'}

    plate = {'cfg':r'models\plate_recognition\y4t-letters_final.cfg',
                'weights':r'models\plate_recognition\y4t-letters_final.weights',
                'names':r'models\plate_recognition\obj.names'}
    models = {'vehicle':vehicle, 'props':props, 'plate':plate}

    # detector = YoloDetector(vehicle['names'],vehicle['weights'],vehicle['cfg'])
    detector = YoloDetector("api/models/detector/coco.names", 
                        "api/models/detector/y4t-custom.weights",
                        "api/models/detector/y4t-custom.cfg")

    def videotest():
        video_folder = r'D:\neoris.com\CX US Image Recognition - ComputerVision\data\datasets\videos'

        import os
        import random

        video_files = [file for file in os.listdir(video_folder) if file.endswith('mp4')]

        random_video = os.path.join(video_folder, random.choice(video_files))
        
        cv2.startWindowThread()
        cv2.namedWindow("detections", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detections', 800,600)
        
        detections = detector.DetectOnVideoFile(random_video, draw=True, show=True, wait_delay=100)
    def imgtest(filename):
        img = cv2.imread(filename)
        detections = detector.DetectOnImage(img, conf=.3)
        detector.DrawDetections(img, detections)
        return img, detections

    def fpsTest(filename):
        img = cv2.imread(filename)
        t=time.perf_counter()
        n=300
        for i in tqdm(range(n)):
            detections = detector.DetectOnImage(img, conf=.3)
        dt=time.perf_counter()-t
        print('Total: ',dt, ' - FPS = ', n/dt)
        detector.DrawDetections(img, detections)
        return img, detections


    img, detections = fpsTest(r'prueba.jpg')
    cv2.imshow('test',img)
    cv2.waitKey(0)