import glob
import random
import os
import cv2
import time
import subprocess
import sys
import numpy as np
import copy
import string
import imutils
# import some common detectron2 utilities

from detectors.detectron2detector import Detectron2Detector

#function to order points to proper rectangle
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


#function to transform image to four points
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    # # multiply the rectangle by the original ratio
    # rect *= ratio

    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def ContourTo4Points(cnt):

    cnt_ordered = sorted(cnt, key = lambda k: np.sqrt(k[0]**2 + k[1]**2) )

    top_left = cnt_ordered[0]
    bot_right = cnt_ordered[-1]

class PlateMaskDetector(Detectron2Detector):
    def __init__(self,path=rf"models/patentes_mask.pth", thres =0.8, device = 'cpu'):
        classes = ["plate",]
        type = 'COCO-InstanceSegmentation'
        network = 'mask_rcnn_R_50_FPN_3x'
        super().__init__(path,classes , thres =thres, device = device, type = type, network = network)
    
    @classmethod
    def getWarped_old(cls, im, box):
        ctr = box.cnt
                
        peri= cv2.arcLength(ctr, True)
        approx = cv2.approxPolyDP(ctr,.02*peri,True)

        if len(approx)>=4 and False:
            pts = approx.reshape(4, 2)
        elif False:
            pts = cv2.boxPoints(cv2.minAreaRect(ctr))
        elif True:
            pts = order_points(ctr)
        else:
            # determine the most extreme points along the contour
            extLeft = tuple(ctr[ctr[:,  0].argmin()])
            extRight = tuple(ctr[ctr[:,  0].argmax()])
            extTop = tuple(ctr[ctr[:,  1].argmin()])
            extBot = tuple(ctr[ctr[:,  1].argmax()])
            pts = np.array([extLeft,extRight,extTop,extBot])

        (X, Y, W, H) = cv2.boundingRect(ctr)

        to_draw = copy.copy(im)

        return four_point_transform(to_draw, pts), pts

    @classmethod
    def getWarped(cls, im, box, opt=2, pad=10):
        ctr = box.cnt
                
        peri= cv2.arcLength(ctr, True)
        approx = cv2.approxPolyDP(ctr,.02*peri,True)

        if len(approx)>=4 and opt==1:
            pts = approx.reshape(4, 2)
        elif opt==2:
            pts = order_points(ctr)
        elif opt==3:
            pts = cv2.boxPoints(cv2.minAreaRect(ctr))
        else:
            # determine the most extreme points along the contour
            extLeft = tuple(ctr[ctr[:,  0].argmin()])
            extRight = tuple(ctr[ctr[:,  0].argmax()])
            extTop = tuple(ctr[ctr[:,  1].argmin()])
            extBot = tuple(ctr[ctr[:,  1].argmax()])
            pts = np.array([extLeft,extRight,extTop,extBot])

        (X, Y, W, H) = cv2.boundingRect(ctr)

        to_draw = copy.copy(im)

        return four_point_transform(to_draw, pts, pad=pad), pts

class LettersDetector(Detectron2Detector):
    def __init__(self,path=rf"models/plates.pth", thres =0.8, device = 'cpu'):
        classes = list(string.digits+string.ascii_letters.upper())
        type = 'COCO-Detection'
        network = 'faster_rcnn_R_50_FPN_3x'
        network = 'faster_rcnn_X_101_32x8d_FPN_3x'
        super().__init__(path,classes , thres =0.8, device = device, type = type, network = network)


if __name__ == "__main__":
    file = 'models/plate_mask/patentes_mask.pth'

    detector = PlateMaskDetector(path = file, device = 'cuda')
    #lettersD = LettersDetector(device = 'cpu')

    im = cv2.imread("tests\imgs\warp_img_test.png")
    outputs = detector.Predict(im, .4)                   


    boxes = detector.getBoxes(outputs)
    box = boxes[0]
    if box.cnt is not None:
        ctr = box.cnt
        
        peri= cv2.arcLength(ctr, True)
        approx = cv2.approxPolyDP(ctr,.02*peri,True)

        if len(approx)>=4 and False:
            pts = approx.reshape(4, 2)
        elif False:
            pts = cv2.boxPoints(cv2.minAreaRect(ctr))
        elif False:
            pts = order_points(ctr)
        else:
            # determine the most extreme points along the contour
            extLeft = tuple(ctr[ctr[:,  0].argmin()])
            extRight = tuple(ctr[ctr[:,  0].argmax()])
            extTop = tuple(ctr[ctr[:,  1].argmin()])
            extBot = tuple(ctr[ctr[:,  1].argmax()])
            pts = np.array([extLeft,extRight,extTop,extBot])

        (X, Y, W, H) = cv2.boundingRect(ctr)

        to_draw = copy.copy(im)

        warped = four_point_transform(to_draw, pts)
        cv2.circle(im, tuple(pts[0]), 3, (0, 0, 255), -1)
        cv2.circle(im, tuple(pts[1]), 3, (0, 255, 0), -1)
        cv2.circle(im, tuple(pts[2]), 3, (255, 0, 0), -1)
        cv2.circle(im, tuple(pts[3]), 3, (255, 255, 0), -1)
        cv2.drawContours(im, [ctr], 0, (0,255,0), 2)

    img = detector.Draw(im, outputs)


    if box.cnt is not None:
        warped = cv2.resize( warped, (320, 320) )
        # outputs = lettersD.Predict(warped)
        # boxes = detector.getBoxes(outputs)

        cv2.imwrite("warped.jpg", warped)


    cv2.imwrite("salida.jpg", img)
    print(boxes)