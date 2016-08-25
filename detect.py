# -*- coding: UTF-8 -*- 
import numpy as np
import os
import cv2
import cv2.cv as cv





def detect(img, cascade):
    rects = cv.HaarDetectObjects(img, cascade, cv.CreateMemStorage(), 1.1, 2,cv.CV_HAAR_DO_CANNY_PRUNING, (255,255)) 
    if len(rects) == 0:
        return []
    result = []
    for r in rects:
        result.append((r[0][0], r[0][1], r[0][0]+r[0][2], r[0][1]+r[0][3]))
    if result[0][2]> 300 and result[0][3] > 300:
        return result
    else:
        return []


def draw_rects(img, rects, color):
    if rects:
        for i in rects:
            cv.Rectangle(img, (int(rects[0][0]), int(rects[0][1])),(int(rects[0][2]),int(rects[0][3])),cv.CV_RGB(0, 255, 0), 1, 8, 0)



cascade = cv.Load("/home/flyvideo/caffe-master/examples/Face_rect/haarcascade_frontalface_alt.xml")



if __name__ == '__main__':
    img=cv.LoadImage('/home/flyvideo/caffe-master/examples/Face_rect/4.jpg')
    gray=cv.CreateImage(cv.GetSize(img), 8, 1)
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    cv.EqualizeHist(gray,gray)
    rects = detect(img, cascade)
    rect=(rects[0][0],rects[0][1],rects[0][2]-rects[0][0],rects[0][3]-rects[0][1])
    #draw_rects(img, rects, (0, 255, 0))
    cv.SetImageROI(img,rect)
    cv.SaveImage('face.jpg',img)   









