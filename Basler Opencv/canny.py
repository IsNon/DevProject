from pypylon import pylon
import cv2 as cv
import numpy as np
import glob
import pandas as pd
import time
import os
from datetime import datetime
# 82, 105, 51
# 91, 116, 57
# 81, 108, 50
# 88, 113, 59
# 96, 123, 63

# 65, 103, 87
def getSubImage(rect, src, img, drawPolyline = False):
        
    box = cv.boxPoints(rect)
    box = np.intp(box)
        
    width = int(rect[1][0])
    height = int(rect[1][1])

    
    theta = rect[2]

    if drawPolyline == True:
        cv.polylines(img, [box], True, (0,0,255), 3)

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(src, M, (width, height))

    if width < height :
        warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)   
                
    return warped, width  


def findContour(threshold):
    gray = cv.cvtColor(threshold, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    if len(contour)>0:
        contour = max(contour, key= cv.contourArea) 
        rect = cv.minAreaRect(contour)

    return rect, threshold

def nothing(x):
        pass

cv.namedWindow("namespace")
cv.createTrackbar("lh", "namespace", 0, 255, nothing)
cv.createTrackbar("ls", "namespace", 0, 255, nothing)
cv.createTrackbar("lv", "namespace", 0, 255, nothing)
cv.createTrackbar("hh", "namespace", 255, 255, nothing)
cv.createTrackbar("hs", "namespace", 255, 255, nothing)
cv.createTrackbar("hv", "namespace", 255, 255, nothing)

def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,', ' ,y)
        font = cv.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', '+ str(y)
        cv.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        cv.imshow('image', img)
    if event == cv.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ', '+ str(green)+ ', '+ str(red)
        cv.putText(img, strBGR, (x, y), font, .5, (0, 255, 255), 2)
        cv.imshow('image', img)


while True:
    img = cv.imread("D:\\Project\\b\\calibresult2.jpg")
    imgv = cv.imread("D:\\Project\\b\\image.jpg")

    img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # a = cv.getTrackbarPos("a", "namespace")

    lh = cv.getTrackbarPos("lh", "namespace")
    ls = cv.getTrackbarPos("ls", "namespace")
    lv = cv.getTrackbarPos("lv", "namespace")  
    hh = cv.getTrackbarPos("hh", "namespace")
    hs = cv.getTrackbarPos("hs", "namespace")
    hv = cv.getTrackbarPos("hv", "namespace")

    # low_red = np.array([lh, ls, lv])
    # high_red = np.array([hh, hs, hv])
    # mask = cv.inRange(hsv, low_red, high_red)
    # result = cv.bitwise_and(img, img, mask= mask)

    low_red = np.array([65, 116, 72])
    high_red = np.array([255, 255, 255])

    mask = cv.inRange(hsv, low_red, high_red)
    result = cv.bitwise_and(img, img, mask= mask)
    threshold = ~mask

    kernel = np.ones((7,7),np.uint8)
    threshold1 = cv.medianBlur(threshold, 5)
    threshold2 = cv.morphologyEx(threshold1, cv.MORPH_CLOSE, kernel)
    contour = cv.findContours(threshold1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    
    contour = max(contour, key= cv.contourArea)

    newImg= np.zeros(img.shape, dtype= np.uint8)
    cv.drawContours(newImg, [contour], 0, (255, 255, 255), -1)

    newImg = cv.split(newImg)[0]
    img = cv.bitwise_and(img, img, mask= newImg)
    rect, img1 = findContour(imgv)
    img1 = getSubImage(rect, img1, img1.copy())[0]
   # # img = cv.bitwise_and(img, img, mask= threshold)
    # cv.setMouseCallback('namespace', click_event)
    # img = cv.resize(result, (0,0), fx=0.25, fy=0.25)

    # cv.imshow("mask", mask)
    cv.imshow("img", img)
    cv.imshow("img1", img1)
    # cv.imshow("img2", img2)
    cv.imshow("result", result)
    cv.imshow("toggle mask", newImg)
    # cv.imshow("filter", threshold)
    # cv.imshow("filter1", threshold1)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
