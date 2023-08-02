'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2 as cv
import numpy as np
import time

def nothing(x):
        pass

def drawLine(rect, img):
    box = cv.boxPoints(rect)
    box = np.int0(box) 
    cv.line(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0,0,255), 2) 
    width = int(box[1][0]- box[0][0])
    return width
  
def getSubImage(rect, src, img):
	
	box = cv.boxPoints(rect)
	box = np.int0(box)

	width = int(rect[1][0])
	height = int(rect[1][1])
	theta = rect[2]
	cv.polylines(img, [box], True, (255,0,0), 1)

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

cv.namedWindow("namespace")
cv.createTrackbar("a", "namespace", 0, 255, nothing)

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Set the upper limit of the camera's frame rate to 30 fps
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(20.0)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # cv.namedWindow('title', cv.WINDOW_NORMAL)
        ORIIMG = img.copy()

        h, w = img.shape[:2]

        # print(img.shape)
        img = cv.resize(img, (int(w/2), int(h/2)), fx=0.5, fy=0.5)
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # threshold = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)[1]
        # contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        h, w = img.shape[:2]

        imgcpy = img.copy()
        imgshow = imgcpy[150:450, 100:700]
        imgshowcpy = imgshow.copy()
        a = cv.getTrackbarPos("a", "namespace")
        gray = cv.cvtColor(imgshow, cv.COLOR_BGR2GRAY)
        threshold = cv.threshold(gray, a, 255, cv.THRESH_BINARY)[1]

        CX, CY = 0, 0

        try:
            contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

            contour = sorted(contour, key= cv.contourArea, reverse= True)
            contour = contour[0]
            rect = cv.minAreaRect(contour)

            # value about weight as length
            # print(getSubImage(rect, threshold, imgshowcpy)[1])

            subImage = getSubImage(rect, threshold, imgshowcpy)[0]

            A = 32
            W = 1920 
            length = subImage.shape[1]* A / W

            print("width = " , subImage.shape[1] )
            print("length = " , 2*length)
            # if len(contour) >0:
            C = cv.moments(contour)

            if C["m00"] != 0:
                CX = int(C["m10"] / C["m00"])
                CY = int(C["m01"] / C["m00"])
            else:
                CX, CY = 0, 0

            # x, y, we, he = cv.boundingRect(contour)
            # cv.rectangle(img, (x, y), (x+we, y+he), (0,0,255), 1)
            # cv.putText(img, 'Length = {} pixel'.format(x+w), (CX, CY + 50), cv.FONT_HERSHEY_COMPLEX, 1, 255, 1)
        
        except:
            pass
        
        # print(img.shape)

        

        cv.circle(imgshow, (CX, CY), 2, (0,0,255), 3)
        cv.line(imgshow, (int(w/2)+30, 0), (int(w/2)+30, h), (255,0,0), 2)
        cv.line(imgshow, (int(w/2)-30, 0), (int(w/2)-30, h), (0,255,0), 2)

        cv.putText(imgshow, 'cx={} cy={}'.format(CX, CY), (CX, CY), cv.FONT_HERSHEY_COMPLEX, 1, 255, 1)

        
        # if CX < int(img.shape[1]/2) and CX > 450:
        #     cv.imwrite('D:\Project\Take a picture\image{}_{}ori.jpg'.format(CX, CY), threshold)
        #     cv.imwrite('D:\Project\Take a picture\image{}_{}thre.jpg'.format(CX, CY), imgcpy)
        #     print("image saved")
        #     time.sleep(0.2)


        # cv.drawContours(img, [contour], 0, (0, 255, 0), -1)
        mask = cv.bitwise_and(imgshowcpy, imgshowcpy, mask= threshold)
        cvtThreshold = np.stack((threshold, )*3, axis= -1)
        mergeImg = np.hstack((imgshow, mask, cvtThreshold))
        mergeImg = cv.resize(mergeImg, (0,0), fx=0.5, fy= 0.5)

        cv.imshow('namespace', mergeImg)
        cv.imshow('show', subImage)
        cv.imshow('oushow', imgshowcpy)

        # cv.imshow('title', ORIIMG)

        k = cv.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('s'):
            print("Image saved !")
            cv.imwrite('D:\Project\images\image.jpg', img)
                
    grabResult.Release()
        
# Releasing the resource    
camera.StopGrabbing()

cv.destroyAllWindows()