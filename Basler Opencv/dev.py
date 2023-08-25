'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2 as cv
import numpy as np
import glob
import pandas as pd
import time
import os
from datetime import datetime
import serial

# ser = serial.Serial('COM3', 115200)


n = 0
id = 0

lengthFish = 0
positionCurrent = 0
positionAhead = 0

currentTime = datetime.now()

pathSaveImage = 'b'
pathSaveImageCalib = 'c'





def nothing(x):
        pass

cv.namedWindow("namespace")
cv.createTrackbar("a", "namespace", 120, 255, nothing)


CHECK_DIR = os.path.isdir(pathSaveImage)

# if directory does not exist create
if not CHECK_DIR:
    os.makedirs(pathSaveImage)
    print(f'"{pathSaveImage}" Directory is created')
else:
    print(f'"{pathSaveImage}" Directory already Exists.') 


def convertLenghToInjection(length):
    pass


def isCheckDirection(position):
    if positionCurrent < positionAhead :
        transmitData(1, position)

    elif positionCurrent > positionAhead:
        transmitData(0, position)

    else:
        pass     


def transmitData(direction, position):
    value = bytes('{} {}\n'.format(direction, position), encoding= 'utf8')
    # ser.write(value)


def calibInputImage(img, cameraMatrix, Distortion_Parameters, path, id):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, Distortion_Parameters, (w,h), 1, (w,h))
    # undistort
    Distortion = cv.undistort(img, cameraMatrix, Distortion_Parameters, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    Distortion = Distortion[y:y+h, x:x+w]
    cv.imwrite('{}/calibresult{}.jpg'.format(path, id), Distortion)
    outputImage= cv.imread('{}/calibresult{}.jpg'.format(path, id))

    return outputImage


def findContour(threshold):
    gray = cv.cvtColor(threshold, cv.COLOR_BGR2GRAY)
    contour = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if len(contour)>0:
        contour = sorted(contour, key= cv.contourArea, reverse= True)
        contour = contour[0]
        rect = cv.minAreaRect(contour)

    return rect 


def processImage(img, path, id, printValue= False):
    global positionCurrent, positionAhead  
    # a = cv.getTrackbarPos("a", "namespace")
    newImg= np.zeros(img.shape, dtype= np.uint8)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray, 165, 255, cv.THRESH_BINARY)[1]
    contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    if len(contour) > 0:
        contour = sorted(contour, key= cv.contourArea, reverse= True)
        contour = contour[0]
        rects = cv.minAreaRect(contour)   

        # subImg = calculatorLength(32, 1920, rects, threshold, img, printValue)[1]
        cv.drawContours(newImg, [contour], 0, (255, 255, 255), -1)
        rect2= findContour(newImg)
        lengthFish, subImg1 = calculatorLength(32, 1920, rect2, newImg, img, writeLength= True)

        #	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#
        
        positionCurrent = lengthFish
        isCheckDirection(positionCurrent)
        positionAhead = positionCurrent

        #	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#

        # cv.imwrite('{}\subImage{}.jpg'.format(path, id), subImg)
        cv.imwrite('{}\\newImage{}.jpg'.format(path, id), newImg)
        cv.imwrite('{}\\newConvertImage{}.jpg'.format(path, id), subImg1)

    return rects


def findThreshold(img):
    contour = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    if len(contour) > 0:
        contour = sorted(contour, key= cv.contourArea, reverse= True)
        contour = contour[0]
        rects = cv.minAreaRect(contour) 

    return rects


def saveImage(drawframe, img, threshold, path, takeImage = False):
    global id

    contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    
    try:
        contour = sorted(contour, key= cv.contourArea, reverse= True)
        contour = contour[0]
        rect = cv.minAreaRect(contour)

        if len(contour) > 0:
            C = cv.moments(contour)

            if C["m00"] != 0:
                CX = int(C["m10"] / C["m00"])
                CY = int(C["m01"] / C["m00"])

                cv.circle(drawframe, (CX , CY), 2, (0,0,255), 15)
                cv.putText(drawframe, 'CX= {} CY= {}'.format(CX , CY), (CX , CY), cv.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                # print("value coordinates= ", CX , CY)

                if cv.contourArea(contour) > 10000 and takeImage == True:
                    if CX < int(img.shape[1]/2) + 30 and CX > int(img.shape[1]/2) - 30:
                        id += 1
                        cv.imwrite('{}\image_{}_original.jpg'.format(path, id), img)
                        # cv.imwrite('{}\image_{}_threshold.jpg'.format(path, id), threshold)
                        print("image saved")

                        time.sleep(0.2)

                        readImage = cv.imread('{}\image_{}_original.jpg'.format(path, id))     
                        processImage(readImage, path, id, True)
                        # outimage= calibInputImage(readImage, cameraMatrix, Distortion_Parameters, path, id)
                        # processImage(outimage, path, id, True)

    except:
        pass


def calculatorLength(lengthFrame, widthPixelFrame, rect, threshold, img, writeLength = False, drawPolyline = False):

    subImage = getSubImage(rect, threshold, img, drawPolyline)[0]

    length = subImage.shape[1]* lengthFrame / widthPixelFrame

    # points= findPoints(subImage)
    # cv.line(img, points[0][2], points[-1][2], (0,0,0), 2)

#     print("width = " , subImage.shape[1] )

    if writeLength == True:
        print("length = " , length)

    return length, subImage


def calibPylon(cameraMatrix, dist, img):
    # Undistortion  
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)

    return dst


def determineLocateLine(img):
    cv.line(img, (int(img.shape[1]/2)+30, 0), (int(img.shape[1]/2)+30, img.shape[0]), (255,0,0), 5)
    cv.line(img, (int(img.shape[1]/2)-30, 0), (int(img.shape[1]/2)-30, img.shape[0]), (0,255,0), 5)


def drawLine(rect, img):
    box = cv.boxPoints(rect)
    box = np.intp(box) 
    cv.line(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0,0,255), 2) 
    width = int(box[1][0]- box[0][0])
    return width


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


def calibCamera(chessboardSize, frameSize, pathImage, saveImage = False):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0: chessboardSize[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('{}/*.jpg'.format(str(pathImage)))

    a_number_of_images = 1

    for fname in images:

        img = cv.imread(fname)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

            h, w = img.shape[:2]
            print(img.shape)
            img = cv.resize(img, (int(w/2), int(h/2)), fx= 0.5, fy= 0.5)
            
            # cv.imshow('img', img)

            # cv.waitKey(1000)

        a_number_of_images += 1

    print('a_number_of_images:', a_number_of_images)

    cv.destroyAllWindows()

    # calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    # print("Camera Calibrate:", ret)
    print("\ncameraMatrix", cameraMatrix)
    print("\nDistortion Parameters:", dist)
    # print("\nRotation Vector:", rvecs)
    # print("\nTranstation Vector:", tvecs)

    # pickle.dump((cameraMatrix, dist), open('calibration.pkl', "wb"))
    # pickle.dump(cameraMatrix, open('cameraMatrix.pkl', "wb"))
    # pickle.dump(dist, open('calibration.pkl', "wb"))

    k = 1
    #               the same feature            #
    # THE FIRST METHOD
    if saveImage == True:
        # Undistortion  
        img = cv.imread('{}\Image1.png'.format(pathImage))
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('{}\calibresult{}.png'.format(pathImage, k), dst)
        k += 1

    # check error
    mean_error = 0

    for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) ) 
        
    return cameraMatrix, dist


def findPoints(img, indxStart = 0, indxEnd = 1):
	# find smalles value in each column
	origi = int(img.shape[1])
	midPt = int(img.shape[1] / 2)
	firstPt = int(img.shape[1] /7)
	limitPt = int(img.shape[1] - img.shape[1] /10)
	coordinate = []
	disPoint = []
	points = []

	indxStart= int(origi*indxStart)
	indxEnd = int(origi*indxEnd)
	

	# for i in range(img.shape[1]):
	for i in range(indxStart, indxEnd):
	# for i in range(midPt, limitPt):
		tmp_col = np.where((img[:, i] == 255))

		if len(tmp_col[0]) > 2:
			firstPoint =  (i, tmp_col[0][0])
			lastPoint =  (i, tmp_col[0][-1])
			distance = tmp_col[0][-1] - tmp_col[0][0]
			midPoint = (i, int((tmp_col[0][0] + tmp_col[0][-1])/2))
			points.append((distance, (firstPoint, lastPoint), midPoint))

	return points


def drawMid(img, array, numline = 40):
	# # draw line along image
	reser = []
	t = len(array)
	value = t / numline
	value = int(value - value / 10)

	for i in range(t) :
		t = t - value

		cv.line(img, array[t][1][0], array[t][1][1], (255,t,0), 2)
		cv.circle(img, array[t][2], 5, (255,255,t), -1)
		# cv2.line(roi, test[t-50][2], test[t][2], (t,200,0), 2)
			
		# reserve.append(test[t][2])
		if t < value :
			reser.insert(0, array[-1][2])
			reser.append(array[0][2])
			break

		reser.append(array[t][2])
	
	return reser


def isDetermineBackAndBelly(subImageToDraw, subImageThreshold):
     #              determine back and belly                         #
    subImageCopy = subImageToDraw.copy()
    splitImage = subImageToDraw.copy()
    points = findPoints(subImageThreshold)
    aNumberOfLine = drawMid(subImageToDraw, points, numline= 20)
    

    #	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#
                                    # SPLIT IMAGE INTO 2 PARTS

    cv.polylines(subImageCopy, [np.array(aNumberOfLine)], isClosed= False, color= ( 0, 0, 0), thickness= 1)                                
    backImage = splitImage.copy()
    bellyImage = splitImage.copy()
    newMidBack = aNumberOfLine.copy()
    newMidBelly = aNumberOfLine.copy()

    # # bottom side
    newMidBelly.append((0,0))
    newMidBelly.append((subImageCopy.shape[1], 0))

    # # top side
    newMidBack.append((0, subImageCopy.shape[0]))
    newMidBack.append((subImageCopy.shape[1], subImageCopy.shape[0]))
    
        # # back side and belly
    cv.fillPoly(backImage, pts= [np.array(newMidBack)], color= (0,0,0))  	# BACK
    cv.fillPoly(bellyImage, pts= [np.array(newMidBelly)], color= (0,0,0))	# BELLY


    backGray = cv.cvtColor(backImage, cv.COLOR_BGR2GRAY)
    # backGray = cv2.GaussianBlur(backGray, (3,3), 0)

    bellyGray = cv.cvtColor(bellyImage, cv.COLOR_BGR2GRAY)
    # bellyGray = cv2.GaussianBlur(bellyGray, (3,3), 0)

    meanBackGray = round(np.mean(backGray), 2)
    meanBellyGray = round(np.mean(bellyGray), 2)

    # meanBackGray = np.mean(cv2.countNonZero(backGray))
    # meanBellyGray = np.mean(cv2.countNonZero(bellyGray))

    if meanBellyGray > meanBackGray :
        # print('continues proccessing then the next step is segment fin')
        pass

    return backImage, bellyImage, subImageCopy


def parameterCalib():
    # cameraMatrix, Distortion_Parameters = calibCamera((15,9), (1920, 1200), 'a', False)

    cameraMatrix = np.array([[1.50392060e+03, 0.00000000e+00, 1.00861013e+03],
                    [0.00000000e+00, 1.50877936e+03, 5.83831826e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    Distortion_Parameters = np.array([[-0.15297373,  0.09234028,  0.00272974, -0.00363138,  0.00640345]])
    return cameraMatrix, Distortion_Parameters



backSub = cv.createBackgroundSubtractorKNN(detectShadows= False)

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
camera.AcquisitionFrameRate.SetValue(30.0)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
  
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # cv.namedWindow('title', cv.WINDOW_NORMAL)
        # ORIIMG = img.copy()

        # pylonFrame = calibPylon(cameraMatrix, Distortion_Parameters, img)    
        # img = img[200:800, 0:1920]
        
        pureImg = img.copy()
        pureImg2 = img.copy()
        madeImg= img.copy()
        originalImgshow = img.copy()
        maskImage = backSub.apply(img)

        a = cv.getTrackbarPos("a", "namespace")
        gray = cv.cvtColor(pureImg, cv.COLOR_BGR2GRAY)
        threshold = cv.threshold(gray, a, 255, cv.THRESH_BINARY)[1]

        #           determine width take a      #
        determineLocateLine(madeImg)

        #           take a picture          #
        saveImage(madeImg, originalImgshow, maskImage, pathSaveImage, False)



        # rect= processImage(pureImg, pathSaveImage, 1)

        rect= findThreshold(threshold)

        # value about weight as length
        subImage = calculatorLength(32.5, 1920, rect, threshold, madeImg, drawPolyline= True)[1]       
        subImage1 = calculatorLength(32.5, 1920, rect, pureImg, madeImg, writeLength= False, drawPolyline= True)[1]
        
        # length = calculatorLength(32.5, 1920, rect, threshold, madeImg, writeLength= False, drawPolyline= True)[0]   

        backImage, bellyImage, subImageCopy = isDetermineBackAndBelly(subImage1, subImage)
        


        mask = cv.bitwise_and(pureImg, pureImg, mask= threshold)
        cvtThreshold = np.stack((threshold, )*3, axis= -1)
        mergeImg = np.hstack((madeImg, mask, cvtThreshold))
        mergeImg = cv.resize(mergeImg, (0,0), fx=0.25, fy= 0.25)

        subImage = np.stack((subImage, )*3, axis= -1)
        # subImage1 = np.stack((subImage1, )*3, axis= -1)
        subMergeImage = np.hstack((subImage1, subImage, subImageCopy))
        subMergeImage = cv.resize(subMergeImage, (0,0), fx=0.25, fy= 0.25)

        maskKNN = cv.bitwise_and(pureImg2, pureImg2, mask= maskImage)
        Thre_maskKNN = np.stack((maskImage, )*3, axis= -1)
        mergeKNN = np.hstack((maskKNN, Thre_maskKNN))
        mergeKNN = cv.resize(mergeKNN, (0,0), fx=0.25, fy= 0.25)

        BBImage = np.hstack((backImage, bellyImage))
        BBImage = cv.resize(BBImage, (0,0), fx=0.25, fy= 0.25)


        cv.imshow('namespace', mergeImg)
        cv.imshow('show', subMergeImage)
        cv.imshow('background frame', mergeKNN)
        # cv.imshow('back and belly', BBImage)


        # cv.imshow('title', ORIIMG)
       

        k = cv.waitKey(1)

        if k == ord('q'):
            break

        elif k == ord('s'):
            cv.imwrite('D:\Project\Take a picture\image{}.jpg'.format(n), originalImgshow)
            # cv.imwrite('D:\Project\c\image{}.jpg'.format(n), originalImgshow)  
            n += 1
            print("Image saved !")

    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv.destroyAllWindows()

