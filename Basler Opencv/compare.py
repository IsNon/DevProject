import cv2 as cv
import numpy as np

def nothing(x):
        pass


cv.namedWindow("namespace")
cv.createTrackbar("a", "namespace", 0, 255, nothing)


def getSubImage(rect, src, img, drawPolyline = False):
        
    box = cv.boxPoints(rect)
    box = np.intp(box)
        
    width = int(rect[1][0])
    height = int(rect[1][1])

    
    theta = rect[2]

    if drawPolyline == True:
        cv.polylines(img, [box], True, (0,0,255), 3)

        cv.line(img, (box[0][0], box[0][1] - 50), (box[1][0], box[1][1] - 50), (255,0,0), 5) 

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

def calculatorLength(lengthFrame, widthPixelFrame, rect, threshold, img, writeLength = False, drawPolyline = False):

    subImage = getSubImage(rect, threshold, img, drawPolyline)[0]

    length = subImage.shape[1]* lengthFrame / widthPixelFrame

#     print("width = " , subImage.shape[1] )

    if writeLength == True:
        print("length = " , length)

    return length, subImage

def findContour(img):
    a = cv.getTrackbarPos("a", "namespace")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray, a, 255, cv.THRESH_BINARY)[1]
    contour = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if len(contour)>0:
        contour = sorted(contour, key= cv.contourArea, reverse= True)
        contour = contour[0]
        rect = cv.minAreaRect(contour)

    return rect, threshold

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


cameraMatrix = np.array([[1.50392060e+03, 0.00000000e+00, 1.00861013e+03],
                [0.00000000e+00, 1.50877936e+03, 5.83831826e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

Distortion_Parameters = np.array([[-0.15297373,  0.09234028,  0.00272974, -0.00363138,  0.00640345]])

path = 'c'

while True:
        img = cv.imread('D:\Project\c\image0.jpg')
        imgcalib = calibInputImage(img, cameraMatrix, Distortion_Parameters, path, id=1)
        # print(img.shape, imgcalib.shape)
        a = cv.getTrackbarPos("a", "namespace")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        threshold = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)[1]
        contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        if len(contour)>0:
            contour = sorted(contour, key= cv.contourArea, reverse= True)
            contour = contour[0]
            rect = cv.minAreaRect(contour)

            subimage = getSubImage(rect, threshold, img)[0]
            calculatorLength(32.5, 1920, rect, threshold, img, writeLength= True)

            threshold = np.stack((threshold, )*3, axis=-1)   
            # subimage = np.stack((subimage, )*3, axis=-1) 
            mergeImage = np.hstack((img, threshold)) 
            mergeImage =  cv.resize(mergeImage, (0,0), fx=0.25, fy=0.25)

            cv.imshow('namespace', mergeImage)
            cv.imshow('subimage', subimage)


        k = cv.waitKey(1)
        if k == 27:
            break

cv.destroyAllWindows()