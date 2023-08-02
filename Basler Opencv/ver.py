import cv2 as cv
import numpy as np

def nothing(x):
        pass

cv.namedWindow("namespace")
cv.createTrackbar("a", "namespace", 0, 255, nothing)

while True:

    img = cv.imread('D:\Project\Take a picture\image476_295.jpg')
    h, w = img.shape[:2]
    imgcpy = img.copy()
    a = cv.getTrackbarPos("a", "namespace")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray, 213, 255, cv.THRESH_BINARY)[1]

    try:
        contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        contour = sorted(contour, key= cv.contourArea, reverse= True)
        contour = contour[0]
        
        C = cv.moments(contour)

        if C["m00"] != 0:
            CX = int(C["m10"] / C["m00"])
            CY = int(C["m01"] / C["m00"])
        else:
            CX, CY = 0, 0

    except:
         pass
    
    x, y, w, h = cv.boundingRect(contour)

    cv.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
    cv.circle(img, (CX, CY), 2, (0,0,255), 3)
    cv.line(img, (int(w/2), 0), (int(w/2), h), (255,0,0), 1)
    cv.putText(img, 'cx={} cy={}'.format(CX, CY), (CX, CY), cv.FONT_HERSHEY_COMPLEX, 1, 255, 1)
    cv.putText(img, 'Length = {} pixel'.format(x+w), (CX, CY + 50), cv.FONT_HERSHEY_COMPLEX, 1, 255, 1)

    # if(CX < int(img.shape[1]/2)):
    #     cv.imwrite('D:\Project\images\image{}_{}.jpg'.format(CX, CY), imgcpy)
    #     print("image saved")

    # cv.drawContours(img, [contour], 0, (0, 255, 0), -1)
    mask = cv.bitwise_and(img, img, mask= threshold)
    cvtThreshold = np.stack((threshold, )*3, axis= -1)
    mergeImg = np.hstack((img, mask, cvtThreshold))
    mergeImg = cv.resize(mergeImg, (0,0), fx=0.5, fy= 0.5)
    cv.imshow('namespace', mergeImg)

    k = cv.waitKey(10)
    if k == 27:
        break
    # cv.waitKey(0)

cv.destroyAllWindows()

