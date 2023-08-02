import cv2 as cv
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from untitled import Ui_MainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# class My_UI(Ui_MainWindow):
#     def __init__(self):
#         self.setupUi(MainWindow)
#         self.Slider.setValue(0)
#         self.Slider.valueChanged.connect(self.handle)

#     def handle(self):
#         self.value = self.Slider.value()
#         self.textEdit.setText(str(self.value))

#         img = cv.imread('D:\Project\images\image.jpg')

#         # while True:
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         threshold = cv.threshold(gray, self.value, 255, cv.THRESH_BINARY_INV)[1]

#         try:
#             contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
#             # print(len(contour))
#             contour = sorted(contour, key= cv.contourArea, reverse= True)[:3]
#             # print(contour[0], contour[1])
#             cv.drawContours(img, [contour[0]], 0, (0,255,0), -1)

#         except:
#             pass

#         cv.putText(img, str(self.value), (100,200), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

#         # threshold = np.stack((threshold, )*3, axis= -1)
#         # print(threshold.shape)
#         # cv.imshow('img', np.hstack(img))
#         # print(self.value)

#         cv.imshow('img', threshold)
#         cv.waitKey()
#         # k = cv.waitKey(1)
#         # if k == 27:
#         #     break

#         cv.destroyAllWindows()

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = My_UI()
#     MainWindow.show()
#     sys.exit(app.exec_())


def nothing(x):
    pass

cv.namedWindow('img')
cv.createTrackbar('threshol', 'img', 0, 255, nothing)
img = cv.imread('D:\Project\images\image.jpg')

image = np.zeros((img.shape), np.uint8)

while(True):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.getTrackbarPos('threshol', 'img')
    threshold = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)[1]

    try:
        contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        # print(len(contour))
        contour = sorted(contour, key= cv.contourArea, reverse= True)[:3]
        # print(contour[0], contour[1])
        cv.drawContours(img, [contour[0]], 0, (0,255,0), -1)

    except:
        pass

    cv.putText(img, str(thresh), (100,200), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

    # threshold = np.stack((threshold, )*3, axis= -1)
    # print(threshold.shape)
    # cv.imshow('img', np.hstack(img))
    # print(thresh)
    cv.imshow('img', img)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()



