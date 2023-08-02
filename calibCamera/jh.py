'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2
import os
import numpy as np

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv2.findChessboardCorners(grayImage, boardDimension)  
    if ret == True:
        corners1 = cv2.cornerSubPix(grayImage, corners, (11,11), (-1,-1), criteria)
        image = cv2.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret



# checking if images dir is exits not, if not then create images directory
image_dir_path = "D:\Project\imageCalib"
CHESS_BOARD_DIM = (15,9)

CHECK_DIR =  os.path.isdir(image_dir_path)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
n = 0 # image_counter

#if directory does not exist create
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created ')
else:
    print(f'"{image_dir_path}" Directory already Exist ')



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

        # cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        frame = img
        copyFrame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        image, board_deteced = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
        
        cv2.putText(frame, f'saved_img:{n}', (30, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        gray = np.stack((gray, )*3, axis= -1)
        merge = np.hstack((frame, copyFrame, gray))
        merge = cv2.resize(merge, (0,0), fx=0.25, fy= 0.25)

        cv2.imshow("merge", merge)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        if key == ord('s') and board_deteced == True:
            cv2.imwrite(f'{image_dir_path}/imagess{n}.png', copyFrame)
            print(f'saved image number {n}')
            n += 1

        # cv2.imshow('title', img)

    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()