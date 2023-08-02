import cv2 as cv
import os
from pypylon import pylon
import numpy as np


CHESS_BOARD_DIM = (15, 9)

n = 0  # image_counter

# checking if  images dir is exist not, if not then create images directory
image_dir_path = "take"

CHECK_DIR = os.path.isdir(image_dir_path)
# if directory does not exist create
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already Exists.')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


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

        frame = img.copy()
        copyFrame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
        # print(ret)
        cv.putText(
            frame,
            f"saved_img : {n}",
            (30, 40),
            cv.FONT_HERSHEY_PLAIN,
            1.4,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        gray = np.stack((gray, )*3, axis= -1)
        merge = np.hstack((frame, copyFrame, gray))
        merge = cv.resize(merge, (0,0), fx=0.25, fy= 0.25)

        cv.imshow("merge", merge)

        key = cv.waitKey(1)

        if key == ord("q"):
            break
        if key == ord("s") and board_detected == True:
            # storing the checker board image
            cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)

            print(f"saved image number {n}")
            n += 1  # incrementing the image counter

    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv.destroyAllWindows()          