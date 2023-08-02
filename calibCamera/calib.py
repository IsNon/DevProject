import numpy as np
import cv2 as cv
import glob, os, pickle


# checking if images dir is exits not, if not then create images directory
image_dir_path = "images"

CHECK_DIR =  os.path.isdir(image_dir_path)

n = 0 # image_counter

#if directory does not exist create
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created ')
else:
    print(f'"{image_dir_path}" Directory already Exist ')


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)  
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (11,11), (-1,-1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret


#   FIND CHESSBOARD CORNERS
chessboardSize = (24, 17) # a number of interesion corners follow width and heigth
frameSize = (1920, 1120) # pixel of cameras

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0: chessboardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = os.listdir(image_dir_path)

for fname in images:

    print(fname)
    imagePath = os.path.join(image_dir_path, fname)

    img = cv.imread(imagePath)


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

# calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
pickle.dump((cameraMatrix, dist), open('calibration.pkl', "wb"))
pickle.dump(cameraMatrix, open('cameraMatrix.pkl', "wb"))
pickle.dump(dist, open('calibration.pkl', "wb"))

#               the same feature            #
# THE FIRST METHOD
# Undistortion  
img = cv.imread('Image1.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

#THE SECOND METHOD
# undistort
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

#               the same feature            #


# check error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

# cap = cv.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     copyFrame = frame.copy()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     image, board_deteced = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
#     cv.putText(frame, f'saved_img:{n}', (30, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
#     cv.imshow("frame", frame)
#     cv.imshow("copy frame", copyFrame)

#     key = cv.waitKey(1)

#     if key == ord('q'):
#         break
#     if key == ord('s') and board_deteced == True:
#         cv.imwrite(f'{image_dir_path}/image{n}.png', copyFrame)

#         print(f'saved image number {n}')
#         n += 1

# cap.release()
# cv.destroyAllWindows()

# print("toatl saved image :", n)