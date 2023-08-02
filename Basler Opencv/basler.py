import os, sys
from pypylon import pylon
import cv2
import numpy as np
import glob
import pandas as pd

class camera():
    def __init__(self):
        self.tl_factory = pylon.TlFactory.GetInstance()

        self.devices = self.tl_factory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(self.devices), 2))

        for idx, cam in enumerate(self.cameras):
            cam.Attach(self.tl_factory.CreateDevice(self.devices[idx]))
            cam.Open()
            # cam.PixelFormat = "Mono8"
            # cam.Width = cam.Width.Max
            # cam.Height = cam.Height.Max
            # pylon.FeaturePersistence.Load(nodeFile[idx], cam[idx].GetNodeMap(), True)
            print("Using device", cam.GetDeviceInfo().GetModelName())
        
        
        self.matPath = "Library/camset/9.2.23/Matrix.csv"
        self.distPath = "Library/camset/9.2.23/Distort.csv"
    
        # Chuyển đổi hệ màu sang RGB (Opencv)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def OpenCam(self):
        # Mở camera
        # self.cam_array.Open()
        # pylon.FeaturePersistence.Load(self.nodeFile1, self.cameras.GetNodeMap(), True)

        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        

    def creatframe(self, camera):
        grab = self.cameras[camera].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            # self.cameraContextValue = grab.GetCameraContext()
            self.frame = self.converter.Convert(grab)
            self.frame = self.frame.GetArray()
            #self.frame = self.undistortImg(self.frame)
        return self.frame

    def undistortImg(self, img, matPath, distPath):
        h,  w = img.shape[:2]
        
        # Đọc các file config
        matrix = pd.read_csv(matPath, header=None).values
        distortion = pd.read_csv(distPath, header=None).values

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))
        # Chống bóp méo hình ảnh
        res = cv2.undistort(img, matrix, distortion, None, newcameramtx)
        res = res[roi[1]:roi[3], roi[0]:roi[2]]

        return res
        
    def calibCam(self, CheckerboardFolderPath, savingPath, boardSize, save = False):
        CHECKERBOARD = boardSize
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[ :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        threedpoints = []
        twodpoints = []

        images = glob.glob(f'{CheckerboardFolderPath}/Chest_*.png')

        ret = False
        for filename in images:
            print(filename)           
            # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # grayColor = image
            image = cv2.imread(filename)
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, threshCB = cv2.threshold(grayColor, 150, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            threshCB = cv2.morphologyEx(threshCB, cv2.MORPH_OPEN, kernel, iterations=1)

            cv2.imshow("test", threshCB)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            res = np.uint8(threshCB)
            ret, corners = cv2.findChessboardCorners(res, CHECKERBOARD,
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)

            
            # ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)
            print(ret)
            
            if ret == True:
                threedpoints.append(objectp3d)
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
                twodpoints.append(corners2)
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

                ret, matrix, distortion, rotateVecs, translateVecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

                mat = pd.DataFrame(matrix)
                dist = pd.DataFrame(distortion)
                
                if save is True:
                    mat.to_csv(f"{savingPath}/Matrix_3.5.csv", header=False, index=False)
                    dist.to_csv(f"{savingPath}/Distort_3.5.csv", header=False, index=False)
                

                cv2.imshow("final", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    def scale(self, img, cnt):
        h, w = img.shape[:2]
        scaledH = int(h*cnt)
        scaledW = int(w*cnt)
        dim = (scaledW, scaledH)
        res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return res


if __name__ == "__main__":
    cam = camera()
    cam.OpenCam()
    while cam.cameras.IsGrabbing:
        frame = cam.creatframe(0)
        frame = cam.scale(frame, 0.5)
        h, w = frame.shape[:2]
        cen_h = h/2
        cen_w = w/2
        cv2.line(frame, (0,int(cen_h)), (w,int(cen_h)), (0,0,255), 1)
        cv2.line(frame, (int(cen_w), 0), (int(cen_w), h), (0,0,255), 1)
        cv2.imshow('window0', frame)

        key = cv2.waitKey(1)
        # Thoát bằng nút 'q'
        if key == ord('q'):
            print("Stop acquiring camera")
            cv2.destroyAllWindows()
            break

        