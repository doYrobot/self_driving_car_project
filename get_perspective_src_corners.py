import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

nx = 9
ny = 6

objpoints =[]
imgpoints = []

objp = np.zeros((nx*ny,3),np.float32)

objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

fnames = glob.glob("CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg")
for fname in fnames:
    #img = mpimg.imread('camera_cal/calibration11.jpg')
    img = mpimg.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)

    if ret == True:

        objpoints.append(objp)
        imgpoints.append(corners)


ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)


image_test = mpimg.imread('CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg')
undistort_img = cv2.undistort(image_test,mtx,dist,None,mtx)

plt.imshow(image_test)

plt.show()
