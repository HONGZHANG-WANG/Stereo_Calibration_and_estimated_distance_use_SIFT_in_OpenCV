import numpy as np
import cv2
import glob
#Author: Lizaozao
#Time: 2021.1.10
#----------------------------------------------------------------------
# Must read before running!
# There are two folders in the folder: left and right under calib.
# The corresponding left and right view names in the left and right
# folders must be the same. Otherwise, the calibration will fail.
# because when glob() reads files, it determines the order according to
# the file name. For example, the left and right views of the same frame
# should be named 1.jpg and stored in two folders
#----------------------------------------------------------------------
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
objp = np.zeros((8*6,3), np.float32) # 8*6 is the number of corners
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)
objp *= 25 # the side length of each Checkerboard (mm)
size=(704,576) # Camera resolution
objpoints_left = [] # 3D points for storing world coordinates
objpoints_right = []
imgpoints_left = [] # 2D points for storing picture coordinates
imgpoints_right = []

images_left = glob.glob('input/calib/left/*.jpg')
for fname in images_left:
    img_left = cv2.imread(fname)
    gray_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    # Find the corner of chessboard
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6,8),None)
    # If found, add the corner information to the list
    if ret_left == True:
        objpoints_left.append(objp)
        #Finding subpixel corners
        cv2.cornerSubPix(gray_left,corners_left,(5,5),(-1,-1),criteria)
        imgpoints_left.append(corners_left)
        # Show corners found
        cv2.drawChessboardCorners(img_left, (6,8), corners_left, ret_left)
        img_left=cv2.resize(img_left, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('img',img_left)
        cv2.waitKey(200)# Delay 200ms, easy to display

cv2.destroyAllWindows()
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints_left, imgpoints_left, size,None,None,flags=cv2.CALIB_FIX_K3)

print('mtx_left')
print(mtx_left)
print('dist_left')
print(dist_left)

images_right = glob.glob('input/calib/right/*.jpg')
# 同上
for fname in images_right:
    
    img_right = cv2.imread(fname)
    gray_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)

    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6,8),None)
    if ret_right == True:
        objpoints_right.append(objp)

        cv2.cornerSubPix(gray_right,corners_right,(5,5),(-1,-1),criteria)
        imgpoints_right.append(corners_right)

        cv2.drawChessboardCorners(img_right, (6,8), corners_right, ret_right)
        img_right=cv2.resize(img_right, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('img',img_right)

        cv2.waitKey(200)

cv2.destroyAllWindows()
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints_right, imgpoints_right, size, None, None,flags=cv2.CALIB_FIX_K3)

print('mtx_right')
print(mtx_right)
print('dist_right')
print(dist_right)
print("----------------------------------------------")

ret,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(objpoints_left, imgpoints_left, imgpoints_right, mtx_left, dist_left,  mtx_right, dist_right, gray_left.shape[::-1],criteria,None)
# print('cameraMatrix1')
# print(cameraMatrix1)
print('R')
print(R)
print('T')
print(T)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, size, R, T)
print('Q') 
print(Q)

print('R1')
print(R1)
print('P1')
print(P1)
print('R2')
print(R2)
print('P2')
print(P2)
print("----------------------------------------------")

#----------------------------------------------------------------------------
# The above part is to calculate the camera parameters, just run it once
# You can save the parameters in a file for easy calling
# This function is not implemented yet and is being implemented
#----------------------------------------------------------------------------

left_map1, left_map2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P2, size, cv2.CV_16SC2) 
right_map1, right_map2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P1, size, cv2.CV_16SC2)



#Turn on the camera to get the left and right views, but they haven't been tested

cap1=cv2.VideoCapture(0) #left
cap2=cv2.VideoCapture(1) #right
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 704) #setting width
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 576) #setting hight

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 704)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

curent_cap = False # If you use an existing image, it will be false

while(curent_cap):
    ret1 ,frame1 = cap1.read()
    ret2, frame2 = cap1.read()
    frame = np.hstack([frame1, frame2])
    frame = cv2.resize(frame, (0, 0), fx=1, fy=1)  # Window zoom, where 1 is no zoom
    cv2.imshow("capture", frame)
    k=cv2.waitKey(40)
    if k==ord('q'):
       break
    if k==ord('s'):
        cv2.imwrite("cap/left/cap.jpg",frame1)
        cv2.imwrite("cap/right/cap.jpg", frame2)
        cap1.release()
        cap2.release()
        break
#--------------------------------------------
# Load picture
#frame1 = cv2.imread("cap/left/cap.jpg") # If you use a camera, use these
#frame2 = cv2.imread("cap/right/cap.jpg") #frame1 is left , frame2 is right
#--------------------------------------------
frame1 = cv2.imread("input/underwear_trepang/trepang_left/trepang left_1.jpg")
frame2 = cv2.imread("input/underwear_trepang/trepang_right/trepang right_1.jpg")  #frame1 is left, frame2 is right
#--------------------------------------------
img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
img1_copy=img1_rectified.copy()
img2_copy=img2_rectified.copy()
cv2.imwrite("output\\left_rectified.jpg", img1_copy)
cv2.imwrite("output\\right_rectified.jpg", img2_copy)

i=40
while(1):
    cv2.line(img1_rectified,(0,i),(size[0],i),(0,255,0),1)
    cv2.line(img2_rectified,(0,i),(size[0],i),(0,255,0),1)
    i+=40
    if i > size[1]:
            break

imgsall = np.hstack([img1_rectified,img2_rectified])
cv2.imwrite("output\\rectified.jpg", imgsall)
imgL = cv2.cvtColor(img1_copy, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img2_copy, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_copy, None)
kp2, des2 = sift.detectAndCompute(img2_copy, None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
good = [] # It is used to store good matching point pairs
point = [] # Used to store matching point to coordinate

label_top = (548, 276) #Coordinates of the upper left corner of the box
label_bottom = (700, 406)#Coordinates of the lower left corner of the box

target_point = (int(label_top[0] + (label_bottom[0] - label_top[0]) / 2),
                int(label_top[1] + (label_bottom[1] - label_top[1]) / 2))


# print(kp1[good[1][0].queryIdx].pt)
# print(kp2[good[1][0].trainIdx].pt)
for i, (m1, m2) in enumerate(matches): # Filter out the unqualified point pairs
    if m1.distance < 1.5 * m2.distance:
        # Filter condition 1: the smaller the Euclidean distance
        # between two feature vectors, the higher the matching degree
        if (label_top[0] < kp1[m1.queryIdx].pt[0] < label_bottom[0] and
        label_top[1] < kp1[m1.queryIdx].pt[1] < label_bottom[1]):
            if abs(kp1[m1.queryIdx].pt[1] - kp2[m1.trainIdx].pt[1]) < 5:
                # Filter condition 2: because the epipolar line has been
                # corrected, the matching point pair should be on the horizontal
                # line, and a threshold of 10 pixels can be set, which can be smaller
                good.append([m1])
                pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
                pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
                # print(i, pt1, pt2)
                point.append([pt1, pt2])
img4 = cv2.drawMatchesKnn(img1_copy, kp1, img2_copy, kp2, matches, None, flags=2)
cv2.imwrite("output\SIFT_MATCH_no_filter.jpg", img4)
img4 = cv2.resize(img4, (0,0), fx=0.6, fy=0.6)
cv2.imshow("sift-no_filter", img4)
img3 = cv2.drawMatchesKnn(img1_copy, kp1, img2_copy, kp2, good, None, flags=2)
cv2.rectangle(img3, label_top, label_bottom, (0, 0, 255), 2)
cv2.imwrite("output\SIFT_MATCH_filtered.jpg", img3)
img3 = cv2.resize(img3, (0,0), fx=0.6, fy=0.6) # Zoom 0.6 times
cv2.imshow("sift_filtered", img3)

def img2_3D(target_point,label_top, label_bottom, point_cloud, Q):
    # target_ x, target_ Y is the coordinate of the point we want
    # to measure the distance. Because the point does not necessarily
    # match, we will search for the nearest matching point pair, and
    # use the distance of the point pair to approximate the distance
    # of the point we want
    final_point = []
    min_distance = 1000000
    for i in range(len(point)):
        distance = (target_point[0] - point_cloud[i][0][0]) ** 2 \
                   + (target_point[1] - point_cloud[i][0][1]) ** 2
        if distance < min_distance :
            final_point = point[i]
            min_distance = distance
    # print(final_point)


    new_img=np.zeros(size,dtype=np.float32)

    x = int(final_point[0][0])
    y = int(final_point[0][1])
    print("input coordinates of the point：({}, {})".format(target_point[0], target_point[1]))
    print("The point coordinates closest to the input point：({}, {})".format(x, y))
    cv2.circle(img1_copy, (x, y), 3, (0, 0, 255), -1)
    cv2.circle(img1_copy, (target_point[0], target_point[1]), 3, (0, 255, 0), -1)
    disp = final_point[0][0] - final_point[1][0]
    print("Parallax at this point: ", disp)

    new_img[x][y] = disp
    threeD = cv2.reprojectImageTo3D(new_img, Q)
    print("The three-dimensional coordinates of the point are: ", threeD[x][y])
    print("The depth of the point is: {:.2f} m".format(threeD[x][y][2] / 1000))

img2_3D(target_point, label_top, label_bottom, point, Q) # 调用函数，前两个参数就是想要测距的坐标

img1_copy=cv2.resize(img1_copy, (0,0), fx=1, fy=1)
img2_copy=cv2.resize(img2_copy, (0,0), fx=1, fy=1)
imgsall=cv2.resize(imgsall, (0,0), fx=0.7, fy=0.7)
cv2.imshow("left_rectified", img1_copy)
cv2.imshow("right_rectified", img2_copy)
cv2.imshow("rectified", imgsall)

def callbackFunc(e, x, y, f, p):
   if e == cv2.EVENT_LBUTTONDOWN:
       print("The point coordinates are: ({}, {})".format(x, y))
cv2.setMouseCallback("left_rectified", callbackFunc, None)
# If you don't know the coordinates, clicking the left_rectified image
# will return the coordinates of the location

cv2.waitKey(0)
cv2.destroyAllWindows()