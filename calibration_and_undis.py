import cv2
import numpy as np
import glob
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.1)
objp = np.zeros(( 6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:6:1, 0:8:1].T.reshape(-1, 2)

objp *= 35 # the side length of each Checkerboard (mm)
obj_points = []  
img_points = [] 
size=(704,576) # Camera resolution

for dir in ["left", "right"]:

    images = glob.glob("input/calib/" + dir + "/*.jpg")
    dist_images = glob.glob("input/calib/" + dir + "/*.jpg")

    j = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)
        print(fname, ": find corners ", ret)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2)
            cv2.drawChessboardCorners(img, (6, 8), corners, ret)
            img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow(dir, img)
            cv2.waitKey(200)
            j += 1
            #Save checkerboard corner picture
            cv2.imwrite("Checkerboard_corner/" + dir + "/" + str(j) + ".png", img)

    print("the number of calib images: ", len(img_points))
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None, flags=cv2.CALIB_FIX_K3)
    print("mtx:")
    print(mtx) #Camera internal matrix
    print("dist:")
    print(dist) #Distortion coefficient
    print("rvecs:")
    print(rvecs) # rotation vector
    print("tvecs:")
    print(tvecs) #translation vector
    print("calib done !")


    for i in range(len(dist_images)):  #Remove distortion
        img = cv2.imread(dist_images[i])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, size, 1, size)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        print(dist_images[i], ": distortion has been removed")
        cv2.imshow(dir, dst)
        cv2.waitKey(200)
        #Save the distorted image in the undis folder
        cv2.imwrite("undis/" + dir + "/" + str(i+1) + ".png", dst)

    cv2.destroyAllWindows()
    print("dist  done!")

