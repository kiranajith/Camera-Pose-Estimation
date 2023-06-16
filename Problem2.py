import cv2
import numpy as np

img1 = cv2.imread('image_1.jpg')
img2 = cv2.imread('image_2.jpg')
img3 = cv2.imread('image_3.jpg')
img4 = cv2.imread('image_4.jpg')

img1_r = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
img2_r = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
img3_r = cv2.resize(img3, (0,0), fx=0.5, fy=0.5)
img4_r = cv2.resize(img4, (0,0), fx=0.5, fy=0.5)

gray1 = cv2.cvtColor(img1_r, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_r, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3_r, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4_r, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)
kp4, des4 = sift.detectAndCompute(gray4, None)

desc = [des1, des2, des3, des4]
key_points = [kp1, kp2, kp3, kp4]

bf = cv2.BFMatcher()

match = []
sizes = []

for i in range(3):
    matches = bf.knnMatch(desc[i], desc[i+1], k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.25*n.distance:
            good.append(m)
    match.append(good)
    sizes.append((gray1.shape[1], gray1.shape[0]))

hm = []
for i in range(3):
    src = np.float32([key_points[i][m.queryIdx].pt for m in match[i]])
    dst = np.float32([key_points[i+1][m.trainIdx].pt for m in match[i]])

    A = []
    for j in range(len(src)):
        x, y = src[j][0], src[j][1]
        u, v = dst[j][0], dst[j][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)

    _, _, V = np.linalg.svd(A)
    h = V[-1,:] / V[-1,-1]
    homography = np.reshape(h, (3, 3))
    hm.append(homography)

res = cv2.warpPerspective(img1_r, hm[0], sizes[0])
res = np.concatenate((res, cv2.warpPerspective(img2_r, hm[1], sizes[1])), axis=1)
res = np.concatenate((res, cv2.warpPerspective(img3_r, hm[2], sizes[2])), axis=1)
res = np.concatenate((res, img4_r), axis=1)

cv2.imshow('Panoramic Image', res)
cv2.waitKey(0) == ord('q')
cv2.destroyAllWindows()
