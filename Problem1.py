import cv2
import numpy as np
from scipy.spatial.transform import Rotation
cap = cv2.VideoCapture('/Users/kiranajith/Documents/UMD/673/Project2/project2.avi')

paper_w_cm = 21.6
paper_h = 27.9
world_corner = np.array([[0, 0, 0], [0, paper_h, 0], [paper_w_cm, paper_h, 0], [paper_w_cm, 0, 0]])

k_matrix = np.array([[1380.0, 0, 946.0],[0, 1380.0, 527.0],[0, 0, 1]])

def compute_homography(world_points, image_points):
    """function to compute homography

    Args:
        world_points (_type_): world frame
        image_points (_type_): points in image
    """
    A = []
    for i in range(4):
        X, Y, _ = world_points[i]
        u, v = image_points[i]
        A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    H = H / H[2, 2]

    return H

def rot_trans(H):
    """function to get the rotaion and translation matrices

    """
    r_1 = H[:, 0]
    r_2 = H[:, 1]
    t = H[:, 2]
    scale = np.sqrt(np.linalg.norm(r_1) * np.linalg.norm(r_2))
    r_1 = r_1 / scale
    r_2 = r_2 / scale
    r3 = np.cross(r_1, r_2)
    R = np.column_stack((r_1, r_2, r3))
    t = t / scale

    r = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

    return R, t, r

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(thresh, (3, 3), 0)

    edges = cv2.Canny(blur, 50, 150, apertureSize=3)


    rho_acc_resolution = 1
    theta_acc_resolution = np.pi/180
    threshold = 150
    minL = 100  
    maxL = 10
    lines = []
    diagonal_length = int(np.ceil(np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)))
    accl = np.zeros((int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)/rho_acc_resolution)+1, 180))
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] > 0:
                for angle in range(0, 180):
                    theta_radians = angle * np.pi / 180
                    rho_val = j * np.cos(theta_radians) + i * np.sin(theta_radians)
                    accl[int(rho_val), angle] += 1

    for i in range(accl.shape[0]):
        for j in range(accl.shape[1]):
            if accl[i, j] > threshold:
                x1 = int(i - maxL * np.sin(j * np.pi / 180))
                y1 = int(j + maxL * np.cos(j * np.pi / 180))
                x2 = int(i + maxL * np.sin(j * np.pi / 180))
                y2 = int(j - maxL * np.cos(j * np.pi / 180))
                lines.append([[x1, y1, x2, y2]])

    cnrs = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x1, y1 = lines[i][0][:2]
            x2, y2 = lines[j][0][:2]
            intersection = np.cross([x1, y1, 1], [x2, y2, 1])
            if intersection[2] != 0:
                x = intersection[0] / intersection[2]
                y = intersection[1] / intersection[2]
                cnrs.append([x, y])

    cnrs = np.array(cnrs)

    H = compute_homography(world_corner, cnrs)
    R, t, r = rot_trans(H)
    print('Pose Estimation')
    print(f"Rotation\nRoll: {r[0]:.2f}, Pitch: {r[1]:.2f}, Yaw: {r[2]:.2f}\nTranslation\n x: {t[0]:.2f}, y: {t[1]:.2f}, z: {t[2]:.2f}")
    for i in range(len(cnrs)):
        cv2.circle(frame, tuple(cnrs[i].astype(int)), 5, (0, 0, 255), -1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()