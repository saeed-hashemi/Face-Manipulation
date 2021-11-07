# This file must be empty on commit
from numpy.lib.type_check import imag
from wand.image import Image
from itertools import chain
from utils.face_landmarks import find_face_landmarks
import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

resize_ratio = +.2/2
# read image from file
image = cv2.imread("face.jpg")
image = cv2.GaussianBlur(image, (11, 11), 0)

image = cv2.resize(image, (round(
    image.shape[1]/1), round(image.shape[0]/1)), interpolation=cv2.INTER_AREA)
# Convert the image color to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect the face
rects = detector(gray, 1)
# Detect landmarks for each face
for rect in rects:
    shape = find_face_landmarks(gray, rect)
    if shape.any():
        print(np.vstack((shape[40], shape[47], shape[33])))
        (x, y, w, h) = cv2.boundingRect(
            np.vstack((shape[41], shape[46], shape[33])))
        roi = image[y:y + h, x:x + w]

        roi_c = roi.copy()

        # transpose 20%

        t0 = [shape[31][0]-x, round((shape[30][1]-y))]
        t1 = [shape[35][0]-x, round((shape[30][1]-y))]
        deltaX = t1[0]-t0[0]
        move = deltaX * resize_ratio
        t0_new = np.array([round(t0[0]+move), round(t0[1])])
        t1_new = np.array([round(t1[0]-move), round(t1[1])])

        # apply prespective
        pts = np.array([[0, 0], [w, 0], t0, t1], dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], t0_new, t1_new], dtype=np.float32)

        with Image.from_array(roi_c) as img:
            img.distort('shepards', (
                pts[0][0], pts[0][1], pts[0][0], pts[0][1],
                pts[1][0], pts[1][1], pts[1][0], pts[1][1],
                pts[2][0], pts[2][1], dst[2][0], dst[2][1],
                pts[3][0], pts[3][1], dst[3][0], dst[3][1],
            ))
            roi_c = np.array(img)

        image[y:y + h, x:x + w] = roi_c


############################################################################


(x, y, w, h) = cv2.boundingRect(
    np.vstack((shape[26], shape[42], shape[45], shape[46], [shape[47][0], shape[47][1]+50])))
roi = image[y:y + h, x:x + w]
t0 = shape[43]-[x, y]
t1 = shape[44]-[x, y]
t2 = shape[47]-[x, y]
t3 = shape[46]-[x, y]
t4 = shape[42]-[x, y]
t5 = shape[45]-[x, y]
# cv2.circle(image, t0, 1, (0, 0, 255), -1)
# cv2.circle(image, shape[35], 1, (0, 0, 255), -1)

roi_c = roi.copy()
delta1 = t3-t0
delta2 = t2-t1
move1 = delta1*resize_ratio
move2 = delta2*resize_ratio
t0_new = t0+move1
t3_new = t3-move1
t1_new = t1+move2
t2_new = t2-move2
print(t0, t0_new, t1, t1_new)
with Image.from_array(roi_c) as img:
    img.distort('shepards', (
                t0[0], t0[1], t0_new[0], t0_new[1],
                t1[0], t1[1], t1_new[0], t1_new[1],
                t2[0], t2[1], t2_new[0], t2_new[1],
                t3[0], t3[1], t3_new[0], t3_new[1],
                ))
    # img.distort('shepards', (

    # ))
    roi_c = np.array(img)

image[y:y + h, x:x + w] = roi_c
# Display the landmarks
# for i, (x, y) in enumerate(shape):
#     # Draw the circle to mark the keypoint
#     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


# image = cv2.blur(image,(3,3))

# Display the image
image = cv2.blur(image, (3, 3))
cv2.imwrite('result.jpg', image)
