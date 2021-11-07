import cv2
import numpy as np
from wand.image import Image


def resize_element(image, shape, resize_ratio, element_name):
    # crop ROI
    x, y, w, h, roi = crop_roi(image, shape, element=element_name)
    # generate points ans destinations
    trans_points = transform_points(
        shape, x, y, w, h, resize_ratio, element=element_name)
    # using shepard method to rescale element
    with Image.from_array(roi) as img:
        img.distort('shepards', trans_points)
        roi = np.array(img)

    image[y:y + h, x:x + w] = roi

    return image


def crop_roi(image, shape, element):
    points = None
    if element == "nose":
        points = np.vstack((shape[41], shape[46], shape[33]))
    elif element == "left_eye":
        #TODO: set a dynamic margin for buttom of the roi
        points = np.vstack((shape[26], shape[42], shape[45], shape[46], [
                           shape[47][0], shape[47][1]+50]))
    (x, y, w, h) = cv2.boundingRect(points)
    roi = image[y:y + h, x:x + w]
    
    return x, y, w, h, roi.copy()


def transform_points(shape, x, y, w, h, resize_ratio, element):
    # find ROI for nose (buttom point of left and right eye and the buttom point of the nose)
    if element == "nose":
        t0 = [shape[31][0]-x, round((shape[30][1]-y))]
        t1 = [shape[35][0]-x, round((shape[30][1]-y))]
        deltaX = t1[0]-t0[0]
        move = deltaX * resize_ratio
        t0_new = np.array([round(t0[0]-move), round(t0[1])])
        t1_new = np.array([round(t1[0]+move), round(t1[1])])

        points = (
            0, 0, 0, 0,
            w, 0, w, 0,
            t0[0], t0[1], t0_new[0], t0_new[1],
            t1[0], t1[1], t1_new[0], t1_new[1],
        )
    elif element == "left_eye":
        t0 = shape[43]-[x, y]
        t1 = shape[44]-[x, y]
        t2 = shape[47]-[x, y]
        t3 = shape[46]-[x, y]
        # t4 = shape[42]-[x, y]
        # t5 = shape[45]-[x, y]
        delta1 = t3-t0
        delta2 = t2-t1
        move1 = delta1*resize_ratio
        move2 = delta2*resize_ratio
        t0_new = t0-move1
        t3_new = t3+move1
        t1_new = t1-move2
        t2_new = t2+move2
        points = (
            t0[0], t0[1], t0_new[0], t0_new[1],
            t1[0], t1[1], t1_new[0], t1_new[1],
            t2[0], t2[1], t2_new[0], t2_new[1],
            t3[0], t3[1], t3_new[0], t3_new[1],
        )
        print(points)
    return points
