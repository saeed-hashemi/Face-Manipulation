import cv2
import numpy as np
from wand.image import Image


def resize_element(image, shape, resize_ratio, element_name):
    """
    resize nose or left_eye in 4 steps:
        - crop ROI
        - transpose points to the destination
        - resize the elements by using the calculated points and distortion with Shepards method
        - replace the distorted ROI inside the image

    Parameters
    ---------------------------
    image: numpy.ndarray-> mat, required
    shape: numpy 2D array-> landmarks points, required
    resize_ratio: float -> resizing element, required
    element_name: string -> can be "nose" or "left_eye", required

    Returns
    ---------------------------
    image
        OpenCV mat image -> applied resized element inside the ROI 
    """
    # crop ROI
    x, y, w, h, roi = crop_roi(image, shape, element=element_name)
    # generate points and destinations
    trans_points = transpose_points(
        shape, x, y, w, h, resize_ratio, element=element_name)
    # using shepard method to rescale element
    with Image.from_array(roi) as img:
        # making localization to avoid breaking other parts (default value is 2)
        img.artifacts['shepards:power'] = '2'
        img.distort('shepards', trans_points)
        roi = np.array(img)

    image[y:y + h, x:x + w] = roi

    return image


def crop_roi(image, shape, element):
    """
    crop the ROI for the nose (bottom point of the left and right eye and the bottom point of the nose)
    or left_eye (left corner of eyebrow and right corner of eye and a dynamic vertical margin for bottom)  elements by using the landmarks

    Parameters
    ---------------------------
    image: numpy.ndarray-> mat, required
    shape: numpy 2D array-> landmarks points, required
    element: string -> can be "nose" or "left_eye", required

    Returns
    ---------------------------
    x, y, w, h, roi : coordinates, width, and height of the ROI. The last value is a copy of the ROI.

    """
    points = None
    if element == "nose":
        points = np.vstack((shape[41], shape[46], shape[33]))
    elif element == "left_eye":

        vertical_margin = shape[47]+(shape[47]-shape[44])
        points = np.vstack(
            (shape[26], shape[42], shape[45], shape[46], vertical_margin))
    (x, y, w, h) = cv2.boundingRect(points)
    roi = image[y:y + h, x:x + w]

    return x, y, w, h, roi.copy()


def transpose_points(shape, x, y, w, h, resize_ratio, element):
    """
    Transposing the key points by resize_ratio

    Parameters
    ---------------------------
    shape: numpy 2D array-> landmarks points, required
    x: integer
    y: integer
    w: integer
    h: integer
    resize_ratio: float -> resizing element, required
    element: string -> can be "nose" or "left_eye", required

    Returns
    ---------------------------
    set of points (x0,y0,x0_transposed,y0_transposed, ...)-> for 3 points contains 12 values
    """
    
    if element == "nose":
        t0 = np.array([shape[31][0]-x, round((shape[30][1]-y))])
        t1 = np.array([shape[35][0]-x, round((shape[30][1]-y))])

        deltaXY = t1-t0
        # TODO: divide the resize_ratio base on the distance of each side of nose to the center (to better suport side face)
        move = np.round(deltaXY * resize_ratio)
        t0_new = t0-move
        t1_new = t1+move

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
    return points
