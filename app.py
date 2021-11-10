# import the necessary packages
import argparse
import cv2
import dlib
from utils.face_landmarks import find_face_landmarks
from utils.manipulation import resize_element

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    app = argparse.ArgumentParser()
    app.add_argument("-i", "--image", type=str, default="face.jpg",
                     help="path to input image")
    app.add_argument("-t", "--threshold", type=str, default=-20,
                     help="percentage of resizing ratio")
    app.add_argument("-r", "--resize", type=str, default=1,
                     help="resizing the input image")
    args = vars(app.parse_args())

    # initilize the model
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(args["image"])
    if image is not None:
        # image = cv2.GaussianBlur(image, (11, 11), 0)
        image = cv2.resize(image, (round(
            image.shape[1]*float(args["resize"])), round(image.shape[0]*float(args["resize"]))), interpolation=cv2.INTER_AREA)

        # initilize the resize ratio for nose and left eye
        resize_ratio = int(args["threshold"])/200

        elements = ["nose", "left_eye"]

        # Convert the image color to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the face
        rects = detector(gray, 1)

        # Detect landmarks for each face (if there is more that one face in a frame)
        for rect in rects:
            shape = find_face_landmarks(gray, rect)

            if shape.any():
                for element in elements:
                    image = resize_element(image, shape, resize_ratio, element)
                cv2.imwrite(f"result.jpg", image)
            else:
                print("Nothing found!")
    else:
        print("The image does not exist!")