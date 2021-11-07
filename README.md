# Face-Manipulation
Manipulate size of the nose and the left eye

You can download the pre-trained model file from <a href="https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat">Here</a>. (put the file in the model directory)
<br>
>Face-Manipulation/model/shape_predictor_68_face_landmarks.dat
# Requirements
> pip install -r requirements.txt
# Run
> python3 app.py --image face.jpg --threshold -20

# Result
<img src="face.jpg" width=300><img src="https://static.vecteezy.com/ti/gratis-vektor/t2/553925-pfeilsymbol-kostenlos-vektor.jpg" width=50><img src="result.jpg" width=300>
-----------------------------------------------
<a href="https://unsplash.com/photos/KbBztc5PTC8">Image Source</a>

# The Distortion Method
For resizing the part of the image I used <a href="https://legacy.imagemagick.org/Usage/distorts/#shepards">Shepard's distortion</a> which is available to use in the wand package as a distortion method.

