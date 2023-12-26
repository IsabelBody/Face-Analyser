
# Press Shift+F10 to execute


import cv2 as cv # computer vision class
from deepface import DeepFace
# cv identifies objects like faces
# deepface analyses faces


detect = input("What do you want to detect?: ")

# pretrained classifier specifically for faces, provided by OpenCV, this is the link to my files.
# you can change this to anything you like.
provided_classifier = (cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# provides a cascade class object for identifying objects which are faces.
face_identifier = cv.CascadeClassifier(provided_classifier)


# 0 represents the default camera (usually the built-in webcam)
video = cv.VideoCapture(0)
# Alternatively, you can specify a video file path -> video = cv.VideoCapture('video.mp4')

if not video.isOpened():
    raise IOError("Cannot open webcam")

# while webcam is open.
while True:
    # .read() gets the next frame
    returning, frame = video.read()
    # frame is an object representing the image data
    # of that still.

    # returning is true or false
    # - video is or is not still playing.

    # converting the frame to grayscale. This makes the
    # detecting process computationally less expensive.
    grayscaled_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # getting a list of rectangles that are likely faces.
    # <class 'numpy.ndarray'> [[284  94 209 209]] x,y of top left corner, width, height
    # minneighbours is the amount of other rectangles that also identified the object as a face.
    face_rectangles_coords = face_identifier.detectMultiScale(grayscaled_frame)

    # for every face in identified faces, get the coords of the rectangle.
    for x, y, width, height in face_rectangles_coords:

        ''' Parameters of rectangle method.
        img: The image on which the rectangle is drawn.
        pt1: Vertex of the rectangle. It is represented as a tuple of (x, y).
        pt2: Vertex of the rectangle opposite to pt1. Also represented as a tuple of (x, y).
        color: Color of the rectangle. It can be a tuple (B, G, R) or a scalar value. The default is white (255, 255, 255).
        thickness: Thickness of the rectangle border. If it is negative, the rectangle will be filled. The default is 1.
        '''

        # (89, 2, 236) is in BGR color formate. use BGR color picker.
        image = cv.rectangle(frame, (x, y), (x + width, y + height), (156, 71, 247))
        try:
            # use deepface dataset to analyse the emotion
            # or action=['race']. do not leave all as it's too computationally expensive.

            analyzed_face = DeepFace.analyze(frame, actions=[detect])

            # putText(image, text to display, origin point for placing text, font face, font scale, color
            # thickness, linetype)
            cv.putText(image, analyzed_face[0]['dominant_'+detect], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (225,225,225))
        except Exception as e:
            print(e)

    # show image in a video window.
    # video sets the name of the display window.
    # this can be changed to anything.
    window_name = 'video'
    cv.imshow(window_name, frame)

    # waiting for a key.
    key = cv.waitKey(1)
    # comparing if its the right key
    # end program if esc or q or window is closed
    if key == 27 or key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
        break


# important to release the resources!
video.release()
cv.destroyAllWindows()
