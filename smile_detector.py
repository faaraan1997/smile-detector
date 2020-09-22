import cv2

# Face Classifier
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier(
    "haarcascade_smile.xml")

# Grab Webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read()

    # If there is an error, abort
    if not successful_frame_read:
        break

    # Convert frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the coordinates
    face_coordinates = face_detector.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get the sub frame (Using numpy dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # Convert frame to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Find all the smile in the face
        # for(x_, y_, w_, h_) in smiles:
        #     cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 2)

        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(225, 225, 225))

    # show the current frame
    cv2.imshow("Face and Smile Detector", frame)

    # Display
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()

print("Code Completed")
