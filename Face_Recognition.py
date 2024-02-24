import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


video_capture = cv2.VideoCapture(0)


elon_image = face_recognition.load_image_file("Students/Elon Musk.jpg")
elon_encoding = face_recognition.face_encodings(elon_image[0])

bill_image = face_recognition.load_image_file("Students/Bill Gates.jpg")
bill_encoding = face_recognition.face_encodings(bill_image[0])

jeff_image = face_recognition.load_image_file("Students/Jeff Bezos.jpg")
jeff_encoding = face_recognition.face_encodings(jeff_image[0])

known_face_encodings = [elon_encoding, bill_encoding, jeff_encoding]
known_face_names = ["Elon Musk", "Bill Gates", "Jeff Bezos"]


# в List of expected students
students = known_face_names.copy()
face_locations = []
face_encodings = []

# = Get the current date and time
now = datetime.now()
current_date = now.strftime("%y-%m-%d")
 # : Open the CSV file for writing
f = open(f"{current_date}.csv" , "wt", newline="")
lnwriter = csv.writer(f)
 # в start the infinite loop to process frames from the video capture device
while True:
    # Read a frame from the video capture device
    _, frame = video_capture.read()

    # Resize the frame to 1/4 of its original size for faster processing
    small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
     # D Recognise faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # II Process each face detected in the framel
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index  = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_face_names[best_match_index]

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerofText = (10, 10)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + "Present", bottomLeftCornerofText, font, fontScale, fontColor, thickness, lineType)
                # If the person is expected, remove them from the list of expected students
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

f.close()




















