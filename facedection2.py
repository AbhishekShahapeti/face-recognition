#from firebase import firebase
import face_recognition
import numpy as np
import cv2
import csv
video_capture = cv2.VideoCapture(0)

#opening csv file
with open('data - data.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

user_image=[]
user_face_encoding=[]
known_face_names=[]
known_face_encodings =[]

#loading image file and comparing it
for i in range(len(data)):
    image=face_recognition.load_image_file("Data/"+data[i][1])
    user_image.append(image)
    encoding=face_recognition.face_encodings(image)[0]
    user_face_encoding.append(encoding)

    known_face_names.append(data[i][0])
    for e in user_face_encoding:
        known_face_encodings.append(e)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Random Person"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if (name != "RandomPerson"):
                print(name, "was here")
        cv2.imshow('Video', frame)

        # Q to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break