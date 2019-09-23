import cv2 as cv
import face_recognition as fr
from PIL import Image

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

danImage = fr.load_image_file("./faceDatabase/dan.jpg")
danFace = fr.face_encodings(danImage)[0]

haoImage = fr.load_image_file("./faceDatabase/hao.jpg")
haoFace = fr.face_encodings(haoImage)[0]

savedFaces = [danFace, haoFace, ]

webcam = cv.VideoCapture(0)

if not webcam.isOpened():

    raise IOError("Webcam input failed.")

else:

    while True:
        ret, frame = webcam.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            faceMatch = fr.compare_faces(savedFaces, face_encoding)
            name = ""

            if faceMatch[0]:
                name = "Daniel"
                face_names.append(name)
                face_locations = fr.face_locations(rgb_frame)

            elif faceMatch[1]:
                name = "friend"
                face_names.append(name)
                face_locations = fr.face_locations(rgb_frame)

            else:
                name = "unknown"
                face_names.append(name)
                face_locations = fr.face_locations(rgb_frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if name is "unknown":
                unknown_face = frame[top:bottom, left:right]
                new_face = Image.fromarray(unknown_face)
                new_face.save(f'./unknownFaces/detectedFace{top}.jpg')
                cv.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
                font = cv.FONT_HERSHEY_DUPLEX
                cv.putText(frame, name, (left + 10, bottom - 10), font, .5, (255, 255, 255), 1)

            else:
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
                font = cv.FONT_HERSHEY_COMPLEX
                cv.putText(frame, name, (left + 10, bottom - 10), font, .5, (0, 255, 0), 1)
        cv.imshow('LIVE_FEED', frame)
        cv.waitKey(1)

webcam.release()
cv.destroyAllWindows()