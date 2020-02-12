import cv2 as cv
import face_recognition as fr
from PIL import Image

danImage = fr.load_image_file("./faceDatabase/dan.jpg")

danFace = fr.face_encodings(danImage)[0]

locations = []

encodings = []

faces = []

savedFaces = [danFace]

webcam = cv.VideoCapture(0)

if not webcam.isOpened():

    raise IOError("Webcam input failed.")

else:

    while True:
        r, f = webcam.read()
        condensed = cv.resize(f, (0, 0), fx=0.25, fy=0.25)
        RGB = condensed[:, :, ::-1]
        locations = fr.face_locations(RGB)
        encodings = fr.face_encodings(RGB, locations)
        faces = []

        for e in encodings:
            faceMatch = fr.compare_faces(savedFaces, e)
            identity = ""

            if faceMatch[0]:
                identity = "Daniel"
                faces.append(identity)
                locations = fr.face_locations(RGB)
                print("Daniel's face detected!")

            elif faceMatch[1]:
                identity = "friend"
                faces.append(identity)
                locations = fr.face_locations(RGB)
                print("Friendly face detected!")

            else:
                identity = "unknown"
                faces.append(identity)
                locations = fr.face_locations(RGB)
                print("Unknown face detected!")

        for (top, right, bottom, left), identity in zip(locations, faces):

            if identity is "unknown":
                unknown_face = RGB[top:bottom, left:right]
                new_face = Image.fromarray(unknown_face)
                new_face.save(f'./unknownFaces/detectedFace{top}.jpg')
                cv.rectangle(condensed, (left, top), (right, bottom), (255, 255, 255), 1)
                f = cv.FONT_HERSHEY_COMPLEX
                cv.putText(condensed, identity, (left + 10, bottom - 10), f, .5, (255, 255, 255), 1)

            else:
                cv.rectangle(condensed, (left, top), (right, bottom), (0, 255, 0), 1)
                f = cv.FONT_HERSHEY_COMPLEX
                cv.putText(condensed, identity, (left + 10, bottom - 10), f, .5, (0, 255, 0), 1)
        cv.imshow('LIVE_FEED', condensed)
        cv.waitKey(60)

webcam.release()
cv.destroyAllWindows()
