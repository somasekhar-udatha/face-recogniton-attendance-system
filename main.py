#pip install face_recognition
#pip install opencv-python
#pip install cmake
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

vid_capture = cv2.VideoCapture(0) # uses webcam to capture video we have to change the number according to the webcam we are using 
#faces folder contains pre loaded faces or pictures
# loading known faces
soma_image = face_recognition.load_image_file("faces/soma.jpg")#we have to make face encododings for these images (converting into numerical form whcih makes comparision easier)
soma_encoding = face_recognition.face_encodings(soma_image)[0]#it returns a list i.e if there are 10 faces in returns list of 10 face encodings
srk_image = face_recognition.load_image_file("faces/srk.jpg")# or we can use png
srk_encoding = face_recognition.face_encodings(srk_image)[0]
mahesh_image = face_recognition.load_image_file("faces/mahesh.jpg")# or we can use png
mahesh_encoding = face_recognition.face_encodings(mahesh_image)[0]

# now we store these encodings
known_face_encodings = [soma_encoding,srk_encoding,mahesh_encoding]
known_face_names = ["Soma Sekhar","Shah Rukh Khan","Mahesh"]

#list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

#get current date and time 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

#storing these in a csv file
f = open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = vid_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx = 0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_face_names[best_match_index]

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale = 1.5
                fontColor = (255,0,0)
                thickness = 3
                lineType = 2
                cv2.putText(frame,name+" Present",bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid_capture.release()
cv2.destroyAllWindows()
f.close()