import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("haarcascade_eye.xml")
stream = cv2.VideoCapture(0)

while True:
    
    st, frame = stream.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in faces:
        face_only = frame[y:y+h,x:x+w]

        eyes = eye_detect.detectMultiScale(face_only,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_only,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    cv2.imshow("live stream", frame)
    if cv2.waitKey(10) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()