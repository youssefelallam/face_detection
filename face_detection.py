import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
stream = cv2.VideoCapture(0)

while True:
    
    st, frame = stream.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)    
    faces = face_detect.detectMultiScale(gray_frame,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        face_only = frame[y:y+h,x:x+w]
    cv2.imshow("live stream", frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
