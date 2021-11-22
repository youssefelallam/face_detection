import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detect = cv2.CascadeClassifier("haarcascade_smile.xml")
stream = cv2.VideoCapture(0)

while True:
    
    st, frame = stream.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in faces:
        face_only = frame[y:y+h,x:x+w]

        smiles = smile_detect.detectMultiScale(face_only,1.3,10)
        for (sx,sy,sw,sh) in smiles:
             cv2.rectangle(face_only,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)

    cv2.imshow("live stream", frame)
    if cv2.waitKey(10) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()