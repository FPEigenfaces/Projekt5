import cv2


def cascade_detection(images):
    # print(images)
    haarcascade_eye_detection = cv2.CascadeClassifier(
         './resources/cascade_detection/haarcascade_profileface.xml')
    for img in images:

        face = haarcascade_eye_detection.detectMultiScale(img)
        resized_img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
        if len(face)>0 :

            for x,y,w,h in face:
                cv2.rectangle(resized_img, (x,y), (x+w, x+h), (100,100,255),5)

            cv2.imshow("Face Detection", resized_img)
            cv2.waitKey(100)
            
            
              
