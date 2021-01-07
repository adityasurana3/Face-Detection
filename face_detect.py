import cv2

# Cascading classifiers are trained with several hundred "positive" sample views of a particular object and arbitrary "negative" images of the same size.
trained_face_data= cv2.CascadeClassifier('haarcascade.xml')
# to read image
# img = cv2.imread('RDJ.jpeg')

webcam=cv2.VideoCapture(0)
while True:
    successful_frame_read,frame = webcam.read()

    garyscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Detect Face
    face_coordinates=trained_face_data.detectMultiScale(garyscaled_img)
     # Draw Reactangle around Face 
    for (x,y,h,w) in face_coordinates:
        cv2.rectangle(frame ,(x,y),(x+h,y+w),(0,255,0),2)
    cv2.imshow("Face Detector",frame)
    key=cv2.waitKey(1)

    # Stop if Q is pressed
    if key==81 or key==113:
        break

webcam.release()
# Detect Face 
# # print(face_coordinates)
# # Draw Reactangle around Face 
# for (x,y,h,w) in face_coordinates:
#     cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),2)

# # to print image
# cv2.imshow("Face Detector",img)

# cv2.waitKey(1500)


print("code complete")