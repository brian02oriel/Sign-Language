import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# trigger tells us when to start recording
trigger = False

class_name = ''

# Counter keeps count of the number of samples collected
counter = 0

# This the ROI size, the size of images saved will be box_size -10
box_size = 240
    
# Getting the width of the frame from the camera properties
width = int(cap.get(3))

while(True):
    _, img = cap.read()
    img = cv2.flip(img, 1)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img, (width - box_size, 0), (width, box_size), (0, 0, 255), 2)

    if counter == 200:
            trigger = not trigger
            counter = 0

    if(trigger):
        # Grab only slected roi
        roi = img[5: box_size-5 , width-box_size + 5: width -5]
        roi = cv2.resize(roi, (200, 200))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('dataset/'+ class_name + '/' +str(counter) + '.jpg', roi)
        
        # Append the roi and class name to the list with the selected class_name
        #eval(class_name).append([roi, class_name])
                                
        # Increment the counter 
        counter += 1 
    
        # Text for the counter
        text = "Collected Samples of {}: {}".format(class_name, counter)
        cv2.putText(img, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        text1 = "Press 'a' to capture A samples, 'b' for B samples, 'c' for C samples"
        text2 = "Press 'd' to capture D samples, 'e' for E samples, 'f' for F samples"
        text3 = "Press 'n' for nothing and 'q' to quit"
        cv2.putText(img, text1, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text2, (3, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text3, (3, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    
    
    
    cv2.imshow('img', img)

    k = cv2.waitKey(1)

    # If user press 'a' save the examples for A gesture
    if k == ord('a'):
        trigger = not trigger
        class_name = 'A_gesture'
        
        
    # If user press 'b' save the examples for B gesture
    if k == ord('b'):
        trigger = not trigger
        class_name = 'B_gesture'
    
    # If user press 'c' save the examples for C gesture
    if k == ord('c'):
        trigger = not trigger
        class_name = 'C_gesture'
    
    # If user press 'c' save the examples for C gesture
    if k == ord('d'):
        trigger = not trigger
        class_name = 'D_gesture'
    
    # If user press 'c' save the examples for C gesture
    if k == ord('e'):
        trigger = not trigger
        class_name = 'E_gesture'
    
    # If user press 'c' save the examples for C gesture
    if k == ord('f'):
        trigger = not trigger
        class_name = 'F_gesture'
                
    # If user press 'n' save the examples for nothing
    if k == ord('n'):
        trigger = not trigger
        class_name = 'nothing'

    if(k == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
