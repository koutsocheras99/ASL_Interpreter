import cv2
import numpy as np
import os

camera = cv2.VideoCapture(0)   # 0 -> index of camera

i = 0 # numOfPhotos

# fgBackGround = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

camera.set(3,1080) # width=640
camera.set(4,720) # height=480


while True:

    check, frame1 = camera.read() 

    cv2.rectangle(frame1, (20, 70), (275, 325), (0,255,0), thickness=1) # create rectangle at the main frame

    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) # turning main frame to gray

    median_blurred = cv2.medianBlur(gray_frame, 3)  # blurring (median) the gray frame

    gaus_blurred = cv2.GaussianBlur(median_blurred,(3,3),0) # blurring (gaussian) the already blurred gray frame

    canny_frame = cv2.Canny(gaus_blurred,30,30)

    cropped_frame = canny_frame[71:325, 21:275] # cropped to match the rectangle + cropped will be threshed | test with grayscale

    if not check:
        break
    
    # fgMask = fgBackGround.apply(frame1) # applying the subtractor to frame1

    cv2.imshow('WINDOW', frame1)
    cv2.imshow('CROPPED WINDOW', cropped_frame)

    if frame1 is None:
        break

    keyboard = cv2.waitKey(1) & 0xff

    if keyboard == ord('n'): # if n(ext) pressed capture,save and move on
        filename = 'file_%i.jpg'%i
        cv2.imwrite(os.path.join('NewDataSet/DELETE', filename), cropped_frame) # save at the dataset folder 
        i+=1
        
    elif keyboard == 27: # esc to terminate
        break
        
    
camera.release()
cv2.destroyAllWindows()

