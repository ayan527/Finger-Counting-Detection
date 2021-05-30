# Importing starting libraries
import cv2
import numpy as np
import math
from sklearn.metrics import pairwise


# Initalize gloabl variables
backgroundScreen = None
accumulated_weight = 0.5


# Initialize ROI coordinates
roi_top = 50
roi_bottom = 300

roi_right = 600
roi_left = 380


## Calculating background accumulated weight avg ##
def calc_bg_accum_avg(frame,accumulated_weight):
    # Define global variable
    global backgroundScreen
    
    # Check if Background Screen initialized
    if backgroundScreen is None:
        backgroundScreen = frame.copy().astype('float') # Initalize background screen
        return None
    
    # Else updates the accumulated weight avergae for the Background Screen
    cv2.accumulateWeighted(src=frame,dst=backgroundScreen,alpha=accumulated_weight)

    
## Monitoring the Hand Segment ##
def hand_segment(frame,threshold=25):
    # Define global variable
    global backgroundScreen
    
    # Calculating absolute difference between backgroundScreen & current frame
    diff_frame = cv2.absdiff(backgroundScreen.astype('uint8'),frame)
    
    # Applying thresholding to extract hand-segment(foreground-white) from background screen(black)
    ret, segment_frame = cv2.threshold(diff_frame,threshold,255,cv2.THRESH_BINARY)
    
    # Finding external contours of hand
    ext_contours,hierarchy = cv2.findContours(segment_frame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if countour set is empty or not
    if len(ext_contours) == 0:
        return None
    else:
        # Get the larget external contour - contour of hand segment
        hand_contour = max(ext_contours,key=lambda x : cv2.contourArea(x))
        return (segment_frame,hand_contour)

## Counting number of fingers ##
def count_fingers(segment_frame,hand_contour):
    convex_hull = cv2.convexHull(hand_contour)
    
    area_hull = cv2.contourArea(convex_hull)
    area_contours = cv2.contourArea(hand_contour)
    area_ratio = ((area_hull-area_contours)/area_contours)*100
    
    convex_hull = cv2.convexHull(hand_contour,returnPoints=False)
    convexity_defects = cv2.convexityDefects(hand_contour,convex_hull)
    
    if convexity_defects is not None:
        finger_counter = 0
        for i in range(convexity_defects.shape[0]):  # calculate the angle
            s, e, f, d = convexity_defects[i][0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])

            a = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            
            angle = np.arccos((b ** 2 + a ** 2 - c ** 2) / (2 * b * a))
            
            if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                finger_counter += 1
        if finger_counter == 0:
            if area_ratio > 12:
                finger_counter += 1
        elif finger_counter > 0:
            finger_counter += 1
        
        return finger_counter

    
## Running the code ##
#Initialize Video Capture
capture = cv2.VideoCapture(0)

# Initialize frame counter
frame_counter = 0

while True:
    ret,frame = capture.read()
    
    if not ret:
        break
    
    # Flip the frame
    frame = cv2.flip(frame, 1)
    # Clone the frame
    frame_copy = frame.copy()
    
    # Grab hand-roi
    hand_roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    
    # Preprocess the roi
    hand_roi_gray = cv2.cvtColor(hand_roi,cv2.COLOR_BGR2GRAY)
    hand_roi_gray = cv2.GaussianBlur(hand_roi_gray,(9,9),0)
    
    # Wait for 60 frames to calculate background screen accumularted weight avg
    if frame_counter < 60:
        calc_bg_accum_avg(hand_roi_gray,accumulated_weight)
        if frame_counter <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)         
    else:
        hand = hand_segment(hand_roi_gray)
        
        if hand is not None:
            segment_frame,hand_contour = hand
            
            # Drawing contours on hand
            cv2.drawContours(frame_copy,[hand_contour + (roi_left,roi_top)],-1,(255,0,0),2)
            
            # Calculate fingers
            finger_count = count_fingers(segment_frame,hand_contour)
            
            # Display count
            cv2.putText(frame_copy,'Finger Count: ' + str(finger_count), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Display the background image
            cv2.imshow("Background", segment_frame)
        
        # Displaying rectangular roi
        cv2.putText(frame_copy,'Put your hand here',(roi_left - 20,roi_top - 20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,255,0),5)
        
    # Incrementing number of frames
    frame_counter += 1
        
    # Display original frame
    #cv2.putText(frame_copy,'Frame Count: ' +str(frame_counter),(10,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Finger Count', frame_copy)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()