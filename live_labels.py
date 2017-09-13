#import the necessary modules
print("importing freenect")
import freenect
print("importing cv2")
import cv2
print("importing numpy")
import numpy as np
print("importing object detection")
import object_detect_demo
#import label_image
print("starting...")
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array


    

#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array



while 1:
    #get a frame from RGB camera
    frame = get_video()
    print("got video")
    #get a frame from depth sensor
    ###depth = get_depth()
    #display RGB image
    cv2.imshow('RGB image',frame)
    cv2.imwrite('live.png',frame)

    #label_image.classify_image('live.png')
    object_detect_demo.detect_objects('live.png')
    tagged_image = cv2.imread('tags.jpg')
    cv2.imshow("tagged image", tagged_image)
    #display depth image
    ###cv2.imshow('Depth image',depth)
 
    # quit program when 'esc' key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
